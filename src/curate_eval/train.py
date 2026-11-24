from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .io import read_examples


def format_example(example) -> str:
    return format_example_fields(example.instruction, example.input, example.output)


def format_prompt(example) -> str:
    return format_prompt_fields(example.instruction, example.input)


def format_example_fields(instruction: str, context: str, output: str) -> str:
    return f"{format_prompt_fields(instruction, context)}{output}"


def format_prompt_fields(instruction: str, context: str) -> str:
    context_text = f"\nContext: {context}" if context else ""
    return f"### Instruction:\n{instruction}{context_text}\n\n### Response:\n"


def train(data_path: str, model_name: str, output_dir: str, seed: int, epochs: float,
          learning_rate: float, batch_size: int, max_length: int, max_steps: int = -1,
          gradient_accumulation_steps: int = 4) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    if max_length <= 0 or max_steps == 0 or max_steps < -1 or gradient_accumulation_steps <= 0:
        raise ValueError("max_length and gradient_accumulation_steps must be positive; max_steps must be -1 or positive")
    examples = read_examples(data_path)
    if not examples:
        raise ValueError(f"No training examples in {data_path}")
    dataset = Dataset.from_dict({
        "instruction": [example.instruction for example in examples],
        "input": [example.input for example in examples],
        "output": [example.output for example in examples],
    })
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    adapter = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, adapter)

    def tokenize_batch(batch):
        texts = []
        prompts = []
        for instruction, context, output in zip(batch["instruction"], batch["input"], batch["output"]):
            example_text = format_example_fields(instruction, context, output)
            texts.append(example_text + (tokenizer.eos_token or ""))
            prompts.append(format_prompt_fields(instruction, context))
        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        prompt_tokens = tokenizer(prompts, truncation=True, padding=False, max_length=max_length)["input_ids"]
        labels = []
        for input_ids, attention_mask, prompt_ids in zip(
                tokenized["input_ids"], tokenized["attention_mask"], prompt_tokens):
            label = list(input_ids)
            prompt_length = min(len(prompt_ids), len(label))
            label[:prompt_length] = [-100] * prompt_length
            label = [-100 if mask == 0 else token for token, mask in zip(label, attention_mask)]
            if all(token == -100 for token in label):
                raise ValueError("An example has no answer tokens within --max-length")
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataloader_pin_memory=use_cuda,
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
    )
    batch_token_counts: list[tuple[int, int, int]] = []

    def collate_and_count(features):
        batch = default_data_collator(features)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            batch_token_counts.append((
                len(features),
                int(attention_mask.sum().item()),
                int(attention_mask.numel()),
            ))
        return batch

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=collate_and_count,
    )
    started = time.perf_counter()
    trainer.train()
    runtime_seconds = time.perf_counter() - started
    consumed_batches = min(
        len(batch_token_counts),
        trainer.state.global_step * gradient_accumulation_steps,
    )
    consumed_counts = [
        sum(batch[index] for batch in batch_token_counts[:consumed_batches]) * args.world_size
        for index in range(3)
    ]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    training_metrics = {
        "device": str(args.device),
        "model_dtype": str(next(trainer.model.parameters()).dtype),
        "mixed_precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps * args.world_size,
        "max_length": max_length,
        "max_steps": max_steps,
        "steps_completed": trainer.state.global_step,
        "examples_seen": consumed_counts[0],
        "non_padding_tokens_seen": consumed_counts[1],
        "padded_token_positions_seen": consumed_counts[2],
        "training_runtime_seconds": runtime_seconds,
        "non_padding_tokens_per_second": consumed_counts[1] / runtime_seconds,
        "padded_token_positions_per_second": consumed_counts[2] / runtime_seconds,
    }
    metrics_path = Path(output_dir) / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(training_metrics, handle, indent=2)
    print(json.dumps(training_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fixed LoRA SFT configuration.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1560)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    train(args.data, args.model, args.output_dir, args.seed, args.epochs,
          args.learning_rate, args.batch_size, args.max_length, args.max_steps,
          args.gradient_accumulation_steps)


if __name__ == "__main__":
    main()
