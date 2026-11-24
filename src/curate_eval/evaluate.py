from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .filters import FilterConfig
from .io import read_examples
from .judge import call_judge
from .schema import Example
from .train import format_example, format_prompt, format_prompt_fields


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions and references must be non-empty and have equal length")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(rouge.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(predictions, references)) / len(predictions)
    bleu = BLEU().corpus_score(predictions, [references]).score
    return {"rougeL": rouge_l, "bleu": bleu}


def generate(model_path: str, examples: list[Example], max_new_tokens: int) -> tuple[list[str], float]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not examples:
        raise ValueError("Evaluation data contains no examples")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.is_file():
        with adapter_config.open(encoding="utf-8") as handle:
            base_model = json.load(handle)["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    predictions: list[str] = []
    total_negative_log_likelihood = 0.0
    total_response_tokens = 0
    model.eval()
    with torch.no_grad():
        for example in examples:
            prompt = format_prompt(example)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            predictions.append(tokenizer.decode(generated[0][encoded.input_ids.shape[1]:],
                                                skip_special_tokens=True).strip())
            full = tokenizer(format_example(example) + (tokenizer.eos_token or ""), return_tensors="pt",
                             truncation=True, max_length=1024).to(device)
            prompt_length = min(
                len(tokenizer(format_prompt_fields(example.instruction, example.input),
                              truncation=True, max_length=1024)["input_ids"]),
                full["input_ids"].shape[1],
            )
            labels = full["input_ids"].clone()
            labels[:, :prompt_length] = -100
            response_tokens = int((labels != -100).sum())
            if response_tokens == 0:
                raise ValueError("An evaluation answer was truncated before any target tokens remained")
            loss = model(**full, labels=labels).loss
            total_negative_log_likelihood += float(loss) * response_tokens
            total_response_tokens += response_tokens
    perplexity = math.exp(total_negative_log_likelihood / total_response_tokens)
    return predictions, perplexity


def judge_win_rate(predictions: list[str], references: list[str], examples: list[Example],
                   config: FilterConfig) -> float:
    if not config.judge_url or not config.judge_model:
        raise ValueError("LLM win-rate requires --judge-url and --judge-model")
    if not examples or len(predictions) != len(references) or len(predictions) != len(examples):
        raise ValueError("Predictions, references, and examples must be non-empty and have equal length")
    wins = 0.0
    for index, (example, prediction, reference) in enumerate(zip(examples, predictions, references)):
        model_is_a = index % 2 == 0
        candidate_a, candidate_b = (prediction, reference) if model_is_a else (reference, prediction)
        prompt = (
            "Compare candidate A and candidate B for correctness, relevance, and clarity. "
            "Return exactly A, B, or tie. Candidate order alternates between examples.\n"
            f"Instruction: {example.instruction}\nA: {candidate_a}\nB: {candidate_b}"
        )
        response = call_judge(prompt, config, max_tokens=10)
        label = re.match(r"\s*(A|B|TIE)\b", response, flags=re.IGNORECASE)
        if label is None:
            raise ValueError(f"Judge returned an invalid win-rate label: {response!r}")
        winning_candidate = label.group(1).upper()
        if winning_candidate == ("A" if model_is_a else "B"):
            wins += 1.0
        elif winning_candidate == "TIE":
            wins += 0.5
    return wins / len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a trained model on held-out instruction data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-url", default="")
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args()
    examples = read_examples(args.test_data)
    predictions, perplexity = generate(args.model, examples, args.max_new_tokens)
    references = [example.output for example in examples]
    metrics = {**score(predictions, references), "perplexity": perplexity}
    if args.judge_url:
        metrics["judge_win_rate"] = judge_win_rate(
            predictions, references, examples,
            FilterConfig(judge_url=args.judge_url, judge_model=args.judge_model),
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "predictions": predictions, "references": references}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
