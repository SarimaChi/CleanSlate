from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .filters import FilterConfig, curate, deduplicate_examples
from .io import read_examples, write_examples
from .synthetic import generate


def _run_lm_eval_on_cpu(
    base_model: str,
    adapter_dir: Path | None,
    task_dir: Path,
    output_path: Path,
) -> None:
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, device="cpu", batch_size=1)
    results = simple_evaluate(
        model=lm,
        tasks=["curate_eval"],
        task_manager=TaskManager(include_path=[str(task_dir)]),
        batch_size=1,
        bootstrap_iters=0,
        log_samples=False,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=handle_non_serializable, ensure_ascii=False)


def run_lm_eval(model_dir: Path, test_data: str, output_path: Path,
                max_new_tokens: int = 256) -> None:
    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    executable = shutil.which("lm_eval")
    if executable is None:
        raise RuntimeError("lm_eval is not installed; install the 'harness' extra: pip install -e '.[harness]'")
    task_source = Path(__file__).resolve().parents[2] / "lm_eval" / "tasks" / "curate_eval" / "task.yaml"
    if not task_source.is_file():
        raise FileNotFoundError(f"Custom lm-evaluation-harness task config not found: {task_source}")
    task_config = task_source.read_text(encoding="utf-8")
    task_config = task_config.replace(
        "test: data/test.jsonl",
        f'test: "{Path(test_data).resolve()}"',
    )
    task_config = task_config.replace("max_gen_toks: 256", f"max_gen_toks: {max_new_tokens}")
    with tempfile.TemporaryDirectory(prefix="curate-eval-task-") as task_dir:
        task_dir_path = Path(task_dir)
        task_file = task_dir_path / "curate_eval.yaml"
        task_file.write_text(task_config, encoding="utf-8")
        base_model = str(model_dir.resolve())
        adapter_config = model_dir / "adapter_config.json"
        adapter_dir = None
        if adapter_config.is_file():
            with adapter_config.open(encoding="utf-8") as handle:
                base_model = json.load(handle)["base_model_name_or_path"]
            adapter_dir = model_dir.resolve()

        if torch.cuda.is_available():
            device = "cuda"
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            model_args = f"pretrained={base_model},device={device},dtype={dtype}"
            if adapter_dir is not None:
                model_args += f",peft={adapter_dir}"
            subprocess.run([
                executable,
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", "curate_eval",
                "--include_path", str(task_dir_path),
                "--output_path", str(output_path.resolve()),
                "--batch_size", "auto",
            ], check=True)
        else:
            _run_lm_eval_on_cpu(base_model, adapter_dir, task_dir_path, output_path)


def prepare_filtered_synthetic(
    filtered_path: str,
    combined_path: str,
    candidates_path: str,
    count: int,
    judge_url: str,
    judge_model: str,
    *,
    language: str = "",
    trust_language_labels: bool = False,
    eval_path: str = "",
    judge_api_key_env: str = "OPENAI_API_KEY",
) -> dict[str, int]:
    seeds = read_examples(filtered_path)
    if not seeds:
        raise ValueError(f"No seed examples found in {filtered_path}")
    language_labels = {example.language for example in seeds if example.language}
    if not language and len(language_labels) > 1:
        raise ValueError("Seed examples have mixed language labels; pass --language explicitly")
    if not language and len(language_labels) == 1:
        language = next(iter(language_labels))
    elif not language:
        language = "en"

    combined_file = Path(combined_path)
    if combined_file.is_file():
        combined = read_examples(combined_file)
        if not any(example.metadata.get("synthetic") for example in combined):
            raise ValueError(
                f"{combined_path} exists but contains no synthetic examples; "
                "choose another --filtered-synthetic path"
            )
        seed_ids = {example.source_id for example in seeds if example.source_id}
        combined_ids = {example.source_id for example in combined if example.source_id}
        if seed_ids and not seed_ids.issubset(combined_ids):
            raise ValueError(
                f"{combined_path} does not contain all filtered seed records; "
                "choose another --filtered-synthetic path"
            )
        return {"seed_examples": len(seeds), "combined_examples": len(combined), "generated": 0}

    if count <= 0:
        raise ValueError("--synthetic-count must be positive when generating the combined dataset")
    if not judge_url or not judge_model:
        raise ValueError(
            "No filtered-plus-synthetic dataset exists. Generating and judging one requires "
            "--judge-url and --judge-model (and the configured API key environment variable)."
        )

    candidate_file = Path(candidates_path)
    if candidate_file.is_file():
        candidates = read_examples(candidate_file)
        if len(candidates) != count:
            raise ValueError(
                f"Found {len(candidates)} cached synthetic candidates in {candidates_path}, "
                f"but --synthetic-count is {count}. Match the count or choose another --synthetic-candidates path."
            )
        if not all(example.metadata.get("synthetic") for example in candidates):
            raise ValueError(f"Cached candidates in {candidates_path} are not marked synthetic")
    else:
        generation_config = FilterConfig(
            language=language,
            judge_url=judge_url,
            judge_model=judge_model,
            judge_api_key_env=judge_api_key_env,
        )
        candidates = generate(seeds, generation_config, count, language=language)
        write_examples(candidate_file, candidates)

    filter_config = FilterConfig(
        language=language,
        judge_url=judge_url,
        judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        eval_path=eval_path,
        trust_language_labels=trust_language_labels,
    )
    kept_synthetic, filter_report = curate(candidates, filter_config)
    if not kept_synthetic:
        raise ValueError("Synthetic filtering retained zero examples; inspect the filter report before training")

    combined = deduplicate_examples(
        [*seeds, *kept_synthetic],
        threshold=filter_config.minhash_threshold,
        permutations=filter_config.minhash_permutations,
    )
    if not any(example.metadata.get("synthetic") for example in combined):
        raise ValueError("All synthetic examples were duplicates of seed data")
    write_examples(combined_file, combined)
    report_path = combined_file.with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "seed_examples": len(seeds),
            "generated_candidates": len(candidates),
            "retained_synthetic": len(kept_synthetic),
            "combined_examples": len(combined),
            "filters": filter_report,
        }, handle, indent=2)
    return {
        "seed_examples": len(seeds),
        "generated": len(candidates),
        "retained_synthetic": len(kept_synthetic),
        "combined_examples": len(combined),
    }


def run(
    data_paths: dict[str, str],
    model: str,
    output_dir: str,
    *,
    test_data: str = "",
    judge_url: str = "",
    judge_model: str = "",
    judge_api_key_env: str = "OPENAI_API_KEY",
    synthetic_count: int = 500,
    synthetic_candidates: str = "data/synthetic_candidates.jsonl",
    language: str = "",
    trust_language_labels: bool = False,
    run_harness: bool = False,
    seed: int = 17,
    epochs: float = 1.0,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    max_length: int = 1560,
    max_steps: int = -1,
    gradient_accumulation_steps: int = 4,
    max_new_tokens: int = 256,
) -> None:
    for arm in ("raw", "filtered"):
        if arm not in data_paths or not Path(data_paths[arm]).is_file():
            raise FileNotFoundError(f"Required {arm} training dataset not found: {data_paths.get(arm, '')}")
    if test_data and not Path(test_data).is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {test_data}")
    if run_harness and not test_data:
        raise ValueError("--run-lm-eval requires --test-data")
    if run_harness and shutil.which("lm_eval") is None:
        raise RuntimeError("lm_eval is not installed; install the 'harness' extra: pip install -e '.[harness]'")
    if (
        epochs <= 0
        or learning_rate <= 0
        or batch_size <= 0
        or max_length <= 0
        or max_steps == 0
        or max_steps < -1
        or gradient_accumulation_steps <= 0
        or max_new_tokens <= 0
    ):
        raise ValueError(
            "epochs, learning_rate, batch_size, max_length, and gradient_accumulation_steps "
            "must be positive; max_steps must be -1 or positive; max_new_tokens must be positive"
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = dict(data_paths)
    synthetic_counts = prepare_filtered_synthetic(
        paths["filtered"],
        paths["filtered_synthetic"],
        synthetic_candidates,
        synthetic_count,
        judge_url,
        judge_model,
        language=language,
        trust_language_labels=trust_language_labels,
        eval_path=test_data,
        judge_api_key_env=judge_api_key_env,
    )

    training_config = {
        "base_model": model,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "max_length": max_length,
        "max_steps": max_steps,
        "max_new_tokens": max_new_tokens,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    results = {}
    data_counts = {}
    training_diagnostics = {}
    for name in ("raw", "filtered", "filtered_synthetic"):
        data_path = paths[name]
        data_counts[name] = len(read_examples(data_path))
        arm_dir = root / name
        model_dir = arm_dir / "adapter"
        subprocess.run([
            sys.executable, "-m", "curate_eval.train",
            "--data", data_path,
            "--model", model,
            "--output-dir", str(model_dir),
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--max-length", str(max_length),
            "--max-steps", str(max_steps),
            "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        ], check=True)
        training_metrics_path = model_dir / "training_metrics.json"
        if training_metrics_path.is_file():
            with training_metrics_path.open(encoding="utf-8") as handle:
                training_diagnostics[name] = json.load(handle)
        if test_data:
            metrics_path = arm_dir / "metrics.json"
            command = [
                sys.executable, "-m", "curate_eval.evaluate",
                "--model", str(model_dir),
                "--test-data", test_data,
                "--output", str(metrics_path),
                "--max-new-tokens", str(max_new_tokens),
            ]
            if judge_url:
                command.extend(["--judge-url", judge_url, "--judge-model", judge_model])
            subprocess.run(command, check=True)
            with metrics_path.open(encoding="utf-8") as handle:
                results[name] = json.load(handle)["metrics"]
            if run_harness:
                harness_path = arm_dir / "lm_eval_results.json"
                run_lm_eval(model_dir, test_data, harness_path, max_new_tokens)
                results[name]["lm_eval_results"] = str(harness_path)

    summary = {
        "training_config": training_config,
        "dataset_counts": data_counts,
        "training_diagnostics": training_diagnostics,
        "synthetic_augmentation": synthetic_counts,
        "results": results,
    }
    with (root / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched raw/filtered/synthetic LoRA ablations.")
    parser.add_argument("--raw", default="data/raw.jsonl")
    parser.add_argument("--filtered", default="data/filtered.jsonl")
    parser.add_argument("--filtered-synthetic", default="data/filtered_plus_synthetic.jsonl")
    parser.add_argument("--synthetic-candidates", default="data/synthetic_candidates.jsonl")
    parser.add_argument("--synthetic-count", type=int, default=500)
    parser.add_argument("--language", default="")
    parser.add_argument("--trust-language-labels", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--test-data", default="")
    parser.add_argument("--run-lm-eval", action="store_true")
    parser.add_argument("--output-dir", default="runs/ablation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1560)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    run(
        {
            "raw": args.raw,
            "filtered": args.filtered,
            "filtered_synthetic": args.filtered_synthetic,
        },
        args.model,
        args.output_dir,
        test_data=args.test_data,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        judge_api_key_env=args.judge_api_key_env,
        synthetic_count=args.synthetic_count,
        synthetic_candidates=args.synthetic_candidates,
        language=args.language,
        trust_language_labels=args.trust_language_labels,
        run_harness=args.run_lm_eval,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
