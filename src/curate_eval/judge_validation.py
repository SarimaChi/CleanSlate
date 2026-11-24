from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .filters import FilterConfig
from .io import read_examples
from .judge import score_example
from .schema import Example


def export_labels(input_path: str, output_path: str, sample_size: int, seed: int) -> int:
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    examples = read_examples(input_path)
    rng = random.Random(seed)
    selected = rng.sample(examples, min(sample_size, len(examples)))
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for index, example in enumerate(selected):
            handle.write(json.dumps({
                "sample_id": index,
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
                "human_label": None,
            }, ensure_ascii=False) + "\n")
    return len(selected)


def cohen_kappa(human: list[int], judge: list[int]) -> float:
    if len(human) != len(judge) or not human:
        raise ValueError("Need equal, non-empty human and judge label lists")
    if any(label not in (0, 1) for label in human + judge):
        raise ValueError("Labels must be binary: 0=reject, 1=keep")
    observed = sum(left == right for left, right in zip(human, judge)) / len(human)
    human_positive = sum(human) / len(human)
    judge_positive = sum(judge) / len(judge)
    expected = human_positive * judge_positive + (1 - human_positive) * (1 - judge_positive)
    if expected == 1:
        raise ValueError("Cohen's kappa is undefined when expected agreement is 1")
    return (observed - expected) / (1 - expected)


def score_labeled(path: str, config: FilterConfig, threshold: float) -> dict:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(row.get("human_label") not in (0, 1) for row in rows):
        raise ValueError("Every row must have human_label 0 or 1 before scoring")
    human = [int(row["human_label"]) for row in rows]
    judge = [
        int(score_example(Example.from_dict(row), config) >= threshold)
        for row in rows
    ]
    return {"n": len(rows), "threshold": threshold, "cohen_kappa": cohen_kappa(human, judge)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human labels or measure judge agreement.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--sample-size", type=int, default=150)
    export.add_argument("--seed", type=int, default=17)
    score = subparsers.add_parser("score")
    score.add_argument("--labels", required=True)
    score.add_argument("--judge-url", required=True)
    score.add_argument("--judge-model", required=True)
    score.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    if args.command == "export":
        count = export_labels(args.input, args.output, args.sample_size, args.seed)
        print(f"Exported {count} examples for labeling to {args.output}")
    else:
        config = FilterConfig(judge_url=args.judge_url, judge_model=args.judge_model)
        print(json.dumps(score_labeled(args.labels, config, args.threshold), indent=2))


if __name__ == "__main__":
    main()
