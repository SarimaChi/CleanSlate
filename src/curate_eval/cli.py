from __future__ import annotations

import argparse
import json

from .filters import FilterConfig, curate
from .io import read_examples, write_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate noisy instruction data.")
    parser.add_argument("--input", required=True, help="Input JSONL, CSV, or Parquet")
    parser.add_argument("--output", required=True, help="Output JSONL or Parquet")
    parser.add_argument("--report", default="", help="Optional stage-count JSON report")
    parser.add_argument("--language", default="en", help="langid language code (default: en)")
    parser.add_argument("--min-instruction-chars", type=int, default=12)
    parser.add_argument("--min-output-chars", type=int, default=20)
    parser.add_argument("--minhash-threshold", type=float, default=0.82)
    parser.add_argument("--min-judge-score", type=float, default=0.60)
    parser.add_argument("--max-toxicity-score", type=float, default=0.20)
    parser.add_argument("--use-detoxify", action="store_true")
    parser.add_argument(
        "--trust-language-labels",
        action="store_true",
        help="Trust source language labels for languages not supported by langid (e.g. pcm)",
    )
    parser.add_argument("--judge-url", default="", help="OpenAI-compatible /chat/completions URL")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--eval-path", default="", help="Held-out eval JSONL for decontamination")
    args = parser.parse_args()
    config = FilterConfig(
        language=args.language,
        min_instruction_chars=args.min_instruction_chars,
        min_output_chars=args.min_output_chars,
        minhash_threshold=args.minhash_threshold,
        min_judge_score=args.min_judge_score,
        max_toxicity_score=args.max_toxicity_score,
        use_detoxify=args.use_detoxify,
        trust_language_labels=args.trust_language_labels,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        judge_api_key_env=args.judge_api_key_env,
        eval_path=args.eval_path,
    )
    curated, report = curate(read_examples(args.input), config)
    write_examples(args.output, curated)
    report["judge_mode"] = "llm" if config.judge_url else "local_heuristic_proxy"
    report["toxicity_mode"] = "detoxify" if config.use_detoxify else "lexicon_only"
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
