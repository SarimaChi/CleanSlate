from __future__ import annotations

import argparse
import json

from .filters import FilterConfig, curate
from .io import read_examples, write_examples
from .judge import call_judge
from .schema import Example


def generate(
    seeds: list[Example],
    config: FilterConfig,
    count: int,
    seed: int = 17,
    language: str = "",
) -> list[Example]:
    if not config.judge_url or not config.judge_model:
        raise ValueError("Synthetic generation requires --judge-url and --judge-model")
    if count < 0:
        raise ValueError("count cannot be negative")
    if count and not seeds:
        raise ValueError("At least one seed example is required when count is greater than zero")
    generated = []
    language_names = {"pcm": "Nigerian Pidgin", "yo": "Yoruba", "en": "English"}
    for index in range(count):
        source = seeds[index % len(seeds)]
        target_language = source.language or language or config.language
        target_language_name = language_names.get(target_language, target_language)
        article_source = source.metadata.get("article_url", source.source_id)
        prompt = (
            "Create one novel, self-contained instruction and a correct answer grounded only in "
            "the seed facts. Do not copy the seed wording or introduce unsupported facts. "
            f"Write both instruction and answer in {target_language_name}. "
            "Return only JSON with string fields instruction, input, output.\n"
            f"Seed instruction: {source.instruction}\nSeed answer: {source.output}\n"
            f"Source attribution: {article_source}"
        )
        response = call_judge(prompt, config, max_tokens=800, json_mode=True)
        try:
            row = json.loads(response)
            example = Example.from_dict(row)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Synthetic response {index + 1} was not valid example JSON") from exc
        example.source_id = f"synthetic-{seed}-{index + 1}"
        example.language = target_language
        example.metadata.update({
            "synthetic": True,
            "seed_source_id": source.source_id,
            "generation_model": config.judge_model,
        })
        for key in ("source_dataset", "source_revision", "source_license", "article_url"):
            if key in source.metadata:
                example.metadata[key] = source.metadata[key]
        generated.append(example)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and filter synthetic instruction examples.")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--judge-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--trust-language-labels", action="store_true")
    parser.add_argument("--eval-path", default="")
    args = parser.parse_args()
    config = FilterConfig(language=args.language, judge_url=args.judge_url,
                          judge_model=args.judge_model, eval_path=args.eval_path,
                          trust_language_labels=args.trust_language_labels)
    generated = generate(read_examples(args.seeds), config, args.count, language=args.language)
    curated, report = curate(generated, config)
    write_examples(args.output, curated)
    print(json.dumps({"generated": len(generated), "retained": len(curated), "filters": report}, indent=2))


if __name__ == "__main__":
    main()
