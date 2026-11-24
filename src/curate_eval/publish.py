from __future__ import annotations

import argparse
import json

from .io import read_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish curated JSONL data to Hugging Face Hub.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. username/dataset-name")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card", required=True, help="Dataset card markdown file")
    args = parser.parse_args()
    from datasets import Dataset

    examples = read_examples(args.data)
    if not examples:
        raise ValueError("Refusing to publish an empty dataset")
    rows = [example.to_dict() for example in examples]
    for row in rows:
        row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(args.repo_id, private=args.private, commit_message="Publish curated instruction dataset")
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset")
    print(f"Published {len(examples)} examples to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
