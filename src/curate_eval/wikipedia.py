from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from urllib.parse import quote

from .io import write_examples
from .schema import Example

SOURCE_DATASET = "ffatty/plaintext-wikipedia-full-nigerian-pidgin"
SOURCE_REVISION = "cc09caf8ba548e4e52ac881620ce29e0ea31320d"
SOURCE_LICENSE = "CC BY-SA 3.0"
SOURCE_SITE = "https://pcm.wikipedia.org/wiki/"


def convert(shard_path: str | Path, output_path: str | Path) -> int:
    """Turn the source's title/body/blank-line article format into instruction JSONL."""
    examples: list[Example] = []
    with gzip.open(shard_path, "rt", encoding="utf-8", newline="") as stream:
        line_number = 0
        while True:
            title_line = stream.readline()
            if not title_line:
                break
            line_number += 1
            title = title_line.rstrip("\r\n").strip()
            body_line = stream.readline()
            if not body_line:
                raise ValueError(f"{shard_path}:{line_number}: missing article body")
            line_number += 1
            body = body_line.rstrip("\r\n").strip()
            separator = stream.readline()
            if separator:
                line_number += 1
                if separator.strip():
                    raise ValueError(f"{shard_path}:{line_number}: expected a blank line after article")
            if not title or not body:
                continue
            article_url = SOURCE_SITE + quote(title.replace(" ", "_"), safe="()")
            examples.append(Example(
                instruction=f"Wetin be {title}?",
                output=body,
                source_id=f"pcmwiki:{title}",
                language="pcm",
                metadata={
                    "source_dataset": SOURCE_DATASET,
                    "source_revision": SOURCE_REVISION,
                    "source_license": SOURCE_LICENSE,
                    "article_title": title,
                    "article_url": article_url,
                },
            ))
    write_examples(output_path, examples)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the Nigerian Pidgin Wikipedia plaintext shard to instruction JSONL."
    )
    parser.add_argument("--input", required=True, help="Downloaded .txt.gz source shard")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(f"Converted {count} articles to {args.output}")


if __name__ == "__main__":
    main()
