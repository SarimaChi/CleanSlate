from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from .schema import Example


def read_examples(path: str | Path) -> list[Example]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        rows = pl.read_parquet(path).to_dicts()
    elif path.suffix.lower() == ".csv":
        rows = pl.read_csv(path).to_dicts()
    else:
        rows = list(_jsonl_rows(path))
    examples: list[Example] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            examples.append(Example.from_dict(row))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return examples


def write_examples(path: str | Path, examples: Iterable[Example]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        rows = [example.to_dict() for example in examples]
        for row in rows:
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
        if rows:
            pl.DataFrame(rows).write_parquet(path)
        else:
            pl.DataFrame(schema={
                "instruction": pl.String,
                "input": pl.String,
                "output": pl.String,
                "source_id": pl.String,
                "language": pl.String,
                "judge_score": pl.Float64,
                "toxicity_score": pl.Float64,
                "metadata": pl.String,
            }).write_parquet(path)
        return
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def _jsonl_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield row
