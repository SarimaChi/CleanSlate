from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Example:
    instruction: str
    output: str
    input: str = ""
    source_id: str = ""
    language: str = ""
    judge_score: float | None = None
    toxicity_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Example":
        instruction = row.get("instruction", row.get("question", ""))
        output = row.get("output", row.get("answer", ""))
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise ValueError("instruction/question and output/answer must be strings")
        known = {"instruction", "question", "output", "answer", "input", "source_id",
                 "language", "judge_score", "toxicity_score", "metadata"}
        metadata_value = row.get("metadata") or {}
        if isinstance(metadata_value, str):
            try:
                metadata_value = json.loads(metadata_value)
            except json.JSONDecodeError as exc:
                raise ValueError("metadata string must contain a JSON object") from exc
        if not isinstance(metadata_value, dict):
            raise ValueError("metadata must be an object or a JSON-encoded object")
        metadata = dict(metadata_value)
        metadata.update({key: value for key, value in row.items() if key not in known})
        return cls(
            instruction=instruction.strip(),
            input=str(row.get("input") or "").strip(),
            output=output.strip(),
            source_id=str(row.get("source_id") or ""),
            language=str(row.get("language") or ""),
            judge_score=_optional_float(row.get("judge_score")),
            toxicity_score=_optional_float(row.get("toxicity_score")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.instruction, self.input, self.output) if part)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric score, got {value!r}") from exc
