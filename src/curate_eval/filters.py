from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import langid

from .schema import Example

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_URL_RE = re.compile(r"\bhttps?://\S+", re.I)
_TOXIC_TERMS = {"kill yourself", "subhuman", "racial slur", "go die"}
_BIAS_TERMS = {"all women are", "all men are", "those people are"}


@dataclass
class FilterConfig:
    language: str = "en"
    min_instruction_chars: int = 12
    min_output_chars: int = 20
    max_text_chars: int = 12000
    max_symbol_ratio: float = 0.30
    max_repeated_line_ratio: float = 0.50
    minhash_threshold: float = 0.82
    minhash_permutations: int = 64
    min_judge_score: float = 0.60
    max_toxicity_score: float = 0.20
    use_detoxify: bool = False
    trust_language_labels: bool = False
    judge_url: str = ""
    judge_model: str = ""
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_timeout_seconds: int = 45
    eval_path: str = ""

    def __post_init__(self) -> None:
        if not self.language:
            raise ValueError("language must not be empty")
        if not 0.0 <= self.min_judge_score <= 1.0:
            raise ValueError("min_judge_score must be within [0, 1]")
        if not 0.0 <= self.max_toxicity_score <= 1.0:
            raise ValueError("max_toxicity_score must be within [0, 1]")
        if not 0.0 <= self.max_symbol_ratio <= 1.0:
            raise ValueError("max_symbol_ratio must be within [0, 1]")
        if not 0.0 <= self.max_repeated_line_ratio <= 1.0:
            raise ValueError("max_repeated_line_ratio must be within [0, 1]")
        if self.max_text_chars < 1 or self.judge_timeout_seconds < 1:
            raise ValueError("max_text_chars and judge_timeout_seconds must be positive")


def curate(examples: Iterable[Example], config: FilterConfig) -> tuple[list[Example], dict[str, int]]:
    """Run stable, auditable filters in order and return stage-level removal counts."""
    data = list(examples)
    counts = {"input": len(data)}

    normalized = [example for example in data if _has_content(example)]
    counts["empty_or_short_fields"] = len(data) - len(normalized)
    data = normalized

    exact_unique = _exact_deduplicate(data)
    counts["exact_duplicates"] = len(data) - len(exact_unique)
    data = exact_unique

    near_unique = _minhash_deduplicate(data, config.minhash_threshold, config.minhash_permutations)
    counts["near_duplicates"] = len(data) - len(near_unique)
    data = near_unique

    lang_unique = [
        example for example in data
        if _language_matches(example, config.language, config.trust_language_labels)
    ]
    counts["language_mismatch"] = len(data) - len(lang_unique)
    data = lang_unique

    quality_unique = [example for example in data if _heuristic_quality(example, config)]
    counts["heuristic_quality"] = len(data) - len(quality_unique)
    data = quality_unique

    pii_unique = [_scrub_pii(example) for example in data]
    counts["pii_scrubbed"] = sum(a.text != b.text for a, b in zip(data, pii_unique))
    data = pii_unique

    safety_unique = [example for example in data if _safety_check(example, config)]
    counts["toxicity_or_bias"] = len(data) - len(safety_unique)
    data = safety_unique

    judged = [_score_judge(example, config) for example in data]
    judge_unique = [example for example in judged if (example.judge_score or 0.0) >= config.min_judge_score]
    counts["judge_quality"] = len(judged) - len(judge_unique)
    data = judge_unique

    eval_unique = _decontaminate(data, config.eval_path)
    counts["eval_contamination"] = len(data) - len(eval_unique)
    data = eval_unique
    counts["output"] = len(data)
    return data, counts


def deduplicate_examples(examples: Iterable[Example], threshold: float = 0.82,
                         permutations: int = 64) -> list[Example]:
    exact_unique = _exact_deduplicate(list(examples))
    return _minhash_deduplicate(exact_unique, threshold, permutations)


def _has_content(example: Example) -> bool:
    return (len(example.instruction.strip()) >= 2 and len(example.output.strip()) >= 2)


def _normalized_text(example: Example) -> str:
    text = unicodedata.normalize("NFKC", example.text).casefold()
    return " ".join(_WORD_RE.findall(text))


def _exact_deduplicate(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    output = []
    for example in examples:
        fields = (example.instruction, example.input, example.output)
        normalized = "\x1f".join(
            " ".join(_WORD_RE.findall(unicodedata.normalize("NFKC", value).casefold()))
            for value in fields
        )
        key = hashlib.sha256(normalized.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            output.append(example)
    return output


def _minhash_deduplicate(examples: list[Example], threshold: float, permutations: int) -> list[Example]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minhash_threshold must be in (0, 1]")
    if permutations < 16:
        raise ValueError("minhash_permutations must be at least 16")
    prime = 2**61 - 1
    rng = random.Random(1729)
    coefficients = [(rng.randrange(1, prime), rng.randrange(0, prime)) for _ in range(permutations)]
    configurations = []
    for rows_per_band in range(1, permutations + 1):
        bands = permutations // rows_per_band
        if bands:
            candidate_probability = 1.0 - (1.0 - threshold ** rows_per_band) ** bands
            configurations.append((rows_per_band, bands, candidate_probability))
    reliable = [config for config in configurations if config[2] >= 0.95]
    rows_per_band, bands, _ = max(reliable or configurations, key=lambda config: config[0])
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    accepted: list[Example] = []
    accepted_shingles: list[set[str]] = []

    for example in examples:
        words = _WORD_RE.findall(_normalized_text(example))
        shingles = {" ".join(words[index:index + 3]) for index in range(max(1, len(words) - 2))}
        if not shingles:
            shingles = set(words)
        signature = tuple(
            min((a * int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") + b) % prime
                for token in shingles)
            for a, b in coefficients
        ) if shingles else tuple([prime] * permutations)
        candidates: set[int] = set()
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start:min(start + rows_per_band, permutations)])
            candidates.update(buckets[key])
        duplicate = False
        for index in candidates:
            previous = accepted_shingles[index]
            union = len(shingles | previous)
            similarity = len(shingles & previous) / union if union else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        accepted_index = len(accepted)
        accepted.append(example)
        accepted_shingles.append(shingles)
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start:min(start + rows_per_band, permutations)])
            buckets[key].append(accepted_index)
    return accepted


def _language_matches(example: Example, target: str, trust_source_label: bool = False) -> bool:
    if trust_source_label and example.language == target:
        if not example.metadata.get("source_dataset") or not example.metadata.get("source_revision"):
            raise ValueError("Trusted language labels require source_dataset and source_revision metadata")
        return True
    text = f"{example.instruction}\n{example.output}"
    try:
        language, _ = langid.classify(text)
    except (ValueError, IndexError):
        return False
    example.language = language
    return language == target


def _heuristic_quality(example: Example, config: FilterConfig) -> bool:
    text = example.text
    if len(example.instruction) < config.min_instruction_chars or len(example.output) < config.min_output_chars:
        return False
    if len(text) > config.max_text_chars:
        return False
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return False
    symbol_ratio = sum(not char.isalnum() for char in visible) / len(visible)
    if symbol_ratio > config.max_symbol_ratio:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and (len(lines) - len(set(lines))) / len(lines) > config.max_repeated_line_ratio:
        return False
    words = _WORD_RE.findall(text.casefold())
    if len(words) > 12 and len(set(words)) / len(words) < 0.12:
        return False
    return True


def _scrub_pii(example: Example) -> Example:
    scrubbed = Example.from_dict(example.to_dict())
    scrubbed.instruction = _replace_pii(scrubbed.instruction)
    scrubbed.input = _replace_pii(scrubbed.input)
    scrubbed.output = _replace_pii(scrubbed.output)
    return scrubbed


def _replace_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return _URL_RE.sub("[URL]", text)


def _safety_check(example: Example, config: FilterConfig) -> bool:
    text = example.text.casefold()
    if any(term in text for term in _TOXIC_TERMS | _BIAS_TERMS):
        return False
    if config.use_detoxify:
        try:
            from detoxify import Detoxify
        except ImportError as exc:
            raise RuntimeError("Install the 'toxicity' extra to use Detoxify: pip install -e '.[toxicity]'") from exc
        model = _detoxify_model()
        scores = model.predict(text)
        score = float(scores["toxicity"])
        example.toxicity_score = score
        if score > config.max_toxicity_score:
            return False
    return True


def _score_judge(example: Example, config: FilterConfig) -> Example:
    if config.judge_url:
        from .judge import score_example
        example.judge_score = score_example(example, config)
    else:
        # Transparent local proxy used for smoke tests; configure judge_url for actual LLM scoring.
        has_direct_answer = len(_WORD_RE.findall(example.output)) >= 8
        has_instruction = len(_WORD_RE.findall(example.instruction)) >= 4
        example.judge_score = 0.75 if has_direct_answer and has_instruction else 0.35
    return example


def _decontaminate(examples: list[Example], eval_path: str) -> list[Example]:
    if not eval_path:
        return examples
    eval_examples = _load_eval_examples(Path(eval_path))
    eval_shingles = set()
    eval_hashes = set()
    for example in eval_examples:
        normalized = _normalized_text(example)
        eval_hashes.add(hashlib.sha256(normalized.encode()).hexdigest())
        words = normalized.split()
        eval_shingles.update(" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4)))
    output = []
    for example in examples:
        normalized = _normalized_text(example)
        if hashlib.sha256(normalized.encode()).hexdigest() in eval_hashes:
            continue
        words = normalized.split()
        shingles = {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
        if shingles and shingles & eval_shingles:
            overlap = len(shingles & eval_shingles) / len(shingles)
            if overlap >= 0.80:
                continue
        output.append(example)
    return output


def _load_eval_examples(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")
    from .io import read_examples
    return read_examples(path)


_DETOXIFY = None


def _detoxify_model():
    global _DETOXIFY
    if _DETOXIFY is None:
        from detoxify import Detoxify
        _DETOXIFY = Detoxify("original")
    return _DETOXIFY
