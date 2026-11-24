from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .schema import Example


def call_judge(prompt: str, config, *, max_tokens: int = 300, json_mode: bool = False) -> str:
    if not config.judge_url or not config.judge_model:
        raise ValueError("LLM judge requires both judge_url and judge_model")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(config.judge_api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_body = {
        "model": config.judge_model,
        "messages": [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}
    body = json.dumps(request_body).encode()
    request = urllib.request.Request(config.judge_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config.judge_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM judge request failed: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Judge API response did not contain choices[0].message.content") from exc


def score_example(example: Example, config) -> float:
    language_names = {"pcm": "Nigerian Pidgin", "yo": "Yoruba", "en": "English"}
    expected_language = language_names.get(example.language or config.language, example.language or config.language)
    prompt = (
        "Rate this instruction-answer pair for correctness, relevance, clarity, usefulness, "
        f"and whether it is written in {expected_language}. Return only a number from 0 to 1.\n"
        f"Instruction: {example.instruction}\n"
        f"Context: {example.input}\n"
        f"Answer: {example.output}"
    )
    response = call_judge(prompt, config, max_tokens=20)
    try:
        score = float(response.strip().strip("`"))
    except ValueError as exc:
        raise ValueError(f"Judge returned a non-numeric quality score: {response!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score must be within [0, 1], got {score}")
    return score
