---
license: cc-by-sa-3.0
task_categories:
- text-generation
language:
- pcm
---

# Curated instruction dataset

## Dataset description

Templated prompt-and-article pairs derived from Nigerian Pidgin Wikipedia via the Plaintext
Wikipedia dataset at revision `cc09caf8ba548e4e52ac881620ce29e0ea31320d`. The source articles
are CC BY-SA 3.0; preserve article attribution URLs and comply with ShareAlike requirements.
These are not original human-authored Q&A pairs.

## Intended use and limitations

This dataset is intended for research and evaluation of instruction tuning. It may contain
errors, stale advice, and biases inherited from its sources. Automated toxicity, language,
and LLM-judge filters are imperfect; they do not establish that examples are safe or correct.

## Data fields

`instruction`, `input`, `output`, `source_id`, `language`, filter scores, and provenance metadata
(JSON-encoded in Parquet/Hugging Face exports). Per-record metadata includes the article title,
source URL, dataset revision, and source license.
For StackExchange-derived data, retain and document source post URLs and user IDs for attribution.

## Curation

Document deduplication thresholds, trusted source language labels, heuristic thresholds, PII
handling, safety filter, judge model and prompt, decontamination split, and final retained counts
here. Nigerian Pidgin labels come from the pinned source; `langid` is not used to verify `pcm`.
Review the upstream license/terms before publishing.

## Evaluation

Report held-out split construction, contamination checks, judge-label Cohen's kappa, and the
raw / filtered / filtered-plus-synthetic ablation metrics here.
