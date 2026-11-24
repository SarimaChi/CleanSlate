---
license: cc-by-sa-3.0
pretty_name: Nigerian Pidgin Wikipedia Instructions (Filtered + Synthetic)
task_categories:
- text-generation
language:
- pcm
size_categories:
- 1K<n<10K
---

# Naija Pidgin: filtered + synthetic instruction data

## Dataset description

The default Hugging Face `train` split contains 1,456 Nigerian Pidgin (`pcm`)
instruction-answer pairs: 1,412 filtered examples derived from Nigerian Pidgin Wikipedia
articles and 44 synthetic examples generated from the filtered training data. The
article-derived examples use the template `Wetin be <article title>?` as the instruction and
cleaned article text as the answer. They are templated examples, not human-authored Q&A.

The source is
[Plaintext Wikipedia (Nigerian Pidgin)](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin)
at revision `cc09caf8ba548e4e52ac881620ce29e0ea31320d`, derived from Wikimedia's Pidgin
Wikipedia. The source content is licensed CC BY-SA 3.0. Article URLs and source-license
provenance are preserved in each record. This dataset is distributed under CC BY-SA 3.0;
users must preserve attribution and comply with the license's ShareAlike terms.

## Dataset structure

The default `train` split is stored as root-level `train.parquet` and has these fields:

- `instruction`, `input`, `output`
- `source_id`, `language`
- `judge_score`, `toxicity_score` (optional scores)
- `metadata` (JSON-encoded object with source, revision, license, article URL, and synthetic
  provenance where applicable)

Synthetic records are marked with `metadata.synthetic`, `metadata.seed_source_id`, and
`metadata.generation_model`. All 1,456 records retain an article attribution URL and
`CC BY-SA 3.0` source-license metadata.

```python
from datasets import load_dataset

dataset = load_dataset("Richiio/Naija_Pidgin", split="train")
```

## Supplementary files

Additional source and curation files are available under `data/`:

- `raw.jsonl`: all 1,901 converted source records.
- `filtered.jsonl`: all 1,569 records retained by the initial curation.
- `synthetic.jsonl`: a separate 49-record synthetic generation artifact.
- `filter-report.json`, `pcmwiki-build-manifest.json`, and `README.md`: curation counts,
  source-shard metadata, and source instructions.

The raw and filtered JSONL files include the 157 records reserved for the held-out evaluation
set. They are provided as supplementary corpus snapshots, not as leakage-free training or
evaluation splits. The source `.gz` shard itself is not included; see `data/README.md` for its
upstream location and checksum.

## Curation and composition

The pinned source shard contains 1,901 article records. The initial curation retained 1,569;
these counts describe the supplementary corpus files. The 1,412 article-derived records in
the default `train` split are the filtered training portion.
The curation report recorded 7 heuristic-quality rejections, 1 lexical toxicity/bias
rejection, and 324 low-score rejections by a local heuristic quality proxy; exact and near
duplicate removals and language mismatches were zero. PII patterns were scrubbed in 368
records. The local proxy is not an LLM assessment.

The filter configuration used exact deduplication and MinHash deduplication with a Jaccard
threshold of 0.82 and 64 permutations. It required at least 12 instruction characters and 20
answer characters, capped combined text at 12,000 characters, and applied symbol-ratio,
repeated-line, and low-diversity checks. Email addresses, phone numbers, and URLs matching
the configured patterns were replaced with `[EMAIL]`, `[PHONE]`, and `[URL]`. The safety
filter used a small lexical toxicity/bias list; Detoxify was not enabled. Nigerian Pidgin
language labels were trusted from the pinned source metadata rather than inferred by
`langid`.

A deterministic seed-17 held-out set of 157 records was reserved for evaluation. The default
`train` split is decontaminated against that held-out set; raw and filtered supplementary
files contain the held-out records as noted above. The 44 synthetic examples in the default
`train` split were generated only from its training examples.

The 44 synthetic records were generated with `qwen/qwen-2.5-72b-instruct`. The generation
prompt requested novel, self-contained instructions and answers grounded in seed facts,
written in Nigerian Pidgin and returned as JSON. Generated records were passed through the
curation filters and scored by the configured model for correctness, relevance, clarity,
usefulness, and language match (minimum score 0.60); all 44 candidates were retained.
Synthetic text is machine-generated and has not been independently fact-checked or manually
reviewed.

The separate `data/synthetic.jsonl` file comes from an earlier generation run and is not the
same set of 44 synthetic examples in the default `train` split. Five of its records have
`seed_source_id` values in the held-out evaluation set. Do not use this supplementary file
for leakage-free evaluation.

## Intended use and limitations

This dataset is intended for research and instruction-tuning experiments. It is small and
source-specific; the article-to-question template does not represent naturally occurring
conversation or diverse instruction styles. Wikipedia content may be incomplete, outdated,
or incorrect. Synthetic answers may contain unsupported or incorrect claims. Automated
filters are imperfect and do not establish factuality, safety, or language quality.

Do not treat the supplementary raw, filtered, or synthetic files as evaluation sets or infer
model-quality improvements from the three-step local smoke runs.
