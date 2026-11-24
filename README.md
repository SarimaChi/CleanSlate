# Curate & Evaluate

An open-source, reproducible pipeline for turning noisy instruction-answer pairs into a
curated dataset, validating automated judge labels, and comparing matched LoRA fine-tuning
ablations. Core curation runs locally; judge-based filtering and synthesis use any
OpenAI-compatible chat-completions endpoint.

> **Research scaffold, not a pre-made dataset.** You provide corpus files and model choices.
> The local judge proxy and safety lexicon are for smoke tests only; use a real judge and a
> stronger safety model for published results. Review source licenses and personally
> identifiable information before releasing data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

For model training/evaluation, lm-evaluation-harness, and Detoxify, install
`.[train,eval,harness,toxicity]`. Set the API key environment variable expected by your
endpoint (default `OPENAI_API_KEY`) without putting credentials in data or source control.

## 1. Prepare data

Each record has `instruction`, optional `input`, and `output`. `question` and `answer` are
accepted aliases. Input supports JSONL, CSV, or Parquet.

The workspace includes a Nigerian Pidgin Wikipedia plaintext shard in `data/`. Convert it into
instruction JSONL and run curation as follows:

```bash
curate-eval-wikipedia --input data/pcmwiki_01_of_01.txt.gz --output data/raw.jsonl
curate-eval --input data/raw.jsonl --output data/filtered.jsonl \
  --report data/filter-report.json --language pcm --trust-language-labels
```

This source contains 1,901 articles under CC BY-SA 3.0, not native Q&A. The converter creates
template prompts (`Wetin be <title>?`) with article text as answers. `langid` does not reliably
support Nigerian Pidgin, so the explicit option trusts the source's pinned `pcm` language label;
it is not a language-ID result. See [`data/README.md`](data/README.md) for attribution and
limitations.

Alternatively, for a StackExchange archive, download a public data dump and unpack `Posts.xml`, then convert:

```bash
curate-eval-stackexchange --posts-xml Posts.xml --output data/raw.jsonl \
  --site-url https://stackoverflow.com
```

This converter pairs questions with their accepted answers. It retains question text in memory,
so for large StackExchange dumps, first extract a relevant site or subset of Posts.xml. It
preserves source post URLs and user IDs in metadata for attribution; use the actual source
site URL, not necessarily Stack Overflow.

## 2. Curate

```bash
curate-eval \
  --input data/raw.jsonl --output data/filtered.jsonl \
  --report data/filter-report.json --language en \
  --judge-url https://your-endpoint/v1/chat/completions \
  --judge-model your-judge-model --eval-path data/test.jsonl
```

Stages, in order: empty-field removal, normalized exact deduplication, deterministic seeded
MinHash candidate matching with Jaccard verification, `langid` detection, heuristic length/
symbol/repetition checks, email/phone/URL scrubbing, lexicon safety screening (optionally
Detoxify), LLM-as-judge scoring, then overlap decontamination against the supplied held-out
evaluation examples. Every stage emits retained/removal counts. PII is scrubbed rather than
discarding the whole example.

Without `--judge-url`, a simple local heuristic proxy is used so the pipeline can be smoke
tested; this is **not** LLM judging and must not be used to claim judge-based curation. Enable
`--use-detoxify` to add model-based toxicity scoring. The built-in lexicons are intentionally
small and are not a substitute for moderation or human review.

## 3. Validate judge agreement

```bash
curate-eval-judge-labels export --input data/raw.jsonl --output data/to-label.jsonl --sample-size 150
# Fill each human_label with 0 (reject) or 1 (keep), then:
curate-eval-judge-labels score --labels data/to-label.jsonl \
  --judge-url https://your-endpoint/v1/chat/completions --judge-model your-judge-model
```

The report uses Cohen's kappa after thresholding judge scores. Freeze the labeling rubric and
threshold before annotation; report the number labeled and class balance alongside kappa.

## 4. Generate and filter synthetic data

```bash
curate-eval-synthesize --seeds data/filtered.jsonl --output data/synthetic.jsonl \
  --count 1000 --judge-url https://your-endpoint/v1/chat/completions \
  --judge-model your-judge-model --eval-path data/test.jsonl
```

Synthesis is followed by the same curation stages. Combine the kept synthetic records with
the filtered seed file to create the third ablation input.

## 5. Train and evaluate ablations

The ablation runner creates `filtered_plus_synthetic.jsonl` if it is missing: it generates
synthetic examples from the filtered seeds, applies the same quality/safety/decontamination
filters, and combines retained synthetic examples with the filtered seeds. This requires a
configured OpenAI-compatible generator/judge endpoint and model. The same base model, seed,
LoRA rank/alpha/dropout, optimizer settings, epochs, and training arguments are passed to
each training run:

```bash
export OPENAI_API_KEY="your-token" # set securely in your terminal; never commit it
curate-eval-ablate \
  --raw data/raw.jsonl --filtered data/filtered.jsonl \
  --filtered-synthetic data/filtered_plus_synthetic.jsonl \
  --synthetic-count 500 --language pcm --trust-language-labels \
  --model Qwen/Qwen2.5-0.5B-Instruct --output-dir runs/ablation \
  --judge-url https://your-endpoint/v1/chat/completions --judge-model your-generator-and-judge-model
```

For this corpus, tokenizing the train-formatted examples with Qwen2.5-0.5B-Instruct gives
per-arm p50/p90/p95/max lengths of 475/1174/1555/8819 (raw),
502/1150/1510/4198 (filtered), and 483/1137/1507/4198
(filtered-plus-synthetic). The runner defaults to `--max-length 1560`, rounded up from the
largest per-arm p95 to a multiple of eight. This avoids truncating answers in about 95% of
examples; very long outliers are still truncated.

If you provide `--test-data data/test.jsonl`, scores include ROUGE-L, SacreBLEU, and held-out
reference perplexity, plus judge preference when configured. Add `--run-lm-eval` to also run
lm-evaluation-harness for each arm; install `.[harness]` first. The custom task in
`lm_eval/tasks/curate_eval/` measures exact-match answers and is a starter benchmark.

The runner caches unfiltered synthetic candidates separately so a filtering/training rerun
doesn't spend again on generation. `runs/ablation/ablation_summary.json` records the fixed
training configuration, per-arm training device/dtype/token throughput, and dataset sizes.
CUDA training uses bf16 when supported, otherwise fp16; CPU/MPS uses fp32. The Mac MPS
smoke run was extremely slow, so use a Kaggle or Colab GPU for full training.

### Kaggle / Colab GPU run

Prepare/upload these data files from `runs/ablation/data/` as a private Kaggle dataset or
Colab files: `raw_train.jsonl`, `filtered_train.jsonl`,
`filtered_plus_synthetic.jsonl`, and `test.jsonl`. They are ignored by Git and are not
included in this repository. In the notebook, clone/copy this project, set `DATA_DIR` to the
uploaded files' directory, and store `OPENAI_API_KEY` as a Kaggle Secret or Colab secret
(never in the notebook or repository). Then run:

```bash
export DATA_DIR=/path/to/private/ablation-data
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
# Inject OPENAI_API_KEY from the notebook secret store into this process; do not save it in a cell.
bash scripts/run_ablation_gpu.sh
```

The script installs the `train`, `eval`, and `harness` extras, uses one epoch and seed 17 for
all arms, saves LoRA adapters and metrics under `runs/ablation/gpu/`, and runs the custom
lm-evaluation-harness task plus ROUGE-L, BLEU, perplexity, and OpenRouter judge win rates.
Override `DATA_DIR`, `OUTPUT_DIR`, `BASE_MODEL`, or `JUDGE_MODEL` as needed.

For a local pipeline smoke test, create a 20-example sample from the held-out set and run
three optimizer steps per arm:

```bash
python - <<'PY'
import random
from curate_eval.io import read_examples, write_examples

held_out = read_examples("runs/ablation/data/test.jsonl")
write_examples(
    "runs/ablation/data/test_smoke20.jsonl",
    random.Random(17).sample(held_out, 20),
)
PY

curate-eval-ablate \
  --raw runs/ablation/data/raw_train.jsonl \
  --filtered runs/ablation/data/filtered_train.jsonl \
  --filtered-synthetic runs/ablation/data/filtered_plus_synthetic.jsonl \
  --test-data runs/ablation/data/test_smoke20.jsonl \
  --max-length 1560 --max-steps 3 --batch-size 1 \
  --gradient-accumulation-steps 1 --run-lm-eval \
  --judge-url "$OPENAI_BASE_URL/chat/completions" \
  --judge-model qwen/qwen-2.5-72b-instruct \
  --output-dir runs/ablation/smoke
```

## 6. Publish

Review the upstream license and dataset terms, fill in run-specific details in
[`dataset-card.md`](dataset-card.md), then authenticate with `huggingface-cli login` and run:

```bash
curate-eval-publish --data data/filtered.jsonl --repo-id YOUR_NAME/YOUR_DATASET \
  --card dataset-card.md
```

## Limitations

- Language identification depends on `langid`; it may misclassify code, short text, and
  underrepresented languages such as Nigerian Pidgin or Yoruba. Change `--language` only after
  checking a manually reviewed language sample.
- The default technical source is English StackExchange. Low-resource-language use requires
  selecting a suitable source and validating language identification and judge quality first.
- The provided overlap decontamination is conservative n-gram overlap filtering, not semantic
  proof of independence. Keep the final test set private and report its construction.
- Full training/evaluation requires appropriate hardware, compatible model licenses, and
  the optional dependencies. The smoke tests do not establish model-quality improvements.
