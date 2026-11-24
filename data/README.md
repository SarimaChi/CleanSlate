# Data directory

## Downloaded corpus

This workspace includes `pcmwiki_01_of_01.txt.gz`, a complete plaintext shard (1,901 articles)
from Nigerian Pidgin Wikipedia. The source dataset is
[Plaintext Wikipedia (Nigerian Pidgin)](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin),
which is derived from the [Wikimedia Pidgin Wikipedia dump](https://dumps.wikimedia.org/pcmwiki/latest).

- Source revision: `cc09caf8ba548e4e52ac881620ce29e0ea31320d`
- Source license: CC BY-SA 3.0; retain attribution and follow ShareAlike terms for redistributed derivatives.
- Shard completeness: `pcmwiki-build-manifest.json` reports 1/1 shards and 1,901 articles.
- Shard SHA-256: `0a063d5333b027fba36c8b5a3561e2801037ef38e9ca641a45a7e2ae198fbf79`
- Article source URLs are included in each converted record's metadata.

To fetch the same pinned source revision again:

```bash
curl -fL -o data/pcmwiki_01_of_01.txt.gz \
  https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin/resolve/cc09caf8ba548e4e52ac881620ce29e0ea31320d/data/pcmwiki_01_of_01.txt.gz
curl -fL -o data/pcmwiki-build-manifest.json \
  https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin/resolve/cc09caf8ba548e4e52ac881620ce29e0ea31320d/data/build-manifest.json
```

The source is an encyclopedia, not naturally occurring Q&A. Conversion below creates a simple
template instruction (`Wetin be <article title>?`) whose answer is the cleaned article text; treat
these as templated instruction examples, not human-written instruction/answer pairs.

Convert the downloaded shard to the pipeline's JSONL schema:

```bash
curate-eval-wikipedia --input data/pcmwiki_01_of_01.txt.gz --output data/raw.jsonl
```

Then curate using the corpus's verified source language tag:

```bash
curate-eval --input data/raw.jsonl --output data/filtered.jsonl \
  --report data/filter-report.json --language pcm --trust-language-labels
```

`langid` does not reliably identify Nigerian Pidgin (`pcm`); this flag trusts the pinned
source-language metadata rather than claiming automatic Pidgin language identification. Review
samples manually before interpreting results.

The pipeline also accepts JSONL, CSV, or Parquet with `instruction`/`output` fields; `question`/
`answer` are accepted aliases. For another StackExchange dump, use
`curate-eval-stackexchange --posts-xml Posts.xml --output data/raw.jsonl --site-url https://example.com`.
