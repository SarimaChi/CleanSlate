#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_dir="${DATA_DIR:-$repo_root/runs/ablation/data}"
output_dir="${OUTPUT_DIR:-$repo_root/runs/ablation/gpu}"
base_model="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
judge_model="${JUDGE_MODEL:-qwen/qwen-2.5-72b-instruct}"
openai_base_url="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY using the notebook's secret store before running." >&2
    exit 1
fi

for input_file in raw_train.jsonl filtered_train.jsonl filtered_plus_synthetic.jsonl test.jsonl; do
    if [[ ! -s "$data_dir/$input_file" ]]; then
        echo "Missing prepared input file: $data_dir/$input_file" >&2
        exit 1
    fi
done

python -m pip install -e ".[train,eval,harness]"
export USE_TF=0

curate-eval-ablate \
    --raw "$data_dir/raw_train.jsonl" \
    --filtered "$data_dir/filtered_train.jsonl" \
    --filtered-synthetic "$data_dir/filtered_plus_synthetic.jsonl" \
    --test-data "$data_dir/test.jsonl" \
    --model "$base_model" \
    --output-dir "$output_dir" \
    --seed 17 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-length 1560 \
    --max-steps -1 \
    --run-lm-eval \
    --language pcm \
    --trust-language-labels \
    --judge-url "${openai_base_url%/}/chat/completions" \
    --judge-model "$judge_model"
