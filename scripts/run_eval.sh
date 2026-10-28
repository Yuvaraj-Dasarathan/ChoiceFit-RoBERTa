#!/usr/bin/env bash
# Run the untrained RoBERTa baseline and ChoiceFit (both local), JEV (TypeSafe API called from this machine) and JEV (TypeSafe API, called from this machine) on one labeled JSONL file, then compare.
# Usage: scripts/run_eval.sh DATA_FILE NAME [LIMIT]
#   DATA_FILE  labeled JSONL, e.g. data/typed-full-8x256/test.jsonl
#   NAME       label for the output folder, e.g. test (must be new)
#   LIMIT      optional: evaluate only the first N rows with both tools (saves API cost)
# Env: BASE_MODEL (default: cached roberta-base snapshot), MODEL_DIR (default artifacts/sagemaker/choicefit-l4-full-b8-001), TYPESAFE_API_KEY (required)
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${1:?usage: $0 DATA_FILE NAME [LIMIT]}
NAME=${2:?usage: $0 DATA_FILE NAME [LIMIT]}
LIMIT=${3:-0}
MODEL_DIR=${MODEL_DIR:-artifacts/sagemaker/choicefit-l4-full-b8-001}
PY=.venv/bin/python
BASE_MODEL=${BASE_MODEL:-$(echo .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/*)}
OUT="$MODEL_DIR/runs/$NAME"
: "${TYPESAFE_API_KEY:?Set TYPESAFE_API_KEY in your shell first}"
mkdir -p "$OUT"

if [ "$LIMIT" -gt 0 ]; then
  head -n "$LIMIT" "$DATA" > "$OUT/input.jsonl"
  DATA="$OUT/input.jsonl"
fi

$PY -m src.evaluate --model "$BASE_MODEL" --max-length 256 --seed 42 --data "$DATA" \
  --batch-size 8 --pair-batch-size 64 --predictions "$OUT/baseline.jsonl" | tee "$OUT/baseline-metrics.json"
$PY -m src.evaluate --model "$MODEL_DIR" --data "$DATA" --batch-size 8 --pair-batch-size 64 \
  --predictions "$OUT/choicefit.jsonl" | tee "$OUT/choicefit-metrics.json"
$PY -m src.evaluate_jev --data "$DATA" --predictions "$OUT/jev.jsonl" --workers "${JEV_WORKERS:-8}"
$PY -m src.compare_predictions --choicefit "$OUT/choicefit.jsonl" --jev "$OUT/jev.jsonl" \
  --output-dir "$OUT/comparison"
$PY -m src.compare_all --system baseline="$OUT/baseline.jsonl" --system choicefit="$OUT/choicefit.jsonl" \
  --system jev="$OUT/jev.jsonl" --output-dir "$OUT/comparison-all"
echo "Reports: $OUT/comparison/report.md (ChoiceFit vs JEV), $OUT/comparison-all/report.md (all three)"
