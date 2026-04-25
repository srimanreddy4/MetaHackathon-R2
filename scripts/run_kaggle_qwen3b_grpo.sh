#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen2.5-3B-Instruct-bnb-4bit}"
EXTRA_PYTHONPATH="src:scripts"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${EXTRA_PYTHONPATH}"
fi
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"

run_train() {
  local out_dir="$1"
  local max_tasks="$2"
  local max_steps="$3"
  local grad_accum="$4"
  local seq_len="$5"
  local prompt_len="$6"
  local completion_len="$7"
  local eval_tasks="$8"
  local save_steps="$9"

  python scripts/train_unsloth_grpo.py \
    --model-name "${MODEL_NAME}" \
    --curriculum-buffer curriculum_results/buffer.json \
    --out-dir "${out_dir}" \
    --max-tasks "${max_tasks}" \
    --max-steps "${max_steps}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${grad_accum}" \
    --num-generations "${NUM_GENERATIONS}" \
    --max-seq-length "${seq_len}" \
    --max-prompt-length "${prompt_len}" \
    --max-completion-length "${completion_len}" \
    --eval-tasks "${eval_tasks}" \
    --lr "${LR:-5e-6}" \
    --save-steps "${save_steps}" \
    --logging-steps "${LOGGING_STEPS:-1}" \
    --difficulty-source "${DIFFICULTY_SOURCE:-feedback}" \
    --rollouts-per-candidate "${ROLLOUTS_PER_CANDIDATE:-3}" \
    --dynamic-curriculum \
    --curriculum-update-every "${CURRICULUM_UPDATE_EVERY:-40}" \
    --curriculum-evolve-iterations "${CURRICULUM_EVOLVE_ITERATIONS:-10}" \
    --curriculum-evolve-warmup-steps "${CURRICULUM_EVOLVE_WARMUP_STEPS:-75}" \
    --curriculum-evolve-frequency "${CURRICULUM_EVOLVE_FREQUENCY:-2}" \
    --print-completions \
    --verbose \

}

case "${MODE}" in
  gpu)
    nvidia-smi
    ;;
  verify)
    python -m pytest tests -q
    openenv validate .
    ;;
  smoke)
    run_train training_results/unsloth_grpo_qwen3b_smoke 40 50 4 768 512 128 8 50
    ;;
  main)
    run_train training_results/unsloth_grpo_qwen3b_kaggle 160 300 8 1024 768 192 32 100
    ;;
  long)
    run_train training_results/unsloth_grpo_qwen3b_kaggle_long 206 1000 8 1024 768 192 48 100
    ;;
  archive)
    shopt -s nullglob
    result_dirs=(training_results/unsloth_grpo_qwen3b_* training_results/unsloth_grpo_kaggle_ablation)
    if (( ${#result_dirs[@]} == 0 )); then
      echo "No GRPO result directories found to archive." >&2
      exit 1
    fi
    tar -czf /kaggle/working/qwen3b_grpo_results.tar.gz "${result_dirs[@]}"
    ls -lh /kaggle/working/qwen3b_grpo_results.tar.gz
    ;;
  summary)
    python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("training_results").glob("unsloth_grpo*/summary.json"))
if not paths:
    raise SystemExit("No GRPO summary.json files found under training_results/")
for path in paths:
    data = json.loads(path.read_text())
    print(path)
    print("  model:", data.get("model_name"))
    print("  baseline_mean_reward:", data.get("baseline_mean_reward"))
    print("  trained_mean_reward:", data.get("trained_mean_reward"))
    print("  duration_sec:", data.get("duration_sec"))
PY
    ;;
  *)
    cat >&2 <<EOF
Unknown mode: ${MODE}

Usage:
  bash scripts/run_kaggle_qwen3b_grpo.sh gpu
  bash scripts/run_kaggle_qwen3b_grpo.sh verify
  bash scripts/run_kaggle_qwen3b_grpo.sh smoke
  bash scripts/run_kaggle_qwen3b_grpo.sh main
  bash scripts/run_kaggle_qwen3b_grpo.sh long
  bash scripts/run_kaggle_qwen3b_grpo.sh summary
  bash scripts/run_kaggle_qwen3b_grpo.sh archive
EOF
    exit 2
    ;;
esac
