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

  local resume_args=()
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    resume_args=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi

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
    --logging-steps "${LOGGING_STEPS:-5}" \
    "${resume_args[@]}"
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
    run_train training_results/unsloth_grpo_qwen3b_kaggle 160 600 8 1024 768 192 32 100
    ;;
  resume-main)
    shopt -s nullglob
    checkpoints=(training_results/unsloth_grpo_qwen3b_kaggle/checkpoint-*)
    if (( ${#checkpoints[@]} == 0 )); then
      echo "No checkpoints found under training_results/unsloth_grpo_qwen3b_kaggle" >&2
      exit 1
    fi
    latest="$(printf '%s\n' "${checkpoints[@]}" | sort -V | tail -1)"
    echo "Resuming from ${latest}"
    RESUME_FROM_CHECKPOINT="${latest}" run_train training_results/unsloth_grpo_qwen3b_kaggle 160 600 8 1024 768 192 32 100
    ;;
  long)
    run_train training_results/unsloth_grpo_qwen3b_kaggle_long 206 1000 8 1024 768 192 48 100
    ;;
  fallback-1b5)
    MODEL_NAME="${FALLBACK_MODEL_NAME:-unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit}"
    run_train training_results/unsloth_grpo_kaggle_ablation 120 500 4 1024 768 192 24 100
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
    python scripts/summarize_unsloth_grpo.py training_results/unsloth_grpo_qwen3b_kaggle
    ;;
  *)
    cat >&2 <<EOF
Unknown mode: ${MODE}

Usage:
  bash scripts/run_kaggle_qwen3b_grpo.sh gpu
  bash scripts/run_kaggle_qwen3b_grpo.sh verify
  bash scripts/run_kaggle_qwen3b_grpo.sh smoke
  bash scripts/run_kaggle_qwen3b_grpo.sh main
  bash scripts/run_kaggle_qwen3b_grpo.sh resume-main
  bash scripts/run_kaggle_qwen3b_grpo.sh long
  bash scripts/run_kaggle_qwen3b_grpo.sh fallback-1b5
  bash scripts/run_kaggle_qwen3b_grpo.sh summary
  bash scripts/run_kaggle_qwen3b_grpo.sh archive
EOF
    exit 2
    ;;
esac
