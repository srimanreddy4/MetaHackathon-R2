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
  shift 9
  local extra_args=("$@")

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
    "${extra_args[@]}" \
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
  easy-smoke)
    EXTRA_ARGS=(--reward-mode easy --prompt-mode easy --prompt-variants 2)
    run_train training_results/unsloth_grpo_qwen3b_easy_smoke 40 50 4 1024 640 256 8 50 "${EXTRA_ARGS[@]}"
    ;;
  main)
    run_train training_results/unsloth_grpo_qwen3b_kaggle 160 600 8 1024 768 192 32 100
    ;;
  easy-main)
    EXTRA_ARGS=(--reward-mode easy --prompt-mode easy --prompt-variants 3)
    run_train training_results/unsloth_grpo_qwen3b_easy 120 300 8 1280 768 256 32 50 "${EXTRA_ARGS[@]}"
    ;;
  react-generate)
    python scripts/generate_react_trajectories.py \
      --curriculum-buffer curriculum_results/buffer.json \
      --out-dir training_results/react_sft_qwen3b \
      --max-tasks 120 \
      --train-tasks 100
    ;;
  react-sft-smoke)
    python scripts/train_react_sft.py \
      --model-name "${MODEL_NAME}" \
      --curriculum-buffer curriculum_results/buffer.json \
      --out-dir training_results/react_sft_qwen3b_smoke \
      --max-tasks 40 \
      --train-tasks 32 \
      --max-steps 50 \
      --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient-accumulation-steps 8 \
      --max-seq-length 1536 \
      --max-new-tokens 64 \
      --eval-action-rows 40 \
      --eval-rollout-tasks 8 \
      --max-turns 10 \
      --save-steps 50 \
      --logging-steps 5
    ;;
  react-sft-main)
    python scripts/train_react_sft.py \
      --model-name "${MODEL_NAME}" \
      --curriculum-buffer curriculum_results/buffer.json \
      --out-dir training_results/react_sft_qwen3b \
      --max-tasks 120 \
      --train-tasks 100 \
      --max-steps 100 \
      --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient-accumulation-steps 8 \
      --max-seq-length 1536 \
      --max-new-tokens 64 \
      --eval-action-rows 80 \
      --eval-rollout-tasks 20 \
      --max-turns 10 \
      --save-steps 50 \
      --logging-steps 5
    ;;
  react-eval-checkpoint)
    python scripts/evaluate_react_sft_checkpoint.py \
      --run-dir training_results/react_sft_qwen3b \
      --model-name "${MODEL_NAME}" \
      --curriculum-buffer curriculum_results/buffer.json \
      --max-tasks 120 \
      --train-tasks 100 \
      --max-seq-length 1536 \
      --max-new-tokens 64 \
      --eval-action-rows 80 \
      --eval-rollout-tasks 20 \
      --max-turns 10 \
      --plots-dir docs/plots \
      --plot-prefix react_sft_qwen3b
    ;;
  react-eval-checkpoint-fast)
    python scripts/evaluate_react_sft_checkpoint.py \
      --run-dir training_results/react_sft_qwen3b \
      --model-name "${MODEL_NAME}" \
      --curriculum-buffer curriculum_results/buffer.json \
      --max-tasks 120 \
      --train-tasks 100 \
      --max-seq-length 1536 \
      --max-new-tokens 32 \
      --eval-action-rows 24 \
      --eval-rollout-tasks 5 \
      --max-turns 7 \
      --plots-dir docs/plots \
      --plot-prefix react_sft_qwen3b_fast
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
    result_dirs=(training_results/unsloth_grpo_qwen3b_* training_results/unsloth_grpo_kaggle_ablation training_results/react_sft_qwen3b*)
    if (( ${#result_dirs[@]} == 0 )); then
      echo "No GRPO result directories found to archive." >&2
      exit 1
    fi
    tar -czf /kaggle/working/qwen3b_grpo_results.tar.gz "${result_dirs[@]}"
    ls -lh /kaggle/working/qwen3b_grpo_results.tar.gz
    ;;
  export-easy-artifacts)
    shopt -s nullglob
    easy_run="training_results/unsloth_grpo_qwen3b_easy"
    if [[ ! -d "${easy_run}" ]]; then
      echo "Missing ${easy_run}. Run easy-main first, or restore the Kaggle working directory." >&2
      exit 1
    fi
    latest_checkpoint=""
    checkpoints=("${easy_run}"/checkpoint-*)
    if (( ${#checkpoints[@]} > 0 )); then
      latest_checkpoint="$(printf '%s\n' "${checkpoints[@]}" | sort -V | tail -1)"
      echo "Latest checkpoint: ${latest_checkpoint}"
    fi

    python scripts/summarize_unsloth_grpo.py \
      "${easy_run}" \
      --write-report \
      --plots-dir docs/plots \
      --plot-prefix easy_grpo_qwen3b

    manifest="/kaggle/working/easy_grpo_qwen3b_artifact_manifest.txt"
    {
      echo "OnCallEnv Red Shift Easy GRPO artifact export"
      echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "model=${MODEL_NAME}"
      echo "run_dir=${easy_run}"
      echo "latest_checkpoint=${latest_checkpoint:-none}"
      echo
      find "${easy_run}" -maxdepth 2 -type f | sort
      find docs/plots -maxdepth 1 -type f -name 'easy_grpo_qwen3b_*.png' | sort
      [[ -f docs/easy_grpo_interrupted_report.json ]] && echo docs/easy_grpo_interrupted_report.json
    } > "${manifest}"

    tar -czf /kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz \
      "${easy_run}" \
      docs/easy_grpo_interrupted_report.json \
      docs/plots/easy_grpo_qwen3b_*.png \
      "${manifest}"
    ls -lh /kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz
    echo "Download from Kaggle: /kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz"
    ;;
  copy-easy-artifacts)
    archive="/kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz"
    dest="artifacts/models/easy_grpo_qwen3b_artifacts.tar.gz"
    if [[ ! -f "${archive}" ]]; then
      echo "Missing ${archive}. Run export-easy-artifacts first." >&2
      exit 1
    fi
    mkdir -p artifacts/models
    cp "${archive}" "${dest}"
    ls -lh "${archive}" "${dest}"
    echo "Copied archive into repo artifact folder: ${dest}"
    ;;
  export-react-artifacts)
    shopt -s nullglob
    result_dirs=(training_results/react_sft_qwen3b*)
    if (( ${#result_dirs[@]} == 0 )); then
      echo "No ReAct SFT result directories found. Run react-sft-smoke or react-sft-main first." >&2
      exit 1
    fi
    tar -czf /kaggle/working/react_sft_qwen3b_artifacts.tar.gz \
      "${result_dirs[@]}" \
      docs/plots/react_sft_qwen3b_*.png
    ls -lh /kaggle/working/react_sft_qwen3b_artifacts.tar.gz
    echo "Download from Kaggle: /kaggle/working/react_sft_qwen3b_artifacts.tar.gz"
    ;;
  summary)
    python scripts/summarize_unsloth_grpo.py training_results/unsloth_grpo_qwen3b_kaggle
    ;;
  easy-summary)
    python scripts/summarize_unsloth_grpo.py \
      training_results/unsloth_grpo_qwen3b_easy \
      --write-report \
      --plots-dir docs/plots \
      --plot-prefix easy_grpo_qwen3b
    ;;
  easy-eval-checkpoint)
    python scripts/evaluate_unsloth_checkpoint.py \
      --run-dir training_results/unsloth_grpo_qwen3b_easy \
      --model-name "${MODEL_NAME}" \
      --max-seq-length 1280 \
      --max-completion-length 256 \
      --eval-tasks 32 \
      --reward-mode easy
    ;;
  easy-analysis)
    python scripts/analyze_easy_grpo_generations.py \
      --run-dir training_results/unsloth_grpo_qwen3b_easy \
      --artifact-archive "${EASY_ARTIFACT_ARCHIVE:-/kaggle/input/datasets/srimanreddy/artifacts/easy_grpo_qwen3b_artifacts.tar.gz}" \
      --out-dir eval_results/easy_grpo_analysis
    ;;
  easy-checkpoint-curve)
    python scripts/evaluate_easy_checkpoint_curve.py \
      --run-dir training_results/unsloth_grpo_qwen3b_easy \
      --artifact-archive "${EASY_ARTIFACT_ARCHIVE:-/kaggle/input/datasets/srimanreddy/artifacts/easy_grpo_qwen3b_artifacts.tar.gz}" \
      --model-name "${MODEL_NAME}" \
      --out-dir eval_results/easy_grpo_checkpoint_curve \
      --max-seq-length 1536 \
      --max-completion-length 256 \
      --eval-tasks 32 \
      --reward-mode easy
    ;;
  rcaeval-qwen-test)
    python scripts/evaluate_rcaeval_qwen_adapter.py \
      --run-dir training_results/unsloth_grpo_qwen3b_easy \
      --artifact-archive "${RCA_ARTIFACT_ARCHIVE:-/kaggle/input/datasets/srimanreddy/artifacts/easy_grpo_qwen3b_artifacts.tar.gz}" \
      --model-name "${MODEL_NAME}" \
      --out-dir eval_results/rcaeval_qwen_easy \
      --max-seq-length 1536 \
      --max-input-tokens 1400 \
      --max-new-tokens 160
    ;;
  *)
    cat >&2 <<EOF
Unknown mode: ${MODE}

Usage:
  bash scripts/run_kaggle_qwen3b_grpo.sh gpu
  bash scripts/run_kaggle_qwen3b_grpo.sh verify
  bash scripts/run_kaggle_qwen3b_grpo.sh smoke
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-smoke
  bash scripts/run_kaggle_qwen3b_grpo.sh main
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-main
  bash scripts/run_kaggle_qwen3b_grpo.sh react-generate
  bash scripts/run_kaggle_qwen3b_grpo.sh react-sft-smoke
  bash scripts/run_kaggle_qwen3b_grpo.sh react-sft-main
  bash scripts/run_kaggle_qwen3b_grpo.sh react-eval-checkpoint
  bash scripts/run_kaggle_qwen3b_grpo.sh react-eval-checkpoint-fast
  bash scripts/run_kaggle_qwen3b_grpo.sh export-react-artifacts
  bash scripts/run_kaggle_qwen3b_grpo.sh resume-main
  bash scripts/run_kaggle_qwen3b_grpo.sh long
  bash scripts/run_kaggle_qwen3b_grpo.sh fallback-1b5
  bash scripts/run_kaggle_qwen3b_grpo.sh summary
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-summary
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-eval-checkpoint
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-analysis
  bash scripts/run_kaggle_qwen3b_grpo.sh easy-checkpoint-curve
  bash scripts/run_kaggle_qwen3b_grpo.sh rcaeval-qwen-test
  bash scripts/run_kaggle_qwen3b_grpo.sh export-easy-artifacts
  bash scripts/run_kaggle_qwen3b_grpo.sh copy-easy-artifacts
  bash scripts/run_kaggle_qwen3b_grpo.sh archive
EOF
    exit 2
    ;;
esac
