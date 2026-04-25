#!/usr/bin/env bash
# ============================================================================
# SPICE Self-Play Training — Convenience wrapper for Kaggle / Colab
#
# Usage:  bash scripts/run_spice_selfplay.sh
# ============================================================================
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src:scripts"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "=== SPICE Self-Play Training ==="
echo "Model:       unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
echo "Output:      training_results/spice_selfplay/"
echo ""

python scripts/train_spice_selfplay.py \
  --model-name "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" \
  --out-dir "training_results/spice_selfplay" \
  --seed-dir "scenarios_seed" \
  --curriculum-buffer "curriculum_results/buffer.json" \
  --selfplay-iterations 200 \
  --group-size 4 \
  --batch-size 8 \
  --max-steps 600 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-seq-length 1536 \
  --max-prompt-length 1024 \
  --max-completion-length 256 \
  --max-attacker-tokens 200 \
  --max-defender-tokens 256 \
  --lr 5e-6 \
  --temperature 0.8 \
  --report-to "tensorboard" \
  --lora-rank 16 \
  --lora-alpha 32 \
  --eval-tasks 12 \
  --max-tasks 120 \
  --logging-steps 1 \
  --save-steps 50 \
  --seed 20260424 
  
echo ""
echo "=== Done. Results in training_results/spice_selfplay/ ==="
