# Interactive ReAct Defender SFT

This is the next training lane after easy GRPO. It is designed to show a more novel learning process:

```text
observation history -> one next command -> new observation -> one next command
```

The current GRPO defender emits a full `<actions>...</actions>` block from the first alert. ReAct SFT trains the defender to behave more like a real on-call engineer.

## What It Adds

- Turn-level trajectory generation from scripted expert rollouts.
- Prompts containing the current alert, latest observation, and recent command history.
- Targets containing exactly one command in `<command>...</command>` tags.
- Interactive evaluation that lets the model run for up to 10 turns against the simulator.
- Plots for SFT loss and before/after interactive reward.

## Kaggle Commands

Generate the dataset without training:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh react-generate
```

Run the short smoke training first:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh react-sft-smoke
```

If the smoke run shows decreasing loss or improved interactive reward, run the main 100-step job:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh react-sft-main
```

Export artifacts:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh export-react-artifacts
```

## Expected Outputs

Smoke output:

```text
training_results/react_sft_qwen3b_smoke/
```

Main output:

```text
training_results/react_sft_qwen3b/
```

Each run writes:

- `react_trajectories.jsonl`
- `trajectory_metadata.json`
- `baseline_next_action.json`
- `baseline_interactive_rollouts.json`
- `trained_next_action.json`
- `trained_interactive_rollouts.json`
- `summary.json`
- `adapter/`

Plots:

```text
docs/plots/react_sft_qwen3b_loss_curve.png
docs/plots/react_sft_qwen3b_eval_bar.png
```

## How To Read The Result

The useful story is not just final reward. The result is useful if it shows:

- SFT loss decreasing over training.
- Exact next-command accuracy improving.
- Interactive rollout reward improving.
- Saved trajectories that can power the interactive UI/demo.

This gives a stronger narrative than another all-at-once model-size ablation because it shows the defender learning an incident-response process.
