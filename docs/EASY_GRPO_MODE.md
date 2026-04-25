# Easy GRPO Mode

The original Red Shift environment and hard GRPO reward are preserved. The current hard baseline is tagged locally as:

```text
hard-redshift-before-easy-grpo
```

Easy GRPO mode is a separate training lane designed to make the LLM learn the task format quickly and produce a higher reward curve before moving back to the hard setting.

## What Changes

Easy mode changes only the LLM training prompt/reward path in `scripts/train_unsloth_grpo.py`.

It does not replace:

- The OpenEnv environment.
- The simulator.
- The symbolic policy results.
- The hard GRPO runner modes.

## Easy Prompt

Easy prompts include runbook-style hints:

- suspected root service
- fault family
- fault-to-remediation hint
- accepted remediation command

This makes the task closer to guided imitation plus GRPO instead of blind incident discovery.

## Easy Reward

The easy reward gives dense partial credit for:

- valid `<actions>...</actions>` format
- emitting at least one command
- using read-only diagnostic tools
- mentioning the root service
- choosing the right remediation tool
- using the exact accepted remediation command
- declaring resolved
- keeping the command list concise  

It still blends in the real environment reward, so exact recovery remains valuable.

## Kaggle Commands

Run the easy smoke test first:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-smoke
```

If that works, run the easy main job:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-main
```

Summarize:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh easy-summary
```

Archive:

```bash
bash scripts/run_kaggle_qwen3b_grpo.sh archive
```

## Expected Behavior

A correct seed-memory-leak action plan scores approximately:

| Reward mode | Reward |
| --- | ---: |
| Hard | 0.768 |
| Easy | 0.937 |

The easy mode should therefore produce a much higher score than the hard 3B GRPO run, while remaining honest as a separate shaped curriculum.

