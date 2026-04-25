from pathlib import Path

from react_defender import generate_react_dataset, parse_command, run_interactive_rollout
from train_react_sft import truncate_token_ids


def test_react_dataset_rows_are_turn_level_and_valid():
    rows, metadata = generate_react_dataset(
        curriculum_buffer=Path("curriculum_results/buffer.json"),
        max_tasks=8,
        train_tasks=6,
        seed=123,
    )
    assert metadata["num_tasks"] == 8
    assert metadata["train_tasks"] == 6
    assert metadata["eval_tasks"] == 2
    assert rows
    first = rows[0]
    assert "Current state:" in first["prompt"]
    assert "Command history:" in first["prompt"]
    assert first["completion"].startswith("<command>")
    assert parse_command(first["completion"]) == first["target_command"]
    assert all(row["target_command"] for row in rows)


def test_scripted_react_expert_rollout_scores_high():
    commands = [
        "kubectl_get_pods",
        "kubectl_logs payment-service",
        "kubectl_top payment-service",
        "promql_query payment-service",
        "check_deploy_history payment-service",
        "kubectl_rollout_restart payment-service",
        "declare_resolved",
    ]

    def expert_command(_, turn_index: int) -> str:
        return f"<command>{commands[min(turn_index, len(commands) - 1)]}</command>"

    result = run_interactive_rollout(
        task_id="seed_easy_memory_leak",
        command_fn=expert_command,
        max_turns=10,
    )
    assert result["reward"] > 0.85
    assert result["actions"][-1] == "declare_resolved"


def test_react_prompt_token_truncation_preserves_prefix_and_recent_tail():
    input_ids = list(range(2000))
    truncated = truncate_token_ids(input_ids, 1536, head_tokens=256)
    assert len(truncated) == 1536
    assert truncated[:256] == list(range(256))
    assert truncated[-5:] == [1995, 1996, 1997, 1998, 1999]
