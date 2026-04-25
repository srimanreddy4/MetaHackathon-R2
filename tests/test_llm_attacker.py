"""Tests for the LLM Attacker parser and reward functions."""

from __future__ import annotations

import math

import pytest

from oncallenv.core.types import ScenarioSpec

# Make sure scripts/ is importable (PYTHONPATH=src:scripts)
from llm_attacker import (
    attacker_reward,
    build_attacker_prompt,
    normalize_defender_reward,
    parse_attacker_actions,
)


PARENT = ScenarioSpec(
    task_id="test_parent",
    topology="simple_fanout",
    fault_primary="oom_kill",
    inject_service="payment-service",
    latency_ms=1000,
    blast_radius=0.3,
    metric_noise=0.1,
    seed=42,
    max_steps=25,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseAttackerActions:
    def test_single_valid_action(self):
        text = "<actions>\nset_field fault_primary deadlock\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec is not None
        assert spec.fault_primary == "deadlock"
        assert spec.topology == "simple_fanout"  # unchanged
        assert actions == ["fault_primary=deadlock"]

    def test_multiple_valid_actions(self):
        text = """<actions>
set_field fault_primary deadlock
set_field topology mesh
set_field blast_radius 0.8
</actions>"""
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.fault_primary == "deadlock"
        assert spec.topology == "mesh"
        assert spec.blast_radius == 0.8
        assert len(actions) == 3

    def test_invalid_field_name_ignored(self):
        text = "<actions>\nset_field nonexistent_field foo\nset_field topology star\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.topology == "star"
        assert len(actions) == 1

    def test_invalid_value_ignored(self):
        text = "<actions>\nset_field fault_primary totally_made_up_fault\nset_field topology star\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.fault_primary == "oom_kill"  # unchanged, invalid value skipped
        assert spec.topology == "star"

    def test_no_actions_returns_invalid(self):
        text = "I think we should change something but I'm not sure what."
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert not valid
        assert spec is None
        assert actions == []

    def test_empty_actions_block(self):
        text = "<actions>\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert not valid
        assert spec is None

    def test_optional_field_set_to_none(self):
        text = "<actions>\nset_field fault_secondary none\nset_field red_herring none\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.fault_secondary is None
        assert spec.red_herring is None

    def test_optional_field_set_to_value(self):
        text = "<actions>\nset_field fault_secondary cache_stampede\nset_field red_herring false_correlation\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.fault_secondary == "cache_stampede"
        assert spec.red_herring == "false_correlation"

    def test_numeric_field_blast_radius(self):
        text = "<actions>\nset_field blast_radius 0.9\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.blast_radius == 0.9

    def test_out_of_range_blast_radius_rejected(self):
        text = "<actions>\nset_field blast_radius 1.5\n</actions>"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert not valid  # only action was invalid → no mutations applied

    def test_unique_task_id_generated(self):
        text = "<actions>\nset_field topology mesh\n</actions>"
        spec1, _, _ = parse_attacker_actions(text, PARENT, generation=1)
        spec2, _, _ = parse_attacker_actions(text, PARENT, generation=2)
        assert spec1.task_id != spec2.task_id
        assert spec1.task_id.startswith("spice_")

    def test_without_action_tags_still_parses(self):
        text = "set_field topology diamond\nset_field fault_primary cert_expiry"
        spec, valid, actions = parse_attacker_actions(text, PARENT)
        assert valid
        assert spec.topology == "diamond"
        assert spec.fault_primary == "cert_expiry"


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


class TestNormalizeDefenderReward:
    def test_min_maps_to_zero(self):
        assert normalize_defender_reward(0.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        assert normalize_defender_reward(1.0) == pytest.approx(1.0)

    def test_zero_maps_to_zero(self):
        assert normalize_defender_reward(0.0) == 0.0

    def test_clips_below_min(self):
        assert normalize_defender_reward(-1.0) == 0.0

    def test_clips_above_max(self):
        assert normalize_defender_reward(2.0) == 1.0


class TestAttackerReward:
    def test_half_pass_gets_high_reward(self):
        # 2 pass, 2 fail -> p=0.5 -> peak reward
        # Threshold is 0.45
        rewards = [0.8, 0.9, 0.1, 0.2]
        r = attacker_reward(rewards)
        assert r > 0.9

    def test_all_pass_gets_low_reward(self):
        # p=1.0 -> far from 0.5
        rewards = [0.8, 0.9, 1.0, 0.7]
        r = attacker_reward(rewards)
        assert r < 0.2

    def test_all_fail_gets_low_reward(self):
        # p=0.0 -> far from 0.5
        rewards = [0.1, 0.2, 0.3, 0.4]
        r = attacker_reward(rewards)
        assert r < 0.2

    def test_empty_rewards_returns_penalty(self):
        assert attacker_reward([]) == -0.1

    def test_reward_is_between_zero_and_one(self):
        rewards = [0.0, 0.5, 1.0, 0.3]
        r = attacker_reward(rewards)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Prompt builder test
# ---------------------------------------------------------------------------


class TestBuildAttackerPrompt:
    def test_prompt_contains_parent_yaml(self):
        prompt = build_attacker_prompt(PARENT)
        assert "oom_kill" in prompt
        assert "payment-service" in prompt

    def test_prompt_contains_valid_fields(self):
        prompt = build_attacker_prompt(PARENT)
        assert "set_field" in prompt
        assert "topology" in prompt
        assert "fault_primary" in prompt

    def test_prompt_contains_action_tags_example(self):
        prompt = build_attacker_prompt(PARENT)
        assert "<actions>" in prompt
        assert "</actions>" in prompt
