import json

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action


def _rca(service="payment-service", category="oom_kill"):
    return json.dumps(
        {
            "root_cause_service": service,
            "root_cause_category": category,
            "timeline": [{"timestamp": "2026-04-24T09:00:00Z", "service": service, "description": "fault observed"}],
            "five_whys": [
                "The service exhausted memory during request handling",
                "The cache retained unbounded transaction objects",
                "The release lacked a guardrail for heap growth",
            ],
            "action_items": [f"Add heap guardrails and alerts for {service}"],
            "evidence_citations": [{"source": "log", "ref": "kubectl_logs", "excerpt": "OOMKilled exit code 137"}],
            "blast_radius_description": "Checkout requests saw elevated 503s.",
        }
    )


def test_perfect_run_scores_high():
    env = OnCallRedShiftEnv()
    env.reset(task_id="seed_easy_memory_leak")
    env.step(Action(command="kubectl_logs payment-service"))
    env.step(Action(command="kubectl_top payment-service"))
    env.step(Action(command="kubectl_rollout_restart payment-service"))
    env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command=f"submit_rca {_rca()}"))
    assert obs.reward > 0.85


def test_declare_without_investigating_gets_recovery_but_no_rca_quality():
    env = OnCallRedShiftEnv()
    env.reset(task_id="seed_easy_memory_leak")
    env.step(Action(command="kubectl_rollout_restart payment-service"))
    env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command="submit_rca {}"))
    assert obs.reward < 0.75
    assert obs.reward_breakdown["rubric_1"] == 0.0


def test_wrong_fix_keeps_recovery_low():
    env = OnCallRedShiftEnv()
    env.reset(task_id="seed_easy_memory_leak")
    env.step(Action(command="kubectl_rollout_restart user-service"))
    env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command=f"submit_rca {_rca()}"))
    assert obs.reward < 0.65
    assert obs.reward_breakdown["rubric_0"] < 0.5


def test_unsafe_delete_everything_style_strategy_is_penalized():
    env = OnCallRedShiftEnv()
    env.reset(task_id="seed_replica_lag")
    env.step(Action(command="kubectl_rollout_restart postgres-primary"))
    env.step(Action(command="declare_resolved"))
    obs = env.step(Action(command=f"submit_rca {_rca('postgres-primary', 'replica_lag')}"))
    assert obs.reward_breakdown["rubric_3"] <= 0.5
    assert obs.reward < 0.75

