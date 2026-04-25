"""Demo: Walk through one full episode showing exactly what each component does."""

import json
from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action
from oncallenv.simulation.scenario_compiler import compile_scenario
from oncallenv.simulation.faults import FAULTS

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

def section(title, color=BLUE):
    print(f"\n{'='*80}")
    print(f"{color}{BOLD}{title}{RESET}")
    print(f"{'='*80}")

def subsection(title, color=CYAN):
    print(f"\n{color}{BOLD}>>> {title}{RESET}")

# ============================================================
# PART 1: WHAT THE CHAOS ATTACKER CREATES
# ============================================================
section("🔴 PART 1: THE CHAOS ATTACKER — What it builds", RED)

task_id = "seed_easy_memory_leak"
env = OnCallRedShiftEnv()
obs = env.reset(task_id=task_id)

# Show the scenario spec
subsection("Scenario Spec (from YAML)")
spec = env._scenario
print(f"  task_id:        {spec.task_id}")
print(f"  topology:       {spec.topology}")
print(f"  fault_primary:  {spec.fault_primary}")
print(f"  inject_service: {spec.inject_service}")
print(f"  latency_ms:     {spec.latency_ms}")
print(f"  blast_radius:   {spec.blast_radius}")
print(f"  metric_noise:   {spec.metric_noise}")
print(f"  red_herring:    {spec.red_herring}")
print(f"  deploy_window:  {spec.deploy_window}")
print(f"  max_steps:      {spec.max_steps}")

# Show what the fault primitive does
subsection("Fault Primitive being injected")
fault = FAULTS[spec.fault_primary]
print(f"  Fault name:     {fault.name}")
print(f"  Log lines it adds:")
for line in fault.log_lines:
    print(f"    → \"{line}\"")
print(f"  CPU impact:     → {fault.cpu}%")
print(f"  Memory impact:  → {fault.memory}%")
print(f"  Latency impact: → {fault.latency_p99}ms (p99)")
print(f"  Error rate:     → {fault.error_rate*100:.0f}%")
print(f"  Correct fix:    {fault.remediation}")

# Show the state of all 7 services AFTER fault injection
subsection("All 7 Services After Chaos Injection")
graph = env._runtime.graph
for name, svc in graph.services.items():
    status = "🔥 FAULTY" if svc.current_fault else "✅ HEALTHY"
    print(f"  {status} {name:25s} CPU={svc.cpu:5.1f}%  MEM={svc.memory:5.1f}%  ErrRate={svc.error_rate:.3f}  P99={svc.latency_p99:.0f}ms  Fault={svc.current_fault or 'none'}")

subsection("Required Remediations (what the agent MUST do)")
for rem in graph.required_remediations:
    print(f"  → {rem}")

subsection("Service Dependencies (the topology wiring)")
for name, svc in graph.services.items():
    deps = ", ".join(svc.dependencies) if svc.dependencies else "(none - leaf service)"
    print(f"  {name} → {deps}")

# ============================================================
# PART 2: WHAT THE DEFENDER SEES (initial observation)
# ============================================================
section("🛡️ PART 2: THE DEFENDER — What it receives on reset()", GREEN)

subsection("Initial Observation")
print(f"  Goal:            {obs.goal}")
print(f"  Done:            {obs.done}")
print(f"  Reward so far:   {obs.reward}")
print(f"  Available tools: {obs.available_tools}")
print(f"  Services:        {obs.services}")
print(f"  Alert:")
for alert in obs.alerts:
    print(f"    severity={alert.severity}  service={alert.service}  message=\"{alert.message}\"")
print(f"  Last action result: \"{obs.last_action_result}\"")

# ============================================================
# PART 3: DEFENDER TAKES ACTIONS, ENVIRONMENT RESPONDS
# ============================================================
section("🛡️↔️⚙️ PART 3: DEFENDER ACTIONS & ENVIRONMENT RESPONSES", YELLOW)

actions_to_take = [
    ("kubectl_get_pods", "Investigation: See which pods are healthy/unhealthy"),
    ("kubectl_logs payment-service", "Investigation: Read logs from the alerting service"),
    ("kubectl_top payment-service", "Investigation: Check resource usage"),
    ("promql_query payment-service", "Investigation: Check Prometheus metrics"),
    ("jaeger_search payment-service", "Investigation: Check distributed traces"),
    ("check_deploy_history payment-service", "Investigation: Was there a recent deploy?"),
    ("kubectl_logs user-service", "Investigation: Check another service (has red herring!)"),
    ("kubectl_rollout_restart payment-service", "REMEDIATION: Apply the fix!"),
    ("declare_resolved", "TERMINAL: Declare the incident resolved"),
    ('submit_rca {"root_cause_service":"payment-service","root_cause_category":"oom_kill","timeline":[{"timestamp":"2026-04-24T09:00:00Z","service":"payment-service","description":"OOM kill detected in payment-service from telemetry"}],"five_whys":["payment-service showed direct oom_kill symptoms in production telemetry","The dependency chain propagated the failure to customer-facing paths","The incident needed a targeted remediation instead of broad restarts"],"action_items":["Add regression coverage and alerting for payment-service"],"evidence_citations":[{"source":"log","ref":"kubectl_logs payment-service","excerpt":"OOMKilled"}],"blast_radius_description":"Customer-facing requests saw elevated errors before remediation."}',
     "TERMINAL: Submit root cause analysis"),
]

for i, (cmd, description) in enumerate(actions_to_take, 1):
    subsection(f"Step {i}: {description}")
    print(f"  {GREEN}DEFENDER sends:{RESET}  {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    
    obs = env.step(Action(command=cmd))
    
    result = obs.last_action_result
    # Truncate very long results for display
    if len(result) > 600:
        result_display = result[:600] + f"\n    ... ({len(result)} chars total)"
    else:
        result_display = result
    
    print(f"  {BLUE}ENV responds:{RESET}")
    for line in result_display.split("\n"):
        print(f"    {line}")
    
    print(f"  {CYAN}Episode state:{RESET}  done={obs.done}  reward={obs.reward:.4f}  elapsed={obs.time_elapsed_sec}s")
    
    if obs.reward_breakdown:
        print(f"  {CYAN}Reward breakdown:{RESET}")
        for rubric_name, score in obs.reward_breakdown.items():
            print(f"    {rubric_name}: {score:.4f}")

# ============================================================
# PART 4: THE REVIEWER — Final scoring breakdown
# ============================================================
section("📋 PART 4: THE REVIEWER — Final Scores", BLUE)

subsection("Final Reward Breakdown")
state = env.state
for rubric_name, score in state.reward_breakdown.items():
    weight_map = {
        "RecoveryRubric": 0.35,
        "RCAQualityRubric": 0.30,
        "BlastRadiusRubric": 0.25,
        "SafetyRubric": 0.10,
    }
    weight = weight_map.get(rubric_name, "?")
    weighted = score * weight if isinstance(weight, float) else "?"
    print(f"  {rubric_name:25s}  raw={score:.4f}  × weight={weight}  → contribution={weighted:.4f}" if isinstance(weight, float) else f"  {rubric_name}: {score:.4f}")

print(f"\n  {BOLD}FINAL REWARD: {obs.reward:.4f}{RESET}")

subsection("Episode Summary")
print(f"  Total steps taken:     {state.step_count}")
print(f"  Actions taken:         {state.actions_taken}")
print(f"  Resolved declared:     {state.resolved_declared}")
print(f"  RCA submitted:         {state.rca is not None}")
print(f"  Unsafe actions:        {state.unsafe_actions}")
print(f"  Actually remediated:   {state.remediated}")

# ============================================================
# PART 5: Show what happens with WRONG actions
# ============================================================
section("❌ BONUS: What happens with WRONG actions?", RED)

subsection("Episode 2: Agent applies WRONG fix + unsafe action")
env2 = OnCallRedShiftEnv()
obs2 = env2.reset(task_id=task_id)

wrong_actions = [
    ("kubectl_rollout_restart postgres-primary", "WRONG: Restarting the database (unsafe + wrong target!)"),
    ("kubectl_scale redis-cache", "WRONG: Scaling the cache (wrong fix type + wrong target!)"),
    ("declare_resolved", "Declaring resolved (but nothing was actually fixed)"),
    ('submit_rca {"root_cause_service":"redis-cache","root_cause_category":"cache_stampede","timeline":[],"five_whys":["unknown"],"action_items":["todo"],"evidence_citations":[],"blast_radius_description":""}',
     "BAD RCA: Wrong service, wrong category, generic whys"),
]

for i, (cmd, description) in enumerate(wrong_actions, 1):
    subsection(f"Wrong Step {i}: {description}")
    print(f"  {RED}DEFENDER sends:{RESET}  {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    obs2 = env2.step(Action(command=cmd))
    print(f"  {BLUE}ENV responds:{RESET}  {obs2.last_action_result}")
    print(f"  {CYAN}State:{RESET}  done={obs2.done}  reward={obs2.reward:.4f}")

subsection("Wrong Episode — Final Reward Breakdown")
state2 = env2.state
for rubric_name, score in state2.reward_breakdown.items():
    weight_map = {"RecoveryRubric": 0.35, "RCAQualityRubric": 0.30, "BlastRadiusRubric": 0.25, "SafetyRubric": 0.10}
    weight = weight_map.get(rubric_name, 0)
    print(f"  {rubric_name:25s}  raw={score:.4f}  × weight={weight}  → {score*weight:.4f}")
print(f"\n  {BOLD}FINAL REWARD (wrong actions): {obs2.reward:.4f}{RESET}")
print(f"  vs correct actions reward:  {obs.reward:.4f}")
print(f"\n  Unsafe actions committed: {state2.unsafe_actions}")
