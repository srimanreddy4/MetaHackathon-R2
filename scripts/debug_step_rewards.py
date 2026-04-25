# %% [markdown]
# Debugging Step Rewards in OnCallRedShiftEnv
# This script helps visualize how rewards are emitted during a Defender rollout.

# %%
import os
import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "scripts"))

from oncallenv import OnCallRedShiftEnv
from oncallenv.core.types import Action, ScenarioSpec
from oncallenv.simulation.scenario_compiler import compile_scenario
from spice_defender import build_rca

# %% [markdown]
# ### 1. Setup a Test Scenario
# We'll simulate a memory leak on the payment-service.
spec = ScenarioSpec(
    task_id="debug_reward_test",
    topology="simple_fanout",
    fault_primary="oom_kill",
    inject_service="payment-service",
    max_steps=10
)

graph = compile_scenario(spec)
root_service = graph.root_cause_service
root_category = graph.root_cause_category

print(f"Goal: Fix {root_category} on {root_service}")

# %% [markdown]
# ### 2. Step through a manual investigation
# We will run: 1 diagnostic, 1 repair, 1 resolution, and 1 RCA.

env = OnCallRedShiftEnv()
env.reset()
# Inject the specific spec
from oncallenv.core.tools import ToolRuntime
env._scenario = spec
env._runtime = ToolRuntime(graph)

commands = [
    f"kubectl_logs {root_service}",              # Diagnostic
    f"kubectl_rollout_restart {root_service}",  # Repair
    "declare_resolved",                         # Resolution
    f"submit_rca {build_rca(root_service, root_category)}"  # RCA
]

print(f"{'Command':<40} | {'Reward':<10} | {'Done':<5}")
print("-" * 60)

total_reward = 0.0
for cmd in commands:
    obs = env.step(Action(command=cmd))
    r = float(obs.reward or 0.0)
    total_reward += r
    print(f"{cmd:<40} | {r:<10.2f} | {obs.done}")

print("-" * 60)
print(f"{'TOTAL SUMMED':<40} | {total_reward:<10.2f}")
print(f"{'PEAK (MAX)':<40} | {max(total_reward, 0.0):<10.2f} (Estimated)")
