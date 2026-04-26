"""Easy Mode reward shaping rubric for OnCallEnv Red Shift.

Provides dense partial credit for structural and diagnostic correctness
to accelerate early policy learning.
"""

from __future__ import annotations

from oncallenv.core.tools import READ_ONLY_TOOLS

def calculate_easy_shaping_reward(
    completion_text: str,
    commands: list[str],
    env_reward: float,
    required_remediations: list[tuple[str, str]],
    root_service: str,
    services_list: list[str],
) -> float:
    """Provides dense partial credit for structural and diagnostic correctness."""
    lower = completion_text.lower()
    command_set = set(commands)
    
    # Required tools and services sets
    required_tools = {tool for tool, _ in required_remediations}
    required_services = {svc for _, svc in required_remediations}
    
    # Tools and services actually seen in completion
    tools_seen = {command.split()[0] for command in commands if command.split()}
    services_seen = {svc for command in commands for svc in services_list if svc in command}

    score = 0.0
    
    # 1. Format & Structural correctness (0.18 total)
    if "<actions>" in lower and "</actions>" in lower:
        score += 0.10
    if commands:
        score += 0.08
        
    # 2. Diagnostic best practices (0.10 total)
    if any(command.split()[0] in READ_ONLY_TOOLS for command in commands if command.split()):
        score += 0.10
        
    # 3. Root Cause Identification (0.18 total)
    if root_service in services_seen:
        score += 0.18
    elif required_services & services_seen:
        score += 0.12
        
    # 4. Tool matching (0.18 total)
    if required_tools & tools_seen:
        score += 0.18
        
    # 5. Exact Remediation matches (0.28 total)
    exact_matches = 0
    for tool, service in required_remediations:
        if f"{tool} {service}" in command_set:
            exact_matches += 1
    if required_remediations:
        score += 0.28 * (exact_matches / len(required_remediations))
        
    # 6. Lifecycle (0.06 total)
    if "declare_resolved" in command_set:
        score += 0.06
        
    # 7. Conciseness (0.04 total)
    if 2 <= len(commands) <= 8:
        score += 0.04
        
    # 8. Penalty for jumping to RCA without diagnosis (negative)
    if any(command.startswith("submit_rca") for command in commands):
        score -= 0.05

    # 9. Blend with real environment reward (75% shaping, 25% ground truth)
    final_score = 0.75 * score + 0.25 * max(0.0, env_reward)
    
    return max(-0.1, min(1.0, final_score))
