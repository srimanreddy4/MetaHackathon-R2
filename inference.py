"""OpenEnv submission inference script for OnCallEnv Red Shift."""

import argparse
import os
import re
from typing import Any

from openai import OpenAI

# Load .env from this file's directory or any parent
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=False))
except ImportError:
    pass

from oncallenv.core.env import OnCallRedShiftEnv
from models import Action
from oncallenv.core.prompts import build_defender_prompt
from oncallenv.rewards.easy_shaping import calculate_easy_shaping_reward
from oncallenv.core.tools import MUTATING_TOOLS, READ_ONLY_TOOLS

COMMAND_RE = re.compile(
    r"\b("
    + "|".join(re.escape(tool) for tool in [*READ_ONLY_TOOLS, *MUTATING_TOOLS, "declare_resolved", "submit_rca"])
    + r")\b(?:\s+([a-z0-9_.:/={}\"'-]+))?",
    re.IGNORECASE,
)

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct:novita"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


def extract_completion_text(completion: Any) -> str:
    """Extract string content from OpenAI ChatCompletion."""
    if isinstance(completion, str):
        return completion
    if hasattr(completion, "choices") and completion.choices:
        return completion.choices[0].message.content
    return str(completion)


def parse_commands(text: str) -> list[str]:
    """Extract commands from <actions> block."""
    lower = text.lower()
    start = lower.find("<actions>")
    end = lower.find("</actions>")
    if start >= 0 and end >= 0 and end > start:
        block = text[start + 9 : end]
    else:
        block = text
    
    commands = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        match = COMMAND_RE.search(line)
        if match:
            commands.append(match.group(0))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on OnCallEnv Red Shift")
    parser.add_argument("--task-id", type=str, default=os.getenv("TASK_NAME", "seed_easy_memory_leak"), help="Task ID to run")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="OpenAI model to use")
    parser.add_argument("--api-base", type=str, default=os.getenv("API_BASE_URL"), help="OpenAI API base URL")
    parser.add_argument("--easy-mode", action="store_true", help="Enable Easy Mode (hints and shaped rewards)")
    args = parser.parse_args()

    # Initialize OpenAI Client
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(base_url=args.api_base, api_key=api_key)

    # Initialize Environment
    env = OnCallRedShiftEnv(base_url=ENV_BASE_URL)
    obs = env.reset(task_id=args.task_id)
    
    graph = env._runtime.graph
    prompt_mode = "easy" if args.easy_mode else "hard"

    # [START] emit
    print(f"[START] task={args.task_id} env=OnCallEnv-RedShift model={args.model}")

    # Build Prompt using the centralized prompt builder
    prompt_text = build_defender_prompt(
        spec=env._scenario,
        graph=graph,
        alert_message=obs.alerts[0].message,
        prompt_mode=prompt_mode,
        template="standard",
    )

    # Call LLM
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.1,
    )

    completion = extract_completion_text(response)
    commands = parse_commands(completion)
    
    if not commands:
        commands = ["declare_resolved"] # Fallback to avoid hanging

    rewards_list = []
    done = False
    step_count = 0

    # Execute commands in the environment
    for command in commands:
        step_count += 1
        obs = env.step(Action(command=command))
        done = obs.done
        
        # Calculate Reward
        raw_reward = float(obs.reward or 0.0)
        
        if args.easy_mode:
            # If easy mode is on, we calculate the shaped reward
            # Note: For strict submission, you might want to print the raw_reward, 
            # but we use shaped here to demonstrate the easy mode behavior.
            required_pairs = []
            for item in sorted(graph.required_remediations):
                if ":" in item:
                    t, s = item.split(":", 1)
                    required_pairs.append((t, s))
                    
            step_reward = calculate_easy_shaping_reward(
                completion_text=completion,
                commands=commands,
                env_reward=raw_reward,
                required_remediations=required_pairs,
                root_service=graph.root_cause_service,
                services_list=graph.service_names(),
            )
        else:
            step_reward = raw_reward
            
        rewards_list.append(step_reward)
        
        # [STEP] emit
        error_msg = "null" # You can extract exact errors from obs if needed
        if "error" in obs.last_action_result.lower() or "not found" in obs.last_action_result.lower():
             error_msg = '"' + obs.last_action_result.strip().replace('"', "'") + '"'
             
        print(f"[STEP]  step={step_count} action=\"{command}\" reward={step_reward:.2f} done={str(done).lower()} error={error_msg}")
        
        if done:
            break

    # If it didn't finish, force close
    if not done:
        env.close()

    # [END] emit
    final_score = rewards_list[-1] if rewards_list else 0.0
    success = final_score > 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END]   success={str(success).lower()} steps={step_count} score={final_score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    main()
