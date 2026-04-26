import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

md1 = """# Standalone SPICE Self-Play Training (Colab)
This notebook runs the SPICE self-play pipeline for OnCallEnv.
It trains a Qwen model to act as both Attacker and Defender.
**Important:** It connects to the Hugging Face Space for environment evaluation using the `custom_spec` endpoint we added to `OnCallRedShiftEnv`."""

code1 = """!pip install -U pip setuptools wheel
!pip install unsloth trl "openenv-client>=0.2.0" pydantic requests
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""""

md2 = """## 1. Setup Local Utils
We clone the repo to reuse the Pydantic models, prompt builders, and seed scenarios. We are NOT running the environment locally."""

code2 = """import os, subprocess
from pathlib import Path

REPO_URL = "https://github.com/srimanreddy4/MetaHackathon-R2"
BRANCH = "spicy-attacker"
WORKDIR = Path("/content/MetaHackathon-R2")

if not WORKDIR.exists():
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(WORKDIR)], check=True)
else:
    os.chdir(WORKDIR)
    subprocess.run(["git", "pull", "origin", BRANCH], check=True)

import sys
if str(WORKDIR / "src") not in sys.path:
    sys.path.append(str(WORKDIR / "src"))
if str(WORKDIR / "scripts") not in sys.path:
    sys.path.append(str(WORKDIR / "scripts"))

os.chdir(WORKDIR)"""

md3 = """## 2. Remote Rollout Function
This replaces the local simulator with a remote connection to your Hugging Face Space."""

code3 = """from oncallenv.core.types import ScenarioSpec, Action
from spice_defender import parse_commands, extract_completion_text, build_rca
from llm_attacker import parse_attacker_actions, build_attacker_prompt, normalize_defender_reward
from spice_defender import build_defender_prompt
import requests

SPACE_URL = "https://huggingface.co/spaces/NeerjaK/OnCallEnv"

def remote_defender_rollout_reward(spec: ScenarioSpec, completion: str) -> float:
    text = extract_completion_text(completion)
    commands = parse_commands(text)
    if not commands:
        return -0.25

    root_service = getattr(spec, "inject_service", "api-gateway")
    root_category = getattr(spec, "fault_primary", "cpu_hog")

    # Reset environment remotely with our custom spec
    try:
        response = requests.post(f"{SPACE_URL}/reset", json={"custom_spec": spec.model_dump()}, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed remote reset: {e}")
        return -0.25

    max_reward = 0.0
    done = False
    
    for command in commands:
        try:
            resp = requests.post(f"{SPACE_URL}/step", json={"action": {"command": command}}, timeout=30)
            obs = resp.json()
            max_reward = max(max_reward, float(obs.get("reward", 0.0)))
            if obs.get("done", False):
                done = True
                break
        except Exception:
            break
            
    if not done:
        if not any(c == "declare_resolved" for c in commands):
            try:
                resp = requests.post(f"{SPACE_URL}/step", json={"action": {"command": "declare_resolved"}}, timeout=30)
                max_reward = max(max_reward, float(resp.json().get("reward", 0.0)))
                done = resp.json().get("done", False)
            except Exception:
                pass
        
        if not done:
            rca_cmd = f"submit_rca {build_rca(root_service, root_category)}"
            try:
                resp = requests.post(f"{SPACE_URL}/step", json={"action": {"command": rca_cmd}}, timeout=30)
                max_reward = max(max_reward, float(resp.json().get("reward", 0.0)))
            except Exception:
                pass

    format_bonus = 0.05 if "<actions>" in text.lower() and "</actions>" in text.lower() else 0.0
    concise_bonus = 0.03 if 2 <= len(commands) <= 8 else 0.0
    raw_reward = max_reward + format_bonus + concise_bonus
    norm_reward = (raw_reward + 0.25) / 1.35
    return max(0.0, min(1.0, norm_reward))"""

md4 = """## 3. Load Parent Scenarios & Model"""

code4 = """import random
import yaml
from pathlib import Path

# Load seed scenarios
seed_dir = Path("scenarios_seed")
parent_specs = []
for path in sorted(seed_dir.glob("*.y*ml")):
    try:
        parent_specs.append(ScenarioSpec.model_validate(yaml.safe_load(path.read_text())))
    except Exception:
        continue

print(f"Loaded {len(parent_specs)} parent scenarios")

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=1536,
    load_in_4bit=True,
    fast_inference=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad_(True)
model.train()"""

md5 = """## 4. SPICE Self-Play Loop"""

code5 = """import torch
import json
from tqdm.auto import tqdm

def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, num_return=1):
    if isinstance(prompt, str): prompt = [prompt]
    chat_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompt]
    
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = old_pad
    
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=max(temperature, 1e-4), top_p=0.95, num_return_sequences=num_return)
        
    outputs = []
    for i, seq_ids in enumerate(ids):
        input_len = inputs["input_ids"][i // num_return].shape[-1]
        outputs.append(tokenizer.decode(seq_ids[input_len:], skip_special_tokens=True))
    return outputs

def attacker_reward(defender_rewards, penalty=-0.1):
    if not defender_rewards: return penalty
    valid_r = [r for r in defender_rewards if r > 0.0]
    if not valid_r: return penalty
    mean_r = sum(valid_r)/len(valid_r)
    if mean_r == 0: return penalty
    p = mean_r
    return 1.0 * (p * (1 - p)) * 4.0

all_attacker_data, all_defender_data = [], []
GROUP_SIZE = 6

for iteration in tqdm(range(20), desc="SPICE Generation"):
    batch = random.sample(parent_specs, min(4, len(parent_specs)))
    attacker_prompts = [build_attacker_prompt(p) for p in batch]
    
    a_completions = generate_text(model, tokenizer, attacker_prompts, 200, 0.8, GROUP_SIZE)
    valid_specs = []
    
    # Process attacker
    for i, parent in enumerate(batch):
        comps = a_completions[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
        for comp in comps:
            spec, valid, actions = parse_attacker_actions(comp, parent)
            if not valid: continue
            
            # Evaluate with remote defender
            d_comps = generate_text(model, tokenizer, [build_defender_prompt(spec)], 256, 0.8, GROUP_SIZE)
            d_rewards = [remote_defender_rollout_reward(spec, dc) for dc in d_comps]
            
            all_attacker_data.append({
                "parent": parent.task_id,
                "spec": spec.model_dump(),
                "completion": comp,
                "reward": attacker_reward(d_rewards)
            })
            valid_specs.append(spec)
            
            for dc, dr in zip(d_comps, d_rewards):
                all_defender_data.append({
                    "task_id": spec.task_id,
                    "spec": spec.model_dump(),
                    "completion": dc,
                    "reward": dr
                })"""

md6 = """## 5. Build Dataset and Train"""

code6 = """from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

train_rows = []
for row in all_attacker_data:
    parent = next(s for s in parent_specs if s.task_id == row["parent"])
    train_rows.append({
        "prompt": [{"role": "user", "content": build_attacker_prompt(parent)}],
        "reward": row["reward"],
        "role": "attacker"
    })
for row in all_defender_data:
    spec = ScenarioSpec.model_validate(row["spec"])
    train_rows.append({
        "prompt": [{"role": "user", "content": build_defender_prompt(spec)}],
        "reward": row["reward"],
        "role": "defender"
    })

dataset = Dataset.from_list(train_rows)

def static_reward_fn(completions, prompt, **kwargs):
    # Retrieve the pre-computed rewards from the dataset
    return kwargs.get("reward", [0.0]*len(completions))

config = GRPOConfig(
    output_dir="spice_outputs",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=GROUP_SIZE,
    max_prompt_length=1024,
    max_completion_length=256,
    max_steps=200,
    temperature=0.8,
    logging_steps=1,
    beta=0.0,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[static_reward_fn],
    args=config,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("spice_outputs/lora_final")
tokenizer.save_pretrained("spice_outputs/lora_final")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(md2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(md4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(md5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(md6),
    nbf.v4.new_code_cell(code6),
]

with open('c:/Users/npkas/Agentic-RL/MetaHackathon-R2/notebooks/05_kaggle_spice_selfplay.ipynb', 'w') as f:
    nbf.write(nb, f)
