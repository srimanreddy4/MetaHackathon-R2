import os
import sys
import torch
import yaml
from pathlib import Path

# Add src and scripts to path so imports work
sys.path.append("src")
sys.path.append("scripts")

from unsloth import FastLanguageModel
from oncallenv.core.types import ScenarioSpec
from oncallenv.simulation.scenario_compiler import compile_scenario
from llm_attacker import build_attacker_prompt, parse_attacker_actions
from spice_defender import build_defender_prompt, defender_rollout_reward

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text

def main():
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1536,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 1 seed scenario...")
    scenarios_dir = Path("scenarios_seed")
    seed_files = list(scenarios_dir.glob("*.y*ml"))
    
    spec_dict = yaml.safe_load(seed_files[0].read_text(encoding="utf-8"))
    parent_spec = ScenarioSpec.model_validate(spec_dict)
    
    print("\n================== ATTACKER PHASE ==================")
    attacker_prompt = build_attacker_prompt(parent_spec)
    print("--- ATTACKER PROMPT ---")
    print(attacker_prompt)
    
    print("\n--- GENERATING ATTACKER COMPLETION ---")
    attacker_comp = generate_text(model, tokenizer, attacker_prompt, max_new_tokens=200)
    print("--- ATTACKER COMPLETION ---")
    print(attacker_comp)
    
    spec, is_valid, actions = parse_attacker_actions(attacker_comp, parent_spec, generation=0)
    print("\nParsed Valid Actions?", is_valid)
    print("Actions parsed:", actions)
    
    if is_valid and spec is not None:
        try:
            compile_scenario(spec)
            print("Successfully compiled mutated scenario!")
        except Exception as e:
            print(f"Failed to compile mutated scenario: {e}")
            return
            
        print("\n================== DEFENDER PHASE ==================")
        defender_prompt = build_defender_prompt(spec)
        print("--- DEFENDER PROMPT ---")
        print(defender_prompt)
        
        print("\n--- GENERATING DEFENDER COMPLETION ---")
        defender_comp = generate_text(model, tokenizer, defender_prompt, max_new_tokens=256)
        print("--- DEFENDER COMPLETION ---")
        print(defender_comp)
        
        print("\n--- EVALUATING DEFENDER ---")
        try:
            reward = defender_rollout_reward(spec, defender_comp)
            print("Defender Raw Score:", reward)
        except Exception as e:
            print("Defender Evaluation Failed:", e)

if __name__ == "__main__":
    main()
