"""Run the Qwen LoRA checkpoint on the RCAEval transfer prompt."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import torch

from evaluate_rcaeval_transfer import (
    DEFAULT_ANOMALY_TIMESTAMP,
    DEFAULT_EXPECTED_ROOT,
    DEFAULT_SAMPLE_URL,
    build_rca_prompt,
    download_file,
    load_metric_csv,
    parse_model_prediction,
    rank_baro_anomalies,
    rank_metric_anomalies,
    score_prediction,
    split_metric_name,
    write_plots,
)


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under {run_dir}")
    return checkpoints[-1]


def maybe_extract_artifact(archive_path: Path, extract_dir: Path) -> Path | None:
    if not archive_path.exists():
        return None
    target_run_dir = extract_dir / "training_results" / "unsloth_grpo_qwen3b_easy"
    if (target_run_dir / "checkpoint-200" / "adapter_model.safetensors").exists():
        return target_run_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_dir)
    return target_run_dir if target_run_dir.exists() else None


def resolve_run_dir(run_dir: Path, artifact_archive: Path | None, extract_dir: Path) -> Path:
    if (run_dir / "checkpoint-200" / "adapter_model.safetensors").exists() or any(run_dir.glob("checkpoint-*")):
        return run_dir

    candidate_archives: list[Path] = []
    if artifact_archive:
        candidate_archives.append(artifact_archive)
    candidate_archives.extend(
        [
            Path("artifacts/easy_grpo_qwen3b_artifacts.tar.gz"),
            Path("artifacts/models/easy_grpo_qwen3b_artifacts.tar.gz"),
            Path("/kaggle/working/easy_grpo_qwen3b_artifacts.tar.gz"),
            Path("/kaggle/working/MetaHackathon-R2/artifacts/models/easy_grpo_qwen3b_artifacts.tar.gz"),
        ]
    )

    for archive in candidate_archives:
        extracted = maybe_extract_artifact(archive, extract_dir)
        if extracted and any(extracted.glob("checkpoint-*")):
            return extracted

    searched = ", ".join(str(path) for path in candidate_archives)
    raise FileNotFoundError(
        f"Could not find checkpoints under {run_dir} and could not extract an artifact archive. "
        f"Searched archives: {searched}"
    )


def load_model(model_name: str, adapter_path: Path, max_seq_length: int):
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass
        return model, tokenizer, "unsloth"
    except Exception as unsloth_error:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = model_name.replace("unsloth/", "Qwen/")
        base_name = base_name.replace("-bnb-4bit", "")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()
        return model, tokenizer, f"transformers_fallback_after_{type(unsloth_error).__name__}"


def generate_text(model, tokenizer, prompt: str, *, max_input_tokens: int, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an SRE RCA agent. Return only valid compact JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt + "\nJSON:"

    tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, tokenized["input_ids"].shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--run-dir", type=Path, default=Path("training_results/unsloth_grpo_qwen3b_easy"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--artifact-archive", type=Path, default=None)
    parser.add_argument("--extract-dir", type=Path, default=Path("artifacts/models/easy_grpo_qwen3b_extracted"))
    parser.add_argument("--sample-url", default=DEFAULT_SAMPLE_URL)
    parser.add_argument("--data-path", type=Path, default=Path("external_data/rcaeval/simple_data.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/rcaeval_qwen_easy"))
    parser.add_argument("--expected-root", default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--anomaly-timestamp", type=int, default=DEFAULT_ANOMALY_TIMESTAMP)
    parser.add_argument("--method", choices=["baro", "delta"], default="baro")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=1400)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.artifact_archive, args.extract_dir)
    checkpoint = args.checkpoint or latest_checkpoint(run_dir)
    download_file(args.sample_url, args.data_path)
    times, columns = load_metric_csv(args.data_path)
    if args.method == "baro":
        ranks = rank_baro_anomalies(times, columns, anomaly_timestamp=args.anomaly_timestamp)
    else:
        ranks = rank_metric_anomalies(times, columns, anomaly_timestamp=args.anomaly_timestamp, window=args.window)

    prompt = build_rca_prompt(ranks, top_k=args.top_k, anomaly_timestamp=args.anomaly_timestamp)
    service_names = sorted({row["service"] for row in ranks})
    signal_names = sorted({row["signal"] for row in ranks if row["signal"]})

    model, tokenizer, load_backend = load_model(args.model_name, checkpoint, args.max_seq_length)
    completion = generate_text(
        model,
        tokenizer,
        prompt,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    prediction = parse_model_prediction(completion, service_names, signal_names)
    score = score_prediction(prediction, args.expected_root)

    summary = {
        "status": "complete",
        "mode": "qwen_adapter_generation",
        "model_name": args.model_name,
        "load_backend": load_backend,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "source": {
            "benchmark": "RCAEval public quickstart sample",
            "sample_url": args.sample_url,
            "paper_repo": "https://github.com/phamquiluan/RCAEval",
        },
        "data_path": str(args.data_path),
        "num_rows": len(times),
        "num_metrics": len(columns),
        "method": args.method,
        "expected_root": args.expected_root,
        "prediction": prediction,
        "score": score,
        "completion": completion,
        "top_metrics": ranks[: args.top_k],
        "prompt_path": str(args.out_dir / "prompt.txt"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (args.out_dir / "model_output.json").write_text(json.dumps({"completion": completion}, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["plots"] = write_plots(summary, ranks, args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
