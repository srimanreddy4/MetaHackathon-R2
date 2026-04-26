"""Small RCAEval transfer smoke test.

This script intentionally starts with RCAEval's public quickstart metric sample
instead of the multi-GB full benchmark archives. It gives us a fast external
sanity check and a prompt/scorer path for testing a Red Shift model on public
microservice RCA telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE_URL = "https://github.com/phamquiluan/baro/releases/download/0.0.4/simple_data.csv"
DEFAULT_EXPECTED_ROOT = "emailservice_mem"
DEFAULT_ANOMALY_TIMESTAMP = 1692569339


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _to_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def load_metric_csv(path: Path, *, rcaeval_preprocess: bool = True) -> tuple[list[float], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows")

    time_column = "time" if "time" in rows[0] else reader.fieldnames[0] if reader.fieldnames else "time"
    times: list[float] = []
    columns: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        times.append(_to_float(row.get(time_column, "")) or float(index))
        for key, raw in row.items():
            if key == time_column or key == "time.1":
                continue
            metric_name = key
            if rcaeval_preprocess and metric_name.endswith("latency-50"):
                continue
            if rcaeval_preprocess:
                metric_name = metric_name.replace("_latency-90", "_latency")
            value = _to_float(raw)
            if value is None:
                continue
            columns.setdefault(metric_name, []).append(value)

    row_count = len(rows)
    return times, {name: values for name, values in columns.items() if len(values) == row_count}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * ratio
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - position) + sorted_values[high] * (position - low)


def window_indices(times: list[float], anomaly_timestamp: int, window: int) -> tuple[list[int], list[int]]:
    split = next((idx for idx, value in enumerate(times) if value >= anomaly_timestamp), len(times) * 2 // 3)
    before = list(range(max(0, split - window), split))
    after = list(range(split, min(len(times), split + window)))
    if not before or not after:
        split = len(times) * 2 // 3
        before = list(range(max(0, split - window), split))
        after = list(range(split, min(len(times), split + window)))
    return before, after


def split_metric_name(metric: str) -> tuple[str, str]:
    if "_" not in metric:
        return metric, ""
    service, signal = metric.rsplit("_", 1)
    return service, signal


def rank_metric_anomalies(
    times: list[float],
    columns: dict[str, list[float]],
    *,
    anomaly_timestamp: int,
    window: int,
) -> list[dict[str, Any]]:
    before_idx, after_idx = window_indices(times, anomaly_timestamp, window)
    ranks: list[dict[str, Any]] = []
    for metric, values in columns.items():
        before = [values[idx] for idx in before_idx]
        after = [values[idx] for idx in after_idx]
        before_mean = mean(before)
        after_mean = mean(after)
        baseline_std = max(std(before), 1e-9)
        score = abs(after_mean - before_mean) / baseline_std
        service, signal = split_metric_name(metric)
        ranks.append(
            {
                "metric": metric,
                "service": service,
                "signal": signal,
                "score": score,
                "before_mean": before_mean,
                "after_mean": after_mean,
                "delta": after_mean - before_mean,
            }
        )
    return sorted(ranks, key=lambda row: row["score"], reverse=True)


def rank_baro_anomalies(
    times: list[float],
    columns: dict[str, list[float]],
    *,
    anomaly_timestamp: int,
) -> list[dict[str, Any]]:
    """Approximate RCAEval BARO's metric quickstart ranking.

    RCAEval's BARO baseline preprocesses the sample by dropping p50 latency,
    renaming p90 latency, fitting a robust scaler on all pre-injection data,
    and sorting metrics by the maximum positive robust z-score after injection.
    Reimplementing that tiny path avoids pulling the full RCAEval dependency
    stack just for the public sample smoke test.
    """

    split = next((idx for idx, value in enumerate(times) if value >= anomaly_timestamp), len(times) * 2 // 3)
    before_idx = list(range(0, split))
    after_idx = list(range(split, len(times)))
    if not before_idx or not after_idx:
        before_idx, after_idx = window_indices(times, anomaly_timestamp, max(1, len(times) // 3))

    ranks: list[dict[str, Any]] = []
    for metric, values in columns.items():
        before = [values[idx] for idx in before_idx]
        after = [values[idx] for idx in after_idx]
        median = percentile(before, 0.5)
        q1 = percentile(before, 0.25)
        q3 = percentile(before, 0.75)
        iqr = max(q3 - q1, 1e-12)
        robust_scores = [(value - median) / iqr for value in after]
        score = max(robust_scores) if robust_scores else 0.0
        service, signal = split_metric_name(metric)
        ranks.append(
            {
                "metric": metric,
                "service": service,
                "signal": signal,
                "score": score,
                "before_mean": mean(before),
                "after_mean": mean(after),
                "delta": mean(after) - mean(before),
            }
        )
    return sorted(ranks, key=lambda row: row["score"], reverse=True)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_model_prediction(text: str, service_names: list[str], signal_names: list[str]) -> dict[str, str]:
    service = ""
    signal = ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            service = str(parsed.get("root_cause_service", parsed.get("service", "")))
            signal = str(parsed.get("root_cause_signal", parsed.get("root_cause_category", parsed.get("signal", ""))))
    except json.JSONDecodeError:
        pass

    normalized_text = normalize_name(text)
    if not service:
        for candidate in service_names:
            if normalize_name(candidate) in normalized_text:
                service = candidate
                break
    if not signal:
        for candidate in signal_names:
            if normalize_name(candidate) in normalized_text:
                signal = candidate
                break
    return {"root_cause_service": service, "root_cause_signal": signal}


def build_rca_prompt(ranks: list[dict[str, Any]], *, top_k: int, anomaly_timestamp: int) -> str:
    lines = [
        "You are an SRE root-cause analysis agent.",
        "Given public RCAEval microservice telemetry, identify the single most likely root-cause service and signal.",
        "Return only compact JSON with keys root_cause_service, root_cause_signal, evidence.",
        f"Anomaly timestamp: {anomaly_timestamp}",
        "Top anomalous metric deltas:",
    ]
    for row in ranks[:top_k]:
        lines.append(
            "- "
            f"{row['metric']}: score={row['score']:.3f}, "
            f"before_mean={row['before_mean']:.4f}, after_mean={row['after_mean']:.4f}, delta={row['delta']:.4f}"
        )
    return "\n".join(lines)


def score_prediction(prediction: dict[str, str], expected_root: str) -> dict[str, Any]:
    expected_service, expected_signal = split_metric_name(expected_root)
    service_match = normalize_name(prediction.get("root_cause_service", "")) == normalize_name(expected_service)
    signal_match = normalize_name(prediction.get("root_cause_signal", "")) == normalize_name(expected_signal)
    return {
        "expected_root": expected_root,
        "expected_service": expected_service,
        "expected_signal": expected_signal,
        "predicted_service": prediction.get("root_cause_service", ""),
        "predicted_signal": prediction.get("root_cause_signal", ""),
        "service_match": service_match,
        "signal_match": signal_match,
        "exact_metric_match": service_match and signal_match,
    }


def read_model_output(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"{path} is empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        for key in ("output", "text", "generation", "prediction", "completion"):
            if key in parsed:
                return str(parsed[key])
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict):
            for key in ("output", "text", "generation", "prediction", "completion"):
                if key in first:
                    return str(first[key])
    return raw


def write_plots(summary: dict[str, Any], ranks: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    top = ranks[:10]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [row["metric"] for row in top]
    values = [float(row["score"]) for row in top]
    colors = ["#2E7D32" if label == summary["expected_root"] else "#1565C0" for label in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("RCAEval Public Sample: Top Metric Anomalies")
    ax.set_ylabel("Delta z-score")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "rcaeval_sample_metric_ranking.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["service match", "signal match", "exact metric"]
    values = [
        float(summary["score"]["service_match"]),
        float(summary["score"]["signal_match"]),
        float(summary["score"]["exact_metric_match"]),
    ]
    ax.bar(labels, values, color=["#2E7D32", "#00897B", "#6A1B9A"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("RCAEval Public Sample Transfer Score")
    for index, value in enumerate(values):
        ax.text(index, value + 0.03, f"{value:.0f}", ha="center")
    fig.tight_layout()
    path = out_dir / "rcaeval_sample_transfer_score.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-url", default=DEFAULT_SAMPLE_URL)
    parser.add_argument("--data-path", type=Path, default=Path("external_data/rcaeval/simple_data.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/rcaeval_transfer"))
    parser.add_argument("--expected-root", default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--anomaly-timestamp", type=int, default=DEFAULT_ANOMALY_TIMESTAMP)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--method", choices=["baro", "delta"], default="baro")
    parser.add_argument("--model-output", type=Path, default=None, help="Optional JSON/text generation to score instead of the metric ranker.")
    args = parser.parse_args()

    download_file(args.sample_url, args.data_path)
    times, columns = load_metric_csv(args.data_path)
    if args.method == "baro":
        ranks = rank_baro_anomalies(times, columns, anomaly_timestamp=args.anomaly_timestamp)
    else:
        ranks = rank_metric_anomalies(times, columns, anomaly_timestamp=args.anomaly_timestamp, window=args.window)
    if not ranks:
        raise ValueError("No numeric metrics found in RCAEval sample.")

    service_names = sorted({row["service"] for row in ranks})
    signal_names = sorted({row["signal"] for row in ranks if row["signal"]})
    prompt = build_rca_prompt(ranks, top_k=args.top_k, anomaly_timestamp=args.anomaly_timestamp)

    if args.model_output:
        mode = "scored_model_output"
        model_text = read_model_output(args.model_output)
        prediction = parse_model_prediction(model_text, service_names, signal_names)
    else:
        mode = f"{args.method}_telemetry_ranker_baseline"
        best = ranks[0]
        model_text = ""
        prediction = {"root_cause_service": best["service"], "root_cause_signal": best["signal"]}

    score = score_prediction(prediction, args.expected_root)
    summary: dict[str, Any] = {
        "status": "complete",
        "mode": mode,
        "source": {
            "benchmark": "RCAEval public quickstart sample",
            "sample_url": args.sample_url,
            "paper_repo": "https://github.com/phamquiluan/RCAEval",
        },
        "data_path": str(args.data_path),
        "num_rows": len(times),
        "num_metrics": len(columns),
        "anomaly_timestamp": args.anomaly_timestamp,
        "window": args.window,
        "method": args.method,
        "expected_root": args.expected_root,
        "prediction": prediction,
        "score": score,
        "top_metrics": ranks[: args.top_k],
        "prompt_path": str(args.out_dir / "prompt.txt"),
        "model_output": model_text,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["plots"] = write_plots(summary, ranks, args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
