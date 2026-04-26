from pathlib import Path

from scripts.evaluate_rcaeval_transfer import (
    load_metric_csv,
    parse_model_prediction,
    rank_baro_anomalies,
    rank_metric_anomalies,
    score_prediction,
    split_metric_name,
)


def test_split_metric_name_keeps_service_prefix():
    assert split_metric_name("emailservice_mem") == ("emailservice", "mem")
    assert split_metric_name("checkout_service_latency") == ("checkout_service", "latency")


def test_metric_ranker_finds_injected_metric(tmp_path: Path):
    path = tmp_path / "sample.csv"
    rows = ["time,emailservice_mem,cartservice_latency"]
    for idx in range(10):
        rows.append(f"{idx},1,10")
    for idx in range(10, 20):
        rows.append(f"{idx},9,11")
    path.write_text("\n".join(rows), encoding="utf-8")

    times, columns = load_metric_csv(path)
    ranks = rank_metric_anomalies(times, columns, anomaly_timestamp=10, window=5)

    assert ranks[0]["metric"] == "emailservice_mem"
    score = score_prediction({"root_cause_service": ranks[0]["service"], "root_cause_signal": ranks[0]["signal"]}, "emailservice_mem")
    assert score["exact_metric_match"] is True


def test_baro_ranker_matches_public_sample_preprocessing_shape(tmp_path: Path):
    path = tmp_path / "sample.csv"
    rows = ["time,emailservice_mem,emailservice_latency-50,emailservice_latency-90,cartservice_mem"]
    for idx in range(10):
        rows.append(f"{idx},1,0.1,0.2,5")
    for idx in range(10, 20):
        rows.append(f"{idx},8,0.1,0.3,5")
    path.write_text("\n".join(rows), encoding="utf-8")

    times, columns = load_metric_csv(path)
    ranks = rank_baro_anomalies(times, columns, anomaly_timestamp=10)

    assert "emailservice_latency-50" not in columns
    assert "emailservice_latency" in columns
    assert ranks[0]["metric"] == "emailservice_mem"


def test_parse_model_prediction_accepts_json_and_text():
    services = ["emailservice", "cartservice"]
    signals = ["mem", "latency"]

    parsed = parse_model_prediction('{"root_cause_service":"emailservice","root_cause_signal":"mem"}', services, signals)
    assert parsed == {"root_cause_service": "emailservice", "root_cause_signal": "mem"}

    parsed = parse_model_prediction("The incident points to cartservice latency saturation.", services, signals)
    assert parsed == {"root_cause_service": "cartservice", "root_cause_signal": "latency"}
