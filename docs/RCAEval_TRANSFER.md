# RCAEval Transfer Smoke Test

This repo includes a lightweight public-benchmark transfer adapter for RCAEval:

```bash
PYTHONPATH=src:scripts python scripts/evaluate_rcaeval_transfer.py
```

The default run downloads RCAEval's public quickstart metric sample from:

```text
https://github.com/phamquiluan/baro/releases/download/0.0.4/simple_data.csv
```

It then:

- builds an RCA prompt from the top anomalous metrics,
- runs a small BARO-compatible telemetry ranker baseline for the quickstart sample,
- scores against RCAEval's quickstart expected root metric, `emailservice_mem`,
- writes a prompt, summary JSON, and plots under `eval_results/rcaeval_transfer/`.

Artifacts:

```text
eval_results/rcaeval_transfer/prompt.txt
eval_results/rcaeval_transfer/summary.json
eval_results/rcaeval_transfer/rcaeval_sample_metric_ranking.png
eval_results/rcaeval_transfer/rcaeval_sample_transfer_score.png
```

To score a model generation from Kaggle or another GPU runtime, write its RCA answer into a JSON/text file and run:

```bash
PYTHONPATH=src:scripts python scripts/evaluate_rcaeval_transfer.py \
  --model-output path/to/model_output.json
```

Accepted model output formats include raw text or JSON with one of:

```json
{
  "root_cause_service": "emailservice",
  "root_cause_signal": "mem",
  "evidence": ["emailservice_mem has the largest post-anomaly shift"]
}
```

This is intentionally a smoke test, not the full RCAEval benchmark. Full RCAEval RE1/RE2/RE3 archives are much larger, ranging from hundreds of MB to several GB, so they should be run deliberately on a server or Kaggle session.
