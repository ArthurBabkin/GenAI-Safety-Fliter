# Threshold Tuning Results

Test set: 92,061 samples (~20% toxic), stratified split with seed=42.
Threshold tuned on validation set (36,825 samples) by sweeping 0.10–0.90 and maximizing F1.

## TF-IDF + Logistic Regression

| Metric | Default (t=0.50) | Tuned (t=0.32) | Delta |
|---|---|---|---|
| Precision | 0.9102 | 0.8098 | -0.1004 |
| Recall | 0.6399 | 0.7243 | +0.0844 |
| F1 | 0.7515 | 0.7647 | +0.0132 |
| PR-AUC | 0.8560 | 0.8560 | — |

Confusion matrices (TN / FP / FN / TP):
- Default: 72,428 / 1,166 / 6,650 / 11,817
- Tuned:   70,453 / 3,141 / 5,092 / 13,375

Latency: 0.009 ms/sample | Throughput: 109,049 samples/sec

Threshold tuning trades ~10pp precision for ~8pp recall, netting +1.3pp F1. The default threshold is too conservative — misses 36% of toxic content. Lowering to 0.32 catches 1,558 more toxic samples at the cost of 1,975 additional false positives.

## DistilBERT

| Metric | Default (t=0.50) | Tuned (t=0.53) | Delta |
|---|---|---|---|
| Precision | 0.9394 | 0.9430 | +0.0036 |
| Recall | 0.9234 | 0.9208 | -0.0026 |
| F1 | 0.9313 | 0.9318 | +0.0005 |
| PR-AUC | 0.9781 | 0.9781 | — |

Confusion matrices (TN / FP / FN / TP):
- Default: 72,494 / 1,100 / 1,415 / 17,052
- Tuned:   72,567 / 1,027 / 1,462 / 17,005

Latency: 3.58 ms/sample | Throughput: 287 samples/sec

Threshold tuning barely moves anything — the default 0.50 is already near-optimal (tuned lands at 0.53). The transformer's probability calibration is much better than logreg's.

## DistilBERT + LoRA

| Metric | Default (t=0.50) | Tuned (t=0.49) | Delta |
|---|---|---|---|
| Precision | 0.8693 | 0.8649 | -0.0044 |
| Recall | 0.7281 | 0.7320 | +0.0039 |
| F1 | 0.7925 | 0.7929 | +0.0004 |
| PR-AUC | 0.8853 | 0.8853 | — |

Confusion matrices (TN / FP / FN / TP):
- Default: 71,573 / 2,021 / 5,023 / 13,444
- Tuned:   71,482 / 2,112 / 4,950 / 13,517

Latency: 3.27 ms/sample | Throughput: 316 samples/sec

Trained on 90k subset (vs 331k for full fine-tuning). Threshold tuning is negligible (0.49 vs 0.50) — LoRA's calibration is good. Quality sits between logreg and full fine-tuning, with similar inference cost to the full transformer.

## Cross-Model Comparison (tuned thresholds)

| Metric | LogReg (t=0.32) | LoRA (t=0.49) | DistilBERT (t=0.53) |
|---|---|---|---|
| Precision | 0.8098 | 0.8649 | 0.9430 |
| Recall | 0.7243 | 0.7320 | 0.9208 |
| F1 | 0.7647 | 0.7929 | 0.9318 |
| PR-AUC | 0.8560 | 0.8853 | 0.9781 |
| Latency (ms) | 0.009 | 3.27 | 3.58 |
| Throughput (s/s) | 109,049 | 316 | 287 |
| Train samples | 331,417 | 90,000 | 331,417 |

DistilBERT dominates on quality (+17pp F1 over logreg, +14pp over LoRA) at the cost of ~380x slower inference than logreg. LoRA provides a marginal quality gain over logreg (+2.8pp F1) but at transformer-level latency — not a good trade-off in this setup. For a safety filter where missing toxic content is high-risk, full fine-tuning is the clear winner if latency budget allows ~4ms per sample.
