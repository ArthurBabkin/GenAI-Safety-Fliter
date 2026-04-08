# Final Report: Fast and Resource-Efficient Safety Filters for LLM Outputs

**Authors:** Arthur Babkin, Alexander Malyy | **Course:** Generative AI, Spring 2026
**Repo:** [github.com/ArthurBabkin/GenAI-Safety-Fliter](https://github.com/ArthurBabkin/GenAI-Safety-Fliter)

---

## 1. Problem Statement

Post-generation safety filters for LLMs must balance detection quality against inference speed. Lightweight models are fast but miss subtle toxicity; expressive models catch more but cost orders of magnitude more in latency. We systematically compare classical and neural safety filters under a unified evaluation protocol, with specific focus on multilingual (Russian + English) content.

## 2. Data

See [midterm report](../midterm/midterm.md) for full data description. In brief: 460,303 multilingual samples (56% Russian, 44% English) from four public datasets, cleaned and LSH-deduplicated. Stratified 80/10/10 train/val/test split, seed=42.

## 3. Models

See [midterm report](../midterm/midterm.md) for full model descriptions. Three models compared:

- **TF-IDF + LogReg:** 10K TF-IDF features, L2 logistic regression
- **DistilBERT (full finetune):** `distilbert-base-uncased`, 3 epochs, lr=2e-5
- **DistilBERT + LoRA:** same base, LoRA r=8 alpha=16, 2 epochs, lr=3e-4, trained on 90k subset

## 4. Baseline Results

All models evaluated on the same 92,061-sample test set (~20% toxic), tuned thresholds.

| Model | Precision | Recall | F1 | PR-AUC | Latency (ms) | Throughput |
|-------|-----------|--------|----|--------|-------------|------------|
| TF-IDF + LogReg | 0.8098 | 0.7243 | 0.7647 | 0.8560 | 0.009 | 109K/s |
| DistilBERT | **0.9430** | **0.9208** | **0.9318** | **0.9781** | 3.58 | 287/s |
| DistilBERT + LoRA | 0.8649 | 0.7320 | 0.7929 | 0.8853 | 3.27 | 316/s |

## 5. Ablation Studies

The midterm report covered class imbalance ablations (undersampling). Two additional ablations were run for the final report.

### 5.1 Architecture: LoRA vs Full Finetune on 90k

**Question:** Is the quality gap between LoRA and full finetune (13.9pp F1) caused by less training data (90k vs 331k) or by the adapter approach itself?

**Setup:** Both models trained on the same 90k split (the LoRA baseline's `data_splits.pkl`). Only the architecture changes.

![Architecture ablation](../images/90k_dataset_ablation.png)
*Figure 1. LoRA vs full finetune trained on the same 90k samples.*

| Model | Precision | Recall | F1 | PR-AUC | Threshold |
|-------|-----------|--------|----|--------|-----------|
| LoRA (90k) | 0.8649 | 0.7320 | 0.7929 | 0.8853 | 0.49 |
| Full finetune (90k) | **0.9069** | **0.8241** | **0.8635** | **0.9354** | 0.58 |
| Delta | +0.0420 | +0.0921 | **+0.0706** | +0.0501 | |

Full finetune wins by **+7.1pp F1** on identical data. The recall gap is especially large (+9.2pp) — LoRA misses significantly more toxic examples even without the data disadvantage.

**Interpretation.** The 13.9pp gap from the main comparison breaks down roughly in half:
- ~7pp from architecture (LoRA only updates 1% of parameters)
- ~7pp from data size (90k vs 331k)

LoRA pays a quality cost on both fronts, not just one. Its value is training efficiency, not inference quality.

### 5.2 Class Weighting: Full Finetune on 331k

**Question:** Can class-weighted loss (~4x weight on toxic examples) recover the benefit of balanced training without the data loss penalty? The undersampling ablation from the midterm showed -4.7pp F1 due to discarding 60% of training data.

**Setup:** Same 331k splits as the full finetune baseline, same architecture. Only the loss function changes: CrossEntropyLoss with weights inversely proportional to class frequency (safe=0.625, toxic=2.493).

![Class weighting ablation](../images/class_weighting_ablation.png)
*Figure 2. Full finetune baseline vs class-weighted loss on 331k.*

| Model | Precision | Recall | F1 | PR-AUC | Threshold |
|-------|-----------|--------|----|--------|-----------|
| Baseline (331k) | **0.9430** | **0.9208** | **0.9318** | **0.9781** | 0.53 |
| Class-weighted (331k) | 0.9064 | 0.8818 | 0.8939 | 0.9567 | 0.83 |
| Delta | -0.0367 | -0.0390 | **-0.0379** | -0.0214 | |

Class weighting **hurt** the model by -3.8pp F1.

**Interpretation.** The key signal is the threshold shift from 0.53 to 0.83 — the ~4x loss weight on toxic examples pushed the model to output inflated probabilities for the toxic class, breaking its calibration. PR-AUC dropped -2.1pp independently of any threshold effect, confirming the ranking quality itself degraded.

The root cause: the full finetune baseline already achieves 92% recall with no corrections. There is no majority-class bias to fix. Upweighting the minority class only distorts what the model already learned well.

This matches the undersampling result — both approaches assume the model is biased by imbalance, but for a pretrained transformer on 331k samples that assumption is wrong:

| Approach | F1 | Delta | Mechanism |
|----------|-----|-------|-----------|
| Baseline | 0.9318 | — | — |
| Undersampling (133k, 50%) | 0.8844 | -4.7pp | Data loss |
| Class weighting (331k, weighted) | 0.8939 | -3.8pp | Gradient distortion |

For LogReg the story was the opposite (+4.1pp from balancing) because a linear model on sparse TF-IDF features is sensitive to class priors. A pretrained transformer on 331k examples is not.

### 5.3 Summary

| Ablation | Model | Baseline F1 | Ablation F1 | Delta | Finding |
|----------|-------|------------|-------------|-------|---------|
| Architecture (90k) | LoRA → Finetune | 0.7929 | 0.8635 | +7.1pp | Architecture explains half the LoRA–finetune gap |
| Class weighting (331k) | Finetune → Finetune+CW | 0.9318 | 0.8939 | -3.8pp | Full finetune has no imbalance bias to correct |

## 6. Deployment Recommendations

| Use case | Model | Rationale |
|----------|-------|-----------|
| Real-time moderation (CPU) | TF-IDF + LogReg | 0.009ms latency, CPU-only, 0.5 MB |
| Batch safety review | DistilBERT | Best quality (F1=0.93), 287 samples/sec |
| Rapid prototyping / fast training | LoRA | 6x faster training, ~1% trainable params |

LoRA's quality (F1=0.79) does not justify its transformer-level latency for inference. Its value is in training efficiency.

## References

1. Jigsaw Toxic Comment Classification Challenge. *Kaggle.*
2. Blackmoon. *Russian Language Toxic Comments Dataset.* Kaggle.
3. Semiletov, A. *Toxic Russian Comments Dataset.* Kaggle.
4. Abusaqer, M. *Combined Hate Speech Dataset.* Kaggle.
5. Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT, 2019.
6. Sanh, V., et al. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS Workshop, 2019.
7. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
8. Schmidt, A., & Wiegand, M. *A Survey on Hate Speech Detection using NLP.* SocialNLP Workshop, 2017.
