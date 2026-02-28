# GenAI Safety Filter

Fast and resource-efficient safety filters for LLM outputs.

## Project Overview

This project systematically compares classical and neural safety filters under a unified evaluation protocol, focusing on how model capacity, preprocessing, and decision rules affect inference speed, memory usage, and detection quality.

### Team
- Arthur Babkin
- Alexander Malyy

**Course:** Generative AI, Spring 2026

## Problem Statement

Large language models are commonly deployed with post-generation safety filters that detect harmful or toxic text. The core problem is the **trade-off between speed/resource usage and detection quality**:

- Lightweight models are fast and cheap but may miss subtle harmful content
- More expressive models achieve higher quality but increase latency, memory usage, and compute cost

This project aims to build resource-efficient safety mechanisms suitable for CPU-only or edge deployments, with specific focus on **Russian-language content**.

## Dataset

The project uses a combined dataset from multiple public sources, totaling **470,322 samples** with binary classification (`safe` / `toxic`).

### Dataset Statistics

| Source | Samples | Toxicity Rate |
|--------|---------|---------------|
| Toxic Russian | 248,290 | 17.96% |
| Jigsaw | 159,571 | 10.17% |
| Hate Speech | 48,049 | 57.78% |
| Russian Toxic | 14,412 | 33.49% |
| **Total** | **470,322** | **19.86%** |

### Data Sources

1. **Jigsaw Toxic Comment Classification Challenge** ([Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge))
2. **Russian Language Toxic Comments** ([Kaggle](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments))
3. **Toxic Russian Comments** ([Kaggle](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments))
4. **Combined Hate Speech Dataset** ([Kaggle](https://www.kaggle.com/datasets/mahmoudabusaqer/combined-hate-speech-dataset))

### Language Distribution

| Language | Samples | Percentage |
|----------|---------|------------|
| Russian  | 262,165 | 55.7% |
| English  | 208,148 | 44.3% |

![Data Distribution](data/plots/data_distribution.png)

## Models

Three models are trained and evaluated under identical conditions (80/20 train/test split, same test set of 94K samples).

### Results Summary

| Model | F1 | Precision | Recall | PR-AUC | Latency (ms) | Throughput (samples/s) | Size |
|-------|-----|-----------|--------|--------|---------------|----------------------|------|
| TF-IDF + LogReg | 0.7485 | 0.9275 | 0.6275 | 0.8432 | 0.018 | 58,392 | ~0.5 MB |
| DistilBERT (fine-tuned) | 0.8900 | 0.9082 | 0.8725 | 0.9533 | 3.08 | 334 | 255 MB |
| DistilBERT + LoRA | 0.7632 | 0.8239 | 0.7108 | 0.8534 | 3.15 | 322 | 255 MB |

![Quality Comparison](docs/images/quality_comparison.png)

### 1. TF-IDF + Logistic Regression (Baseline)

- Classical bag-of-words approach with 10K TF-IDF features (unigrams + bigrams)
- Extremely fast inference (~58K samples/sec), minimal memory, ~0.5 MB on disk
- High precision (0.93) but lower recall (0.63) — misses implicit toxicity

### 2. DistilBERT (Full Fine-tuning)

- `distilbert-base-uncased` (67.6M parameters), all parameters trained
- 3 epochs on 376K samples, lr=2e-5, batch_size=64
- Best quality (F1=0.89), but ~170x slower than TF-IDF
- Strong contextual understanding of implicit and multilingual toxicity

### 3. DistilBERT + LoRA (Parameter-Efficient)

- Same base model, but only 665K trainable parameters (0.98% of total)
- LoRA rank=4, alpha=16, target modules: `q_lin`, `v_lin`
- 2 epochs on 100K subset, lr=3e-4 — trains ~6x faster than full fine-tuning
- Achieves 86% of full fine-tuning quality with dramatically lower training cost

![Performance Comparison](docs/images/performance_comparison.png)

## Project Structure

```
GenAI-Safety-Fliter/
├── data/
│   ├── train_dataset.csv              # Combined dataset (470K samples)
│   ├── original/                      # Raw source datasets
│   ├── plots/                         # Data visualization plots
│   └── models/
│       ├── logreg/                    # TF-IDF + LogReg artifacts
│       │   ├── tfidf_vectorizer.pkl
│       │   ├── logreg_model.pkl
│       │   ├── test_data.pkl
│       │   └── evaluation_results.json
│       ├── transformer/               # DistilBERT fine-tuned
│       │   ├── model.safetensors
│       │   ├── config.json
│       │   ├── tokenizer.json
│       │   ├── test_data.pkl
│       │   └── evaluation_results.json
│       └── transformer_lora/          # DistilBERT + LoRA
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           ├── test_data.pkl
│           └── evaluation_results.json
├── docs/
│   ├── baseline/
│   │   └── baseline.md                # Baseline report with full analysis
│   ├── proposal/
│   │   ├── proposal.md
│   │   └── proposal.pdf
│   ├── qualitative_samples.md         # Qualitative error analysis
│   └── images/                        # Report visualizations
├── model/
│   ├── __init__.py                    # Package exports
│   ├── models.py                      # Model implementations
│   ├── metrics.py                     # Evaluation framework
│   └── experiments/
│       ├── tf_idf.ipynb               # TF-IDF baseline notebook
│       ├── transformer.ipynb          # DistilBERT fine-tuning notebook
│       └── transformer_lora.ipynb     # LoRA fine-tuning notebook
├── preprocess.ipynb                   # Data preprocessing
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.11+

### Installation

```bash
git clone https://github.com/ArthurBabkin/GenAI-Safety-Fliter.git
cd GenAI-Safety-Fliter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Loading Pre-trained Models

```python
from model import LogRegModel, TransformerClassifier, LoRATransformerClassifier

# TF-IDF + Logistic Regression
logreg = LogRegModel(
    vectorizer_path="data/models/logreg/tfidf_vectorizer.pkl",
    model_path="data/models/logreg/logreg_model.pkl"
)

# DistilBERT (full fine-tuning)
bert = TransformerClassifier(model_dir="data/models/transformer")

# DistilBERT + LoRA
lora = LoRATransformerClassifier(model_dir="data/models/transformer_lora")
```

### Prediction

```python
texts = ["You are stupid", "Have a nice day"]

predictions = logreg.predict(texts)       # [1, 0]
probabilities = logreg.predict_proba(texts)  # [[p_safe, p_toxic], ...]
```

### Evaluation

```python
metrics = logreg.get_metrics(X_test, y_test, n_latency_runs=100)
# Returns: quality (precision, recall, f1, pr_auc), confusion_matrix,
#          latency stats, throughput, peak_memory_mb
```

### Training from Scratch

Run the experiment notebooks in order:

```bash
jupyter notebook model/experiments/tf_idf.ipynb
jupyter notebook model/experiments/transformer.ipynb
jupyter notebook model/experiments/transformer_lora.ipynb
```

## Key Findings

- **Speed vs Quality:** A ~19% F1 improvement (0.75 → 0.89) costs ~170x in latency
- **LoRA Efficiency:** 86% of full fine-tuning quality with 6x faster training and <1% trainable parameters
- **TF-IDF Strengths:** Best for high-throughput, CPU-only deployments where false negatives are acceptable
- **Transformer Strengths:** Superior at catching implicit toxicity and context-dependent harmful language
- **Model Size:** TF-IDF is ~500x smaller on disk (0.5 MB vs 255 MB)

See the full [baseline report](docs/baseline/baseline.md) for detailed analysis, confusion matrices, qualitative examples, and deployment recommendations.

## References

1. Jigsaw Toxic Comment Classification Challenge (Kaggle)
2. Blackmoon. *Russian Language Toxic Comments Dataset*
3. Semiletov, A. *Toxic Russian Comments Dataset*
4. Abusaqer, M. *Combined Hate Speech Dataset*
5. Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT, 2019.
6. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
