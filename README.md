# GenAI Safety Filter

Fast and resource-efficient safety filters for LLM outputs.

**Team:** Arthur Babkin, Alexander Malyy | **Course:** Generative AI, Spring 2026

## Overview

Systematic comparison of classical and neural safety filters for toxic text detection. Three models evaluated on 460K multilingual samples (56% Russian, 44% English):

| Model | F1 (tuned) | Latency | Throughput |
|-------|-----------|---------|------------|
| TF-IDF + LogReg | 0.76 | 0.009 ms | 109K/s |
| DistilBERT | **0.93** | 3.58 ms | 287/s |
| DistilBERT + LoRA | 0.79 | 3.27 ms | 316/s |

See [midterm report](docs/midterm/midterm.md) for full analysis, ablation study, and deployment recommendations.

## Setup

```bash
git clone https://github.com/ArthurBabkin/GenAI-Safety-Fliter.git
cd GenAI-Safety-Fliter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m model.train --model logreg --data data/train_dataset_clean.csv --output data/models/logreg
python -m model.train --model transformer --data data/train_dataset_clean.csv --output data/models/transformer
python -m model.train --model transformer_lora --data data/train_dataset_clean.csv --output data/models/transformer_lora --train-subset 100000
```

### Evaluation

```bash
python -m model.evaluate --model logreg --model-dir data/models/logreg
```

### Inference

```python
from model import LogRegModel, TransformerClassifier, LoRATransformerClassifier

model = TransformerClassifier(model_dir="data/models/transformer")
predictions = model.predict(["You are stupid", "Have a nice day"])  # [1, 0]
probabilities = model.predict_proba(["some text"])  # [[p_safe, p_toxic]]
```

### Ablations

Ablation notebooks live under `model/experiments/`. Each loads the global model checkpoint, derives modified training data, and compares against baseline. See [ablation rules](model/experiments/ablation_rules.md) for the full protocol.

To promote an ablation's weights to the global checkpoint:
```bash
python -m model.promote --from model/experiments/logreg/class_imbalance/data/balanced --to data/models/logreg
```

## Project Structure

```
GenAI-Safety-Fliter/
├── data/
│   ├── train_dataset_clean.csv          # Cleaned + deduplicated dataset (460K)
│   ├── original/                        # Raw source datasets
│   ├── plots/                           # Data visualization plots
│   └── models/                          # Global model checkpoints (git-lfs)
│       ├── logreg/                      #   weights + data_splits.pkl + evaluation_results.json
│       ├── transformer/
│       └── transformer_lora/
├── model/
│   ├── __init__.py
│   ├── models.py                        # BaseModel, LogRegModel, TransformerClassifier, LoRATransformerClassifier
│   ├── metrics.py                       # MetricsCalculator (quality, latency, throughput, memory)
│   ├── utils.py                         # seed_everything
│   ├── train.py                         # CLI: python -m model.train
│   ├── evaluate.py                      # CLI: python -m model.evaluate
│   ├── promote.py                       # CLI: python -m model.promote
│   └── experiments/
│       ├── ablation_rules.md            # Protocol for creating ablations
│       ├── logreg/
│       │   └── class_imbalance/
│       │       └── class_imbalance.ipynb
│       ├── transformer/
│       │   └── class_imbalance/
│       │       └── class_imbalance.ipynb
│       └── transformer_lora/
│           └── class_imbalance/
│               └── class_imbalance.ipynb
├── docs/
│   ├── proposal/                        # Project proposal
│   ├── baseline/                        # Baseline report
│   ├── midterm/                         # Midterm report
│   └── images/                          # Report plots
├── preprocess.ipynb                     # Data cleaning + LSH dedup
├── requirements.txt
└── README.md
```

### Key conventions

- **Global checkpoints** (`data/models/`) are trained via CLI and committed to git (weights via git-lfs). Notebooks never train global models.
- **Ablation outputs** (`model/experiments/**/data/`) are gitignored. Notebooks check for cached results and skip training if present.
- **Data splits** (`data_splits.pkl`) contain train/val/test arrays. Ablations derive modified training data from the global model's splits, keeping val/test fixed.
- All seeds fixed at 42.

## Reports

- [Project Proposal](docs/proposal/proposal.md)
- [Baseline Report](docs/baseline/baseline.md)
- [Midterm Report](docs/midterm/midterm.md)

## References

1. Jigsaw Toxic Comment Classification Challenge (Kaggle)
2. Blackmoon. *Russian Language Toxic Comments Dataset*
3. Semiletov, A. *Toxic Russian Comments Dataset*
4. Abusaqer, M. *Combined Hate Speech Dataset*
5. Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT, 2019.
6. Sanh, V., et al. *DistilBERT, a distilled version of BERT.* NeurIPS Workshop, 2019.
7. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022.
