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
   - Multilingual, multilabel toxic comment dataset
   - Converted to binary classification

2. **Russian Language Toxic Comments** ([Kaggle](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments))
   - Binary labels, Russian-language focus

3. **Toxic Russian Comments** ([Kaggle](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments))
   - Multilabel Russian comments, converted to binary

4. **Combined Hate Speech Dataset** ([Kaggle](https://www.kaggle.com/datasets/mahmoudabusaqer/combined-hate-speech-dataset))
   - English binary classification dataset

### Text Length Distribution

- **Mean:** 190.35 characters
- **Median:** 77 characters
- **Range:** 1 - 20,030 characters

### Language Distribution

| Language | Samples | Percentage |
|----------|---------|------------|
| Russian  | 262,165 | 55.7% |
| English  | 208,148 | 44.3% |
| Other    | 9       | 0.0% |

### Data Visualization

![Data Distribution](data/plots/data_distribution.png)

*Figure: Comprehensive visualization of dataset distribution by source, label balance, text length, and language composition.*

## Approach

### Baseline Model
- **TF-IDF + Logistic Regression**
- Very low inference latency
- Minimal memory footprint
- Strong classical baseline for text classification

### Comparison Model
- **Lightweight Transformer-based classifier**
- Evaluated under identical conditions
- Measures inference latency, memory usage, and detection quality

### Evaluation Metrics

**Quality:**
- Precision, Recall, F1-score
- Precision-Recall AUC (PR-AUC)
- Confusion matrix

**Performance:**
- Average inference latency per sample (CPU/GPU)
- Throughput (samples per second)
- Peak memory usage during inference
- Model size on disk

**Testing:**
- Clean test data
- Obfuscated test data (stress testing)

## Project Structure

```
GenAI-Safety-Fliter/
├── data/                           # Dataset files (not tracked in git)
│   ├── train_dataset.csv           # Combined dataset (470K samples)
│   └── models/                     # Trained models
│       ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│       ├── logreg_model.pkl        # Logistic Regression model
│       └── test_data.pkl           # Test split for evaluation
├── docs/
│   ├── images/                     # Documentation images
│   │   └── baseline_metrics.png   # Baseline model metrics
│   └── proposal/
│       ├── proposal.md             # Project proposal
│       └── proposal.pdf            # Proposal PDF version
├── model/
│   ├── __init__.py                 # Package initialization
│   ├── models.py                   # Model implementations
│   ├── metrics.py                  # Metrics calculation
│   └── experiments/
│       └── tf_idf.ipynb           # TF-IDF baseline experiment
├── preprocess.ipynb                # Data preprocessing notebook
├── requirements.txt                # Python dependencies
├── venv/                           # Virtual environment (not tracked)
├── .gitignore
└── README.md                       # This file
```

## Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArthurBabkin/GenAI-Safety-Fliter.git
cd GenAI-Safety-Fliter
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the preprocessing notebook:
```bash
jupyter notebook preprocess.ipynb
```

## Usage

### Training Models

Train the baseline TF-IDF + Logistic Regression model:

```bash
# Start Jupyter
jupyter notebook model/experiments/tf_idf.ipynb
```

The notebook will:
1. Load and split the dataset (80/20 train/test)
2. Train TF-IDF vectorizer with 10K features
3. Train Logistic Regression classifier
4. Save models to `data/models/`

### Using Trained Models

Load and use a trained model:

```python
from model import LogRegModel

# Load pre-trained model
model = LogRegModel(
    vectorizer_path="data/models/tfidf_vectorizer.pkl",
    model_path="data/models/logreg_model.pkl"
)

# Predict
texts = ["You are stupid", "Have a nice day"]
predictions = model.predict(texts)  # [1, 0]
probabilities = model.predict_proba(texts)  # [[0.001, 0.999], [0.915, 0.085]]
```

### Evaluating Models

Get comprehensive metrics using the built-in evaluation:

```python
# Evaluate on test data
metrics = model.get_metrics(
    X_test=test_texts,
    y_test=test_labels,
    n_latency_runs=100
)

# Returns:
# {
#   'quality': {'precision': 0.93, 'recall': 0.63, 'f1_score': 0.75, 'pr_auc': 0.84},
#   'confusion_matrix': [[786, 10], [76, 128]],
#   'latency': {'latency_mean_ms': 0.017, 'latency_std_ms': 0.001, ...},
#   'throughput_samples_per_sec': 60314.46,
#   'peak_memory_mb': 0.0
# }
```

### Baseline Results

Our TF-IDF + Logistic Regression baseline achieves:

- **F1-Score:** 0.7485
- **Precision:** 0.9275
- **Recall:** 0.6275
- **PR-AUC:** 0.8432
- **Throughput:** 60,314 samples/sec
- **Latency:** 0.017 ms/sample

![Baseline Metrics](docs/images/baseline_metrics.png)

*Figure: Performance metrics for TF-IDF + Logistic Regression baseline model showing quality metrics, confusion matrix, and inference latency.*

## Timeline

- **Week 2-3:** Dataset selection, preprocessing, proposal submission
- **Week 4-5:** Implement TF-IDF + Logistic Regression baseline and speed evaluation
- **Week 6:** Midterm checkpoint (baseline results and initial analysis)
- **Week 7-8:** Implement Transformer classifier and capacity comparison
- **Week 9:** Robustness and deobfuscation experiments
- **Week 10-12:** Final analysis, report writing, reproducibility checks, demo preparation

## License

This project uses publicly available datasets with research-friendly licenses. All dataset licenses and usage constraints are documented in the final report.

## References

1. Jigsaw Toxic Comment Classification Challenge (Kaggle)
2. Blackmoon. *Russian Language Toxic Comments Dataset*
3. Semiletov, A. *Toxic Russian Comments Dataset*
4. Abusaqer, M. *Combined Hate Speech Dataset*
5. Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT, 2019.
6. Liu, Y. et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692, 2019.
7. Schmidt, A., & Wiegand, M. *A Survey on Hate Speech Detection using Natural Language Processing.* 2017.
8. Fortuna, P., & Nunes, S. *A Survey on Automatic Detection of Hate Speech in Text.* ACM Computing Surveys, 2018.
