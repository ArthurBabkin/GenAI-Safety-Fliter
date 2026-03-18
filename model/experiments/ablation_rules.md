# Ablation Rules

## Prerequisites

Each model type has a **global checkpoint** at `data/models/{logreg,transformer,transformer_lora}/`. This must be trained via CLI before running any ablation:

```bash
python -m model.train --model logreg --data data/train_dataset_clean.csv --output data/models/logreg
python -m model.train --model transformer --data data/train_dataset_clean.csv --output data/models/transformer
python -m model.train --model transformer_lora --data data/train_dataset_clean.csv --output data/models/transformer_lora --train-subset 100000
```

The global checkpoint contains:
- Model weights (committed to git)
- `data_splits.pkl` — canonical train/val/test splits (committed to git)
- `evaluation_results.json` — baseline metrics (committed to git)

## Creating a new ablation

1. Create a directory: `model/experiments/<model_type>/<ablation_name>/`
2. Create `class_imbalance.ipynb` (or similar) following this structure:

### Notebook structure (11 cells)

| # | Type | Content |
|---|------|---------|
| 1 | md | Title + description |
| 2 | code | Setup: imports, `ROOT`, `GLOBAL_MODEL_DIR`, `ABLATION_DIR`, assert global model exists |
| 3 | md | `## 1. Baseline (global model)` |
| 4 | code | Load/evaluate global model, print baseline metrics |
| 5 | md | `## 2. Train & Evaluate (ablation)` |
| 6 | code | Load global `data_splits.pkl`, modify **train portion only**, save modified splits, call `train.py --data-splits`, call `evaluate.py` |
| 7 | md | `## 3. Load Results` |
| 8 | code | Load ablation `evaluation_results.json` |
| 9 | md | `## 4. Comparison` |
| 10 | code | Grouped bar chart + confusion matrices + summary table |
| 11 | md | Interpretation |

### Key rules

- **Never train the global model in a notebook.** Notebooks only read from the global checkpoint.
- **Always derive from global `data_splits.pkl`.** Modify only `X_train`/`y_train`. Keep `X_val`, `y_val`, `X_test`, `y_test` unchanged. This ensures all ablations are evaluated on the same test set.
- **Use `--data-splits` flag** when calling `train.py` to pass pre-made splits.
- **Save ablation outputs to `data/` inside the notebook directory.** This path is gitignored.
- **Cache results.** Check if `evaluation_results.json` exists before training. Delete the `data/` subdirectory to force a retrain.

## Promoting an ablation

If an ablation improves on the baseline, promote it to the global checkpoint:

```bash
python -m model.promote \
    --from model/experiments/logreg/class_imbalance/data/balanced \
    --to data/models/logreg
```

This copies **all files** (weights + `data_splits.pkl`) to the global directory. Future ablations will then build on the promoted model's training data as the new baseline.

After promoting, re-run `evaluate.py` on the global checkpoint to regenerate `evaluation_results.json`, then commit.

## Stacking ablations

Since ablations derive from the global checkpoint, promoting one ablation changes the baseline for all future ablations. This allows testing multiple improvements on top of each other:

1. Train global model (original data)
2. Run class imbalance ablation -> balanced training improves metrics
3. Promote balanced model to global
4. Run next ablation (e.g., augmentation) -> now tests augmentation on top of balanced training
