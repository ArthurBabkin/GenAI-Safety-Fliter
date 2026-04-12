"""
Robustness evaluation runner.

Evaluates all 3 models on clean, obfuscated, and deobfuscated test data.
Saves results to data/ subdirectory for notebook visualization.

Usage (from project root):
    python -m model.experiments.robustness.run_evaluation
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

from model.models import LogRegModel, TransformerClassifier, LoRATransformerClassifier
from model.obfuscation import obfuscate_dataset, deobfuscate_dataset
from model.utils import seed_everything

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    'logreg': (LogRegModel, ROOT / "data" / "models" / "logreg"),
    'transformer': (TransformerClassifier, ROOT / "data" / "models" / "transformer"),
    'transformer_lora': (LoRATransformerClassifier, ROOT / "data" / "models" / "transformer_lora"),
}

SEED = 42


def compute_metrics(y_true, y_proba, threshold):
    """Compute quality metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'threshold': threshold,
    }


def main():
    seed_everything(SEED)

    # Load test data from logreg checkpoint (same test set for all models)
    # NOTE: pickle is used here because the existing project convention stores
    # data splits as pickle files (data_splits.pkl). This is trusted local data.
    splits_path = ROOT / "data" / "models" / "logreg" / "data_splits.pkl"
    print(f"Loading test data from {splits_path}...")
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    X_test = splits['X_test']
    y_test = splits['y_test']
    X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)
    print(f"Test set: {len(X_test_list)} samples, {y_test.mean():.2%} toxic")

    # Generate obfuscated test set
    print("\nGenerating obfuscated test set...")
    X_obfuscated = obfuscate_dataset(X_test_list, seed=SEED)
    print(f"Obfuscated {len(X_obfuscated)} samples")

    # Show examples
    print("\n--- Obfuscation examples ---")
    for i in [0, 10, 50, 100, 500]:
        if i < len(X_test_list):
            orig = X_test_list[i][:80]
            obf = X_obfuscated[i][:80]
            if orig != obf:
                print(f"  [{i}] {orig}")
                print(f"    -> {obf}")
                print()

    # Generate deobfuscated version
    print("Applying deobfuscation defense...")
    X_deobfuscated = deobfuscate_dataset(X_obfuscated)
    print(f"Deobfuscated {len(X_deobfuscated)} samples")

    # Show deobfuscation examples
    print("\n--- Deobfuscation examples ---")
    for i in [0, 10, 50, 100, 500]:
        if i < len(X_test_list):
            obf = X_obfuscated[i][:80]
            deobf = X_deobfuscated[i][:80]
            if obf != deobf:
                print(f"  [{i}] {obf}")
                print(f"    -> {deobf}")
                print()

    all_results = {}

    for model_name, (model_cls, model_dir) in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 60}")

        # Load model
        model = model_cls(model_dir=str(model_dir))

        # Load baseline evaluation results for tuned threshold
        eval_path = model_dir / "evaluation_results.json"
        with open(eval_path) as f:
            baseline_results = json.load(f)
        threshold = baseline_results['best_threshold']
        print(f"Using tuned threshold: {threshold:.2f}")

        # Baseline metrics (from saved results)
        baseline_metrics = baseline_results['quality_tuned'].copy()
        baseline_metrics['confusion_matrix'] = baseline_results['confusion_matrix_tuned']
        baseline_metrics['threshold'] = threshold

        # Evaluate on obfuscated data
        print(f"Running inference on obfuscated data ({len(X_obfuscated)} samples)...")
        proba_obf = model.predict_proba(X_obfuscated)[:, 1]
        obf_metrics = compute_metrics(y_test, proba_obf, threshold)
        print(f"  F1: {obf_metrics['f1_score']:.4f} (baseline: {baseline_metrics['f1_score']:.4f}, "
              f"delta: {obf_metrics['f1_score'] - baseline_metrics['f1_score']:+.4f})")

        # Evaluate on deobfuscated data
        print(f"Running inference on deobfuscated data ({len(X_deobfuscated)} samples)...")
        proba_deobf = model.predict_proba(X_deobfuscated)[:, 1]
        deobf_metrics = compute_metrics(y_test, proba_deobf, threshold)
        print(f"  F1: {deobf_metrics['f1_score']:.4f} (baseline: {baseline_metrics['f1_score']:.4f}, "
              f"delta: {deobf_metrics['f1_score'] - baseline_metrics['f1_score']:+.4f})")

        all_results[model_name] = {
            'baseline': baseline_metrics,
            'obfuscated': obf_metrics,
            'deobfuscated': deobf_metrics,
        }

    # Save results
    results_path = OUTPUT_DIR / "robustness_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Condition':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10}")
    print("-" * 80)
    for model_name in MODELS:
        for condition in ['baseline', 'obfuscated', 'deobfuscated']:
            m = all_results[model_name][condition]
            print(f"{model_name:<20} {condition:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1_score']:>10.4f} {m['pr_auc']:>10.4f}")
        print()


if __name__ == '__main__':
    main()
