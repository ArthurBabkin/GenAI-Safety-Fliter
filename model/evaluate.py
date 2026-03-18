"""
CLI evaluation script for safety filter models.

Usage:
    python -m model.evaluate --model logreg --model-dir /tmp/test_logreg
    python -m model.evaluate --model transformer --model-dir /tmp/test_transformer
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

from .models import LogRegModel, TransformerClassifier, LoRATransformerClassifier


MODEL_CLASSES = {
    'logreg': LogRegModel,
    'transformer': TransformerClassifier,
    'transformer_lora': LoRATransformerClassifier,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained safety filter model")
    parser.add_argument('--model', required=True, choices=MODEL_CLASSES.keys(),
                        help="Model type")
    parser.add_argument('--model-dir', required=True, type=str,
                        help="Directory containing saved model and test_data.pkl")
    parser.add_argument('--n-latency-runs', type=int, default=50,
                        help="Number of latency measurement runs (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    # Load model
    print(f"Loading {args.model} model from {model_dir}...")
    model = MODEL_CLASSES[args.model](model_dir=str(model_dir))

    # Load test data
    with open(model_dir / 'test_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']

    X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)
    X_val_list = X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val)

    # Run comprehensive evaluation (default threshold)
    print(f"Evaluating on {len(X_test_list)} test samples...")
    results = model.get_metrics(X_test_list, y_test, n_latency_runs=args.n_latency_runs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (default threshold=0.50)")
    print("=" * 60)
    print("\nQuality Metrics:")
    for metric, value in results['quality'].items():
        print(f"  {metric:15s}: {value:.4f}")

    cm_array = np.array(results['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm_array[0, 0]:5d}, FP={cm_array[0, 1]:5d}]")
    print(f"   [FN={cm_array[1, 0]:5d}, TP={cm_array[1, 1]:5d}]]")

    # Threshold tuning on validation set
    print("\nTuning threshold on validation set...")
    val_proba = model.predict_proba(X_val_list)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    f1_scores = np.array([
        f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds
    ])
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    # Tuned metrics on test set
    test_proba = model.predict_proba(X_test_list)[:, 1]
    y_pred_tuned = (test_proba >= best_threshold).astype(int)
    tuned_metrics = {
        'precision': float(precision_score(y_test, y_pred_tuned)),
        'recall': float(recall_score(y_test, y_pred_tuned)),
        'f1_score': float(f1_score(y_test, y_pred_tuned)),
        'pr_auc': float(average_precision_score(y_test, test_proba)),
    }
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)

    results['quality_tuned'] = tuned_metrics
    results['best_threshold'] = best_threshold
    results['confusion_matrix_tuned'] = cm_tuned.tolist()

    print(f"\n{'=' * 60}")
    print(f"TUNED THRESHOLD = {best_threshold:.2f}")
    print(f"{'=' * 60}")
    for metric, value in tuned_metrics.items():
        print(f"  {metric:15s}: {value:.4f}")

    print(f"\nConfusion Matrix (tuned):")
    print(f"  [[TN={cm_tuned[0, 0]:5d}, FP={cm_tuned[0, 1]:5d}]")
    print(f"   [FN={cm_tuned[1, 0]:5d}, TP={cm_tuned[1, 1]:5d}]]")

    print(f"\nLatency Metrics:")
    for metric, value in results['latency'].items():
        print(f"  {metric:20s}: {value:.4f}")
    print(f"\nThroughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")

    # Save results
    results_path = model_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
