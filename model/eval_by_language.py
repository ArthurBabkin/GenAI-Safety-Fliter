"""
Per-language (RU/EN) evaluation of trained safety-filter checkpoints.

Reconstructs the train/test split exactly (seed=42, stratify=y, test_size=0.2),
tags each test row as RU or EN via source column + Cyrillic heuristic, runs
the model once on the full test set, then computes quality metrics per bucket.

Uses tuned threshold (swept 0.10-0.90 on validation set, argmax F1) matching
the protocol in model/evaluate.py. Reports both default (0.5) and tuned
per-bucket metrics.

Usage:
    python -m model.eval_by_language --model logreg           --model-dir data/models/logreg
    python -m model.eval_by_language --model transformer      --model-dir data/models/transformer
    python -m model.eval_by_language --model transformer_lora --model-dir data/models/transformer_lora
"""

import argparse
import json
import pickle  # loading existing checkpoint splits (trusted repo artifacts)
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .metrics import MetricsCalculator
from .models import LogRegModel, TransformerClassifier, LoRATransformerClassifier


MODEL_CLASSES = {
    'logreg': LogRegModel,
    'transformer': TransformerClassifier,
    'transformer_lora': LoRATransformerClassifier,
}

PUBLISHED_F1 = {
    'logreg': 0.8035,
    'transformer': 0.9318,
    'transformer_lora': 0.8035,
}

RU_SOURCES = {'russian_toxic', 'toxic_russian'}
EN_SOURCES = {'jigsaw'}


def has_cyrillic(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return any(0x0400 <= ord(c) <= 0x04FF for c in s)


def label_language(source: str, text: str) -> str:
    if source in RU_SOURCES:
        return 'ru'
    if source in EN_SOURCES:
        return 'en'
    return 'ru' if has_cyrillic(text) else 'en'


def reconstruct_test_split(csv_path: Path, seed: int = 42, test_size: float = 0.2):
    df = pd.read_csv(csv_path)
    idx = np.arange(len(df))
    y = df['y'].values
    _, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    return df, test_idx


def parse_args():
    p = argparse.ArgumentParser(description="Per-language eval of safety-filter model")
    p.add_argument('--model', required=True, choices=MODEL_CLASSES.keys())
    p.add_argument('--model-dir', required=True, type=str)
    p.add_argument('--data-csv', default='data/train_dataset_clean.csv', type=str)
    p.add_argument('--seed', default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    csv_path = Path(args.data_csv)

    print(f"Loading CSV: {csv_path}")
    df, test_idx = reconstruct_test_split(csv_path, seed=args.seed)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    X_test_csv = test_df['text'].values
    y_test_csv = test_df['y'].values.astype(int)

    splits_path = model_dir / 'data_splits.pkl'
    if not splits_path.exists():
        splits_path = model_dir / 'test_data.pkl'
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)  # noqa: S301 - repo-local trusted artifact
    X_test_pkl = splits['X_test']
    y_test_pkl = np.asarray(splits['y_test']).astype(int)
    X_val_pkl = splits['X_val']
    y_val_pkl = np.asarray(splits['y_val']).astype(int)

    if len(X_test_csv) != len(X_test_pkl):
        raise RuntimeError(
            f"ABORT: length mismatch CSV={len(X_test_csv)} PKL={len(X_test_pkl)}"
        )

    X_test_pkl_arr = np.asarray(X_test_pkl)
    mismatches = np.where(X_test_csv != X_test_pkl_arr)[0]
    if len(mismatches) > 0:
        print(f"ABORT: {len(mismatches)} text rows disagree between CSV-split and pickle.")
        print(f"First 5 mismatch indices: {mismatches[:5]}")
        for i in mismatches[:3]:
            print(f"  idx {i}: CSV={str(X_test_csv[i])[:80]!r}  PKL={str(X_test_pkl_arr[i])[:80]!r}")
        raise RuntimeError("Split reproduction failed — language labels would be misaligned.")
    if not np.array_equal(y_test_csv, y_test_pkl):
        raise RuntimeError("ABORT: y_test disagrees between CSV-split and pickle.")
    print(f"Split reproducibility verified: {len(X_test_pkl)} samples match.")

    langs = np.array([
        label_language(src, txt) for src, txt in zip(test_df['source'].values, X_test_csv)
    ])
    n_ru = int((langs == 'ru').sum())
    n_en = int((langs == 'en').sum())
    print(f"Language distribution: RU={n_ru} ({n_ru/len(langs):.1%}), "
          f"EN={n_en} ({n_en/len(langs):.1%})")

    print(f"Loading {args.model} from {model_dir}...")
    model = MODEL_CLASSES[args.model](model_dir=str(model_dir))

    X_list = X_test_pkl_arr.tolist() if hasattr(X_test_pkl_arr, 'tolist') else list(X_test_pkl_arr)
    X_val_list = X_val_pkl.tolist() if hasattr(X_val_pkl, 'tolist') else list(X_val_pkl)
    y_true = y_test_pkl

    # Validation inference → threshold tuning
    print(f"Tuning threshold on {len(X_val_list)} validation samples...")
    t_val = time.time()
    val_proba_pos = model.predict_proba(X_val_list)[:, 1]
    val_runtime = time.time() - t_val
    thresholds = np.arange(0.10, 0.91, 0.01)
    f1_scores = np.array([
        f1_score(y_val_pkl, (val_proba_pos >= t).astype(int)) for t in thresholds
    ])
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    print(f"Val inference {val_runtime:.1f}s. Best threshold = {best_threshold:.2f} "
          f"(val F1 = {f1_scores[best_idx]:.4f})")

    # Test inference
    print(f"Predicting on {len(X_list)} test samples...")
    t0 = time.time()
    y_proba = model.predict_proba(X_list)
    runtime = time.time() - t0
    print(f"Test inference done in {runtime:.1f}s ({len(X_list)/runtime:.1f} samples/sec)")

    y_proba = np.asarray(y_proba)
    y_proba_pos = y_proba[:, 1]
    y_pred = (y_proba_pos >= 0.5).astype(int)
    y_pred_tuned = (y_proba_pos >= best_threshold).astype(int)

    def bucket_metrics(mask, y_pred_use):
        n = int(mask.sum())
        if n == 0:
            return {'n': 0, 'precision': None, 'recall': None, 'f1': None, 'pr_auc': None}
        q = MetricsCalculator.calculate_quality_metrics(
            y_true[mask], y_pred_use[mask], y_proba[mask]
        )
        return {
            'n': n,
            'precision': float(q['precision']),
            'recall': float(q['recall']),
            'f1': float(q['f1_score']),
            'pr_auc': float(q.get('pr_auc')) if q.get('pr_auc') is not None else None,
        }

    all_mask = np.ones(len(y_true), dtype=bool)
    ru_mask = (langs == 'ru')
    en_mask = (langs == 'en')

    def bucket_pair(mask):
        default_m = bucket_metrics(mask, y_pred)
        tuned_m = bucket_metrics(mask, y_pred_tuned)
        n = default_m['n']
        # strip n from nested dicts (kept at outer level)
        for d in (default_m, tuned_m):
            d.pop('n', None)
        return {'n': n, 'default': default_m, 'tuned': tuned_m}

    results = {
        'model': args.model,
        'best_threshold': best_threshold,
        'all': bucket_pair(all_mask),
        'ru': bucket_pair(ru_mask),
        'en': bucket_pair(en_mask),
        'language_source': 'hybrid_source_and_cyrillic',
        'val_inference_seconds': round(val_runtime, 2),
        'test_inference_seconds': round(runtime, 2),
    }

    published = PUBLISHED_F1.get(args.model)
    tuned_f1_all = results['all']['tuned']['f1']
    if published is not None:
        delta = abs(tuned_f1_all - published)
        flag = delta > 0.005
        results['sanity_check'] = {
            'published_f1': published,
            'observed_tuned_f1': tuned_f1_all,
            'delta': round(delta, 5),
            'flag_deviation': flag,
        }
        msg = "FLAGGED" if flag else "OK"
        print(f"Sanity tuned F1(all)={tuned_f1_all:.4f} vs published={published:.4f} "
              f"delta={delta:.4f} [{msg}]")

    out_json = model_dir / 'evaluation_by_language.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

    out_npz = model_dir / 'predictions_cache.npz'
    np.savez(
        out_npz,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_tuned=y_pred_tuned,
        y_proba_pos=y_proba_pos,
        best_threshold=np.array(best_threshold),
        lang=langs,
    )
    print(f"Wrote {out_npz}")

    print()
    print(f"TUNED threshold = {best_threshold:.2f}")
    print(f"{'bucket':<6} {'n':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'pr_auc':>8}")
    for k in ('all', 'ru', 'en'):
        r = results[k]
        t = r['tuned']
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else 'nan'
        print(f"{k:<6} {r['n']:>8} {fmt(t['precision']):>10} "
              f"{fmt(t['recall']):>8} {fmt(t['f1']):>8} {fmt(t['pr_auc']):>8}")


if __name__ == '__main__':
    main()
