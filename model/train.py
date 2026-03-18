"""
CLI training script for safety filter models.

Usage:
    python -m model.train --model logreg --data data/train_dataset_clean.csv --output data/models/logreg
    python -m model.train --model transformer --data data/train_dataset_clean.csv --output data/models/transformer
    python -m model.train --model transformer_lora --data data/train_dataset_clean.csv --output data/models/transformer_lora --train-subset 100000
    python -m model.train --model logreg --data-splits data/models/logreg/data_splits.pkl --output /tmp/ablation
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .models import LogRegModel, TransformerClassifier, LoRATransformerClassifier
from .utils import seed_everything


MODEL_CLASSES = {
    'logreg': LogRegModel,
    'transformer': TransformerClassifier,
    'transformer_lora': LoRATransformerClassifier,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a safety filter model")
    parser.add_argument('--model', required=True, choices=MODEL_CLASSES.keys(),
                        help="Model type to train")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data', type=str,
                       help="Path to training CSV (must have 'text' and 'y' columns)")
    group.add_argument('--data-splits', type=str,
                       help="Path to data_splits.pkl with pre-made train/val/test splits")
    parser.add_argument('--output', required=True, type=str,
                        help="Directory to save trained model and data splits")
    parser.add_argument('--train-subset', type=int, default=None,
                        help="Subsample training data to N rows before splitting (only with --data)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.data_splits:
        # Load pre-made splits
        print(f"Loading data splits from {args.data_splits}...")
        with open(args.data_splits, 'rb') as f:
            splits = pickle.load(f)
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_val = splits['X_val']
        y_val = splits['y_val']
        X_test = splits['X_test']
        y_test = splits['y_test']
    else:
        # Load CSV and split
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        X = df['text'].values
        y = df['y'].values
        print(f"Total samples: {len(df)}, toxicity rate: {y.mean():.2%}")

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )

        if args.train_subset and args.train_subset < len(X_train_full):
            X_train_full, _, y_train_full, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=args.train_subset,
                random_state=args.seed,
                stratify=y_train_full
            )
            print(f"Subsampled training data to {len(X_train_full)} rows")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1,
            random_state=args.seed, stratify=y_train_full
        )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train toxicity rate: {y_train.mean():.2%}")

    # Instantiate and train
    model = MODEL_CLASSES[args.model]()
    X_train_list = X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train)
    X_val_list = X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val)

    if args.model == 'logreg':
        model.fit(X_train_list, y_train)
    else:
        model.fit(X_train_list, y_train, X_val=X_val_list, y_val=y_val)

    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    # Save data splits alongside model
    with open(output_dir / 'data_splits.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }, f)
    print(f"Data splits saved to {output_dir / 'data_splits.pkl'}")
    print("Done!")


if __name__ == '__main__':
    main()
