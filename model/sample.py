"""
CLI inference script for safety filter models.

Usage:
    python -m model.sample --model logreg --model-dir data/models/logreg "You are stupid"
    python -m model.sample --model transformer --model-dir data/models/transformer "Have a nice day"
    python -m model.sample --model transformer_lora --model-dir data/models/transformer_lora "some text"
"""

import argparse

from .models import LogRegModel, TransformerClassifier, LoRATransformerClassifier


MODEL_CLASSES = {
    'logreg': LogRegModel,
    'transformer': TransformerClassifier,
    'transformer_lora': LoRATransformerClassifier,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on text samples")
    parser.add_argument('texts', nargs='+', type=str,
                        help="One or more text samples to classify")
    parser.add_argument('--model', required=True, choices=MODEL_CLASSES.keys(),
                        help="Model type")
    parser.add_argument('--model-dir', required=True, type=str,
                        help="Directory containing saved model")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model} from {args.model_dir}...")
    model = MODEL_CLASSES[args.model](model_dir=args.model_dir)

    proba = model.predict_proba(args.texts)
    preds = model.predict(args.texts)

    for text, pred, p in zip(args.texts, preds, proba):
        label = "TOXIC" if pred == 1 else "SAFE"
        print(f"\n  [{label}] (safe={p[0]:.4f}, toxic={p[1]:.4f})  {text!r}")


if __name__ == '__main__':
    main()
