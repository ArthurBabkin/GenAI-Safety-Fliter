"""
Promote an ablation model to the global checkpoint.

Copies all files (weights + data_splits.pkl) from the ablation directory
to the global model directory, overwriting existing files.

Usage:
    python -m model.promote --from model/experiments/logreg/class_imbalance/data/balanced --to data/models/logreg
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote an ablation model to the global checkpoint"
    )
    parser.add_argument('--from', dest='src', required=True, type=str,
                        help="Source ablation directory")
    parser.add_argument('--to', dest='dst', required=True, type=str,
                        help="Target global model directory")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    # List what will be copied
    files = [f for f in src.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in {src}")

    print(f"Promoting {src} -> {dst}")
    print(f"Files to copy:")
    for f in sorted(files):
        print(f"  {f.name}")

    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)
        print(f"  Copied {f.name}")

    print("Done!")


if __name__ == '__main__':
    main()
