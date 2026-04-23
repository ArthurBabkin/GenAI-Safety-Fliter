#!/usr/bin/env bash
# Verify Git-LFS artifacts are hydrated (not pointer stubs).
# Exit 0 if all good, 1 if any file missing or still a stub.

set -e
cd "$(dirname "$0")/.."

REQUIRED=(
  "data/models/logreg/logreg_model.pkl"
  "data/models/logreg/tfidf_vectorizer.pkl"
  "data/models/logreg/data_splits.pkl"
  "data/models/transformer/model.safetensors"
  "data/models/transformer/data_splits.pkl"
  "data/models/transformer_lora/model.safetensors"
  "data/models/transformer_lora/data_splits.pkl"
  "data/train_dataset_clean.csv"
)

# A Git LFS pointer stub is ~130 bytes of text. Real artifacts are >>10 KB.
MIN_SIZE=10000
MISSING=0

for f in "${REQUIRED[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    MISSING=1
    continue
  fi
  size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "$size" -lt "$MIN_SIZE" ]; then
    echo "LFS STUB: $f (size=$size bytes)"
    MISSING=1
    continue
  fi
  printf "OK %10d  %s\n" "$size" "$f"
done

if [ "$MISSING" -ne 0 ]; then
  echo ""
  echo "One or more artifacts missing or still a Git LFS pointer stub."
  echo "Run:   git lfs install && git lfs pull"
  exit 1
fi

echo ""
echo "All LFS artifacts hydrated."
