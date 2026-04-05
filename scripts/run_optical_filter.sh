#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/opv_optical_labeled_10k.csv}"
OUTPUT="${2:-data/processed/opv_optical_labeled_10k_filtered.csv}"
FILTER_CONFIG="${3:-configs/filter_rules_optical.yaml}"

python scripts/filter_qc_results.py \
  --qc "$INPUT" \
  --filter "$FILTER_CONFIG" \
  --output "$OUTPUT"
