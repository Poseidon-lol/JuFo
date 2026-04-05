#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/processed/opv_optical_shortlist_10k.csv}"
OUTPUT="${2:-data/processed/opv_optical_labeled_10k.csv}"
QC_CONFIG="${3:-configs/qc_pipeline_optical.yaml}"
WORKERS="${4:-10}"

python scripts/label_monomer_dataset.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --qc-config "$QC_CONFIG" \
  --workers "$WORKERS" \
  --resume
