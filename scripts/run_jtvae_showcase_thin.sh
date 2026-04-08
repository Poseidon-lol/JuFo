#!/usr/bin/env bash
set -euo pipefail

ROWS="${ROWS:-1200}"
DEVICE="${DEVICE:-cuda:0}"
AMP_FLAG="${AMP_FLAG:---amp}" # set AMP_FLAG=--no-amp to disable

echo "[1/2] Building lightweight showcase dataset..."
python scripts/build_jtvae_showcase_dataset.py --max-rows "${ROWS}"

echo "[2/2] Starting JT-VAE showcase training..."
python -m src.main train-generator \
  --config configs/gen_conf_showcase_thin.yaml \
  --device "${DEVICE}" \
  "${AMP_FLAG}"

echo "Done. Dashboard: experiments/showcase_thin/live_decode_dashboard.html"

