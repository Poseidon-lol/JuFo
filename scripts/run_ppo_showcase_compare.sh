#!/usr/bin/env bash
set -euo pipefail

ITERATIONS="${ITERATIONS:-50}"
SEED_ROWS="${SEED_ROWS:-300}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
SURROGATE_DIR="${SURROGATE_DIR:-models/surrogate_3d_full}"
GENERATOR_CKPT="${GENERATOR_CKPT:-models/generator/jtvae_epoch_120.pt}"

[[ -e "${SURROGATE_DIR}" ]] || { echo "Missing surrogate dir: ${SURROGATE_DIR}" >&2; exit 1; }
[[ -f "${GENERATOR_CKPT}" ]] || { echo "Missing generator checkpoint: ${GENERATOR_CKPT}" >&2; exit 1; }

echo "[Preflight] Checking Python dependencies (torch, pandas)..."
python - <<'PY'
import torch, pandas  # noqa: F401
PY

echo "[1/4] Build lightweight active-loop seed..."
python scripts/build_active_loop_showcase_seed.py --rows "${SEED_ROWS}" --seed "${SEED}"

echo "[2/4] Baseline run (ohne RL)..."
python -m src.main active-loop \
  --config configs/active_learn_showcase_thin_baseline.yaml \
  --iterations "${ITERATIONS}" \
  --surrogate-dir "${SURROGATE_DIR}" \
  --generator-ckpt "${GENERATOR_CKPT}" \
  --use-pseudo-dft \
  --device "${DEVICE}" \
  --surrogate-device "${DEVICE}" \
  --generator-device "${DEVICE}"

echo "[3/4] PPO run..."
python -m src.main active-loop \
  --config configs/active_learn_showcase_thin_ppo.yaml \
  --iterations "${ITERATIONS}" \
  --surrogate-dir "${SURROGATE_DIR}" \
  --generator-ckpt "${GENERATOR_CKPT}" \
  --use-pseudo-dft \
  --device "${DEVICE}" \
  --surrogate-device "${DEVICE}" \
  --generator-device "${DEVICE}"
echo "PPO live dashboard: experiments/showcase_ppo/ppo/active_loop_live_dashboard.html"

echo "[4/4] Build comparison figure..."
python scripts/plot_ppo_vs_baseline_hitrate.py \
  --baseline experiments/showcase_ppo/baseline/active_learning_history.csv \
  --ppo experiments/showcase_ppo/ppo/active_learning_history.csv \
  --output experiments/showcase_ppo/ppo_vs_baseline_hitrate.png \
  --score-threshold -2.0

echo "Done."
if [[ -f experiments/showcase_ppo/ppo_vs_baseline_hitrate.png ]]; then
  echo "Figure: experiments/showcase_ppo/ppo_vs_baseline_hitrate.png"
elif [[ -f experiments/showcase_ppo/ppo_vs_baseline_hitrate.svg ]]; then
  echo "Figure: experiments/showcase_ppo/ppo_vs_baseline_hitrate.svg"
else
  echo "Warning: no figure file found after plotting." >&2
fi
