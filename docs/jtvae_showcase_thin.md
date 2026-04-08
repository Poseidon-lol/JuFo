# JT-VAE Showcase (Thin)

This profile is designed for live demos on a laptop GPU (including mid-tier NVIDIA mobile GPUs).

## What is included

- Lightweight dataset builder: `scripts/build_jtvae_showcase_dataset.py`
- Thin training config: `configs/gen_conf_showcase_thin.yaml`
- One-click launch scripts:
  - Windows: `scripts/run_jtvae_showcase_thin.ps1`
  - Linux/macOS: `scripts/run_jtvae_showcase_thin.sh`

## Quick start (Windows)

```powershell
./scripts/run_jtvae_showcase_thin.ps1 -Rows 1200 -Device cuda:0
```

If VRAM is tight, disable AMP fallback explicitly:

```powershell
./scripts/run_jtvae_showcase_thin.ps1 -Rows 900 -Device cuda:0 -NoAmp
```

## Quick start (Linux/macOS)

```bash
ROWS=1200 DEVICE=cuda:0 bash scripts/run_jtvae_showcase_thin.sh
```

## Dashboard output

`experiments/showcase_thin/live_decode_dashboard.html`

## Notes

- The thin profile keeps the model smaller (`z_dim=64`, `hidden_dim=192`, `encoder_layers=4`).
- It uses only 3 conditioning targets (`homo`, `lumo`, `gap`) for stability and speed.
- Dataset size is capped (`--max-rows`) so showcase startup/training stays responsive.

