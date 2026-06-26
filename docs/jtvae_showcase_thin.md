# JT-VAE Showcase (Thin)

This profile is designed for live demos on a laptop GPU (including mid-tier NVIDIA mobile GPUs).

## What is included

- Lightweight dataset builder: `scripts/build_jtvae_showcase_dataset.py`
- Thin training config: `configs/gen_conf_showcase_thin.yaml`
- Small pretrained smoke-test checkpoint:
  - `models/generator_showcase/jtvae_best.pt`
  - `models/generator_showcase/fragment_vocab.json`
  - `models/generator_showcase/condition_stats.json`
- One-click launch scripts:
  - Windows: `scripts/run_jtvae_showcase_thin.ps1`
  - Linux/macOS: `scripts/run_jtvae_showcase_thin.sh`

## Download on another device

```bash
git clone https://github.com/Poseidon-lol/JuFo.git
cd JuFo
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux with NVIDIA CUDA:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Instant smoke test from included checkpoint

This does not train; it verifies that loading and JT-VAE sampling work.

```bash
python scripts/probe_generator.py --ckpt models/generator_showcase/jtvae_best.pt --vocab models/generator_showcase/fragment_vocab.json --n-samples 8 --device cuda:0 --max-tree-nodes 8
```

Use `--device cpu` if the demo device has no CUDA GPU.

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

