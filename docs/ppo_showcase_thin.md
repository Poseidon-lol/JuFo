# PPO Showcase (Thin, Laptop-Friendly)

This setup runs a lightweight `PPO vs ohne RL` comparison for the active loop and generates a hit-rate figure.

## What it does

1. Builds a small labelled seed set (`300` rows by default).
2. Runs baseline active-loop (`RL disabled`) for N rounds.
3. Runs active-loop with `PPO enabled` for N rounds.
4. Plots `Hit-Rate (%)` per active-learning round.

Both runs use pseudo-DFT (`--use-pseudo-dft`) so it is fast enough for laptop demos.
PPO run now also exposes a live dashboard (auto-refresh + local browser view).

## Run (Windows)

```powershell
./scripts/run_ppo_showcase_compare.ps1 -Iterations 50 -Device auto
```

Hinweis:
- Falls `models/surrogate_3d_full` oder `models/generator/*` nur Git-LFS-Pointer sind, bootstrapt das Script automatisch ein kleines lokales SchNet- und JT-VAE-Modell.
- Diese Bootstrap-Modelle landen in:
  - `models/surrogate_3d_showcase_bootstrap`
  - `models/generator_showcase_bootstrap`

CPU-only quick test (skip plotting dependencies):

```powershell
./scripts/run_ppo_showcase_compare.ps1 -Iterations 50 -Device cpu -SkipPlot
```

## Run (Linux/macOS)

```bash
ITERATIONS=50 DEVICE=auto bash scripts/run_ppo_showcase_compare.sh
```

## Output

- Baseline history: `experiments/showcase_ppo/baseline/active_learning_history.csv`
- PPO history: `experiments/showcase_ppo/ppo/active_learning_history.csv`
- PPO live dashboard: `experiments/showcase_ppo/ppo/active_loop_live_dashboard.html`
- Figure: `experiments/showcase_ppo/ppo_vs_baseline_hitrate.png` (or `.svg` fallback)
- Summary table: `experiments/showcase_ppo/ppo_vs_baseline_hitrate.summary.csv`

## Required Python packages

- `torch`
- `pandas`
- `matplotlib`

## Manual plotting

```powershell
py -3 scripts/plot_ppo_vs_baseline_hitrate.py `
  --baseline experiments/showcase_ppo/baseline/active_learning_history.csv `
  --ppo experiments/showcase_ppo/ppo/active_learning_history.csv `
  --output experiments/showcase_ppo/ppo_vs_baseline_hitrate.png `
  --score-threshold -2.0
```
