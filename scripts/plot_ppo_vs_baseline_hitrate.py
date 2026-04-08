#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def _detect_round_col(df: pd.DataFrame) -> str:
    for col in ("iteration", "round", "iter", "active_round", "step"):
        if col in df.columns:
            return col
    raise KeyError("No round column found. Expected one of: iteration, round, iter, active_round, step.")


def _compute_hit_rate_by_round(df: pd.DataFrame, *, score_threshold: float) -> pd.DataFrame:
    round_col = _detect_round_col(df)
    work = df.copy()
    if "red_pass" in work.columns:
        hits = work["red_pass"].fillna(False).astype(bool)
    elif "red_score" in work.columns:
        hits = pd.to_numeric(work["red_score"], errors="coerce") >= float(score_threshold)
    else:
        raise KeyError("Need either 'red_pass' or 'red_score' column to compute hit rate.")

    work["_hit"] = hits.astype(float)
    grp = (
        work.groupby(round_col, as_index=False)["_hit"]
        .mean()
        .rename(columns={"_hit": "hit_rate"})
        .sort_values(round_col)
    )
    grp["hit_rate_pct"] = grp["hit_rate"] * 100.0
    return grp


def _plot_with_matplotlib(
    *,
    base_curve: pd.DataFrame,
    ppo_curve: pd.DataFrame,
    round_col: str,
    out_path: Path,
    title: str,
    last_round: int,
    base_last: float,
    ppo_last: float,
) -> Path:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is not available")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.plot(
        base_curve[round_col],
        base_curve["hit_rate_pct"],
        color="#64748b",
        marker="o",
        linewidth=2.2,
        label="ohne RL",
    )
    ax.plot(
        ppo_curve[round_col],
        ppo_curve["hit_rate_pct"],
        color="#0f766e",
        marker="o",
        linewidth=2.6,
        label="PPO",
    )
    ax.set_title(title, fontsize=22, fontweight="bold", pad=14)
    ax.set_xlabel("Active-Learning-Runde", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hit-Rate (%)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, max(100, float(np.nanmax([base_curve["hit_rate_pct"].max(), ppo_curve["hit_rate_pct"].max()])) + 5))
    ax.legend(loc="best")

    delta = ppo_last - base_last
    ax.text(
        0.99,
        0.02,
        f"Finale Differenz (Runde {last_round}): {delta:+.1f} %-Punkte",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_svg_fallback(
    *,
    base_curve: pd.DataFrame,
    ppo_curve: pd.DataFrame,
    round_col: str,
    out_path: Path,
    title: str,
    last_round: int,
    base_last: float,
    ppo_last: float,
) -> Path:
    svg_path = out_path if out_path.suffix.lower() == ".svg" else out_path.with_suffix(".svg")
    width, height = 1400, 860
    margin_l, margin_r, margin_t, margin_b = 120, 60, 100, 120
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_values = sorted(set(base_curve[round_col].tolist()) | set(ppo_curve[round_col].tolist()))
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    if x_max <= x_min:
        x_max = x_min + 1.0

    y_max_data = float(max(base_curve["hit_rate_pct"].max(), ppo_curve["hit_rate_pct"].max()))
    y_min = 0.0
    y_max = max(100.0, y_max_data + 5.0)

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    def polyline_points(df: pd.DataFrame) -> str:
        pts = [f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(df[round_col], df["hit_rate_pct"])]
        return " ".join(pts)

    y_ticks = np.linspace(0.0, y_max, 6).tolist()
    x_tick_candidates = x_values
    if len(x_tick_candidates) > 10:
        idx = np.linspace(0, len(x_tick_candidates) - 1, 10).round().astype(int).tolist()
        x_ticks = [x_tick_candidates[i] for i in sorted(set(idx))]
    else:
        x_ticks = x_tick_candidates

    delta = ppo_last - base_last

    base_points = polyline_points(base_curve)
    ppo_points = polyline_points(ppo_curve)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="55" font-size="42" font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="{margin_l}" y="84" font-size="18" fill="#475569">x: Active-Learning-Runde, y: Hit-Rate (%)</text>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#0f172a" stroke-width="2"/>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#0f172a" stroke-width="2"/>',
    ]

    for yt in y_ticks:
        y = sy(yt)
        lines.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{margin_l - 16}" y="{y + 6:.2f}" text-anchor="end" font-size="16" fill="#334155">{yt:.0f}</text>')

    for xt in x_ticks:
        x = sx(float(xt))
        lines.append(f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t + plot_h}" stroke="#f1f5f9" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{margin_t + plot_h + 34}" text-anchor="middle" font-size="16" fill="#334155">{int(xt)}</text>')

    lines.extend(
        [
            f'<polyline fill="none" stroke="#64748b" stroke-width="4" points="{base_points}"/>',
            f'<polyline fill="none" stroke="#0f766e" stroke-width="4" points="{ppo_points}"/>',
        ]
    )

    for x, y in zip(base_curve[round_col], base_curve["hit_rate_pct"]):
        lines.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="4.5" fill="#64748b"/>')
    for x, y in zip(ppo_curve[round_col], ppo_curve["hit_rate_pct"]):
        lines.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="4.5" fill="#0f766e"/>')

    legend_x = margin_l + plot_w - 240
    legend_y = margin_t + 20
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="220" height="80" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 28}" x2="{legend_x + 56}" y2="{legend_y + 28}" stroke="#64748b" stroke-width="4"/>',
            f'<text x="{legend_x + 68}" y="{legend_y + 34}" font-size="17" fill="#0f172a">ohne RL</text>',
            f'<line x1="{legend_x + 16}" y1="{legend_y + 56}" x2="{legend_x + 56}" y2="{legend_y + 56}" stroke="#0f766e" stroke-width="4"/>',
            f'<text x="{legend_x + 68}" y="{legend_y + 62}" font-size="17" fill="#0f172a">PPO</text>',
            f'<text x="{margin_l}" y="{height - 34}" font-size="18" fill="#0f172a">Finale Differenz (Runde {last_round}): {delta:+.1f} %-Punkte</text>',
            f'<text x="{margin_l + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Active-Learning-Runde</text>',
            f'<text x="40" y="{margin_t + plot_h / 2:.2f}" transform="rotate(-90 40 {margin_t + plot_h / 2:.2f})" text-anchor="middle" font-size="18" font-weight="700" fill="#0f172a">Hit-Rate (%)</text>',
            "</svg>",
        ]
    )

    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hit-rate per active-learning round: PPO vs baseline.")
    parser.add_argument("--baseline", required=True, help="Path to baseline active_learning_history.csv")
    parser.add_argument("--ppo", required=True, help="Path to PPO active_learning_history.csv")
    parser.add_argument("--output", default="experiments/showcase_ppo/ppo_vs_baseline_hitrate.png")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=-2.0,
        help="Used only if red_pass column is missing; hit := red_score >= threshold.",
    )
    parser.add_argument("--title", default="Hit-Rate pro Runde: PPO vs ohne RL")
    args = parser.parse_args()

    base_path = Path(args.baseline)
    ppo_path = Path(args.ppo)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(base_path)
    ppo_df = pd.read_csv(ppo_path)

    base_curve = _compute_hit_rate_by_round(base_df, score_threshold=args.score_threshold)
    ppo_curve = _compute_hit_rate_by_round(ppo_df, score_threshold=args.score_threshold)
    round_col = base_curve.columns[0]
    last_round = int(min(base_curve[round_col].max(), ppo_curve[round_col].max()))
    base_last = float(base_curve.loc[base_curve[round_col] == last_round, "hit_rate_pct"].iloc[-1])
    ppo_last = float(ppo_curve.loc[ppo_curve[round_col] == last_round, "hit_rate_pct"].iloc[-1])
    delta = ppo_last - base_last

    final_plot_path: Path
    if plt is not None:
        try:
            final_plot_path = _plot_with_matplotlib(
                base_curve=base_curve,
                ppo_curve=ppo_curve,
                round_col=round_col,
                out_path=out_path,
                title=args.title,
                last_round=last_round,
                base_last=base_last,
                ppo_last=ppo_last,
            )
        except Exception as exc:  # pragma: no cover
            print(f"warning: matplotlib plot failed ({exc}); writing SVG fallback instead.")
            final_plot_path = _write_svg_fallback(
                base_curve=base_curve,
                ppo_curve=ppo_curve,
                round_col=round_col,
                out_path=out_path,
                title=args.title,
                last_round=last_round,
                base_last=base_last,
                ppo_last=ppo_last,
            )
    else:
        print(f"warning: matplotlib unavailable ({_MATPLOTLIB_IMPORT_ERROR}); writing SVG fallback.")
        final_plot_path = _write_svg_fallback(
            base_curve=base_curve,
            ppo_curve=ppo_curve,
            round_col=round_col,
            out_path=out_path,
            title=args.title,
            last_round=last_round,
            base_last=base_last,
            ppo_last=ppo_last,
        )

    # Also export numeric summary for poster tables.
    summary = pd.DataFrame(
        {
            "metric": [
                "baseline_mean_hit_rate_pct",
                "ppo_mean_hit_rate_pct",
                "baseline_final_hit_rate_pct",
                "ppo_final_hit_rate_pct",
                "delta_final_pct_points",
            ],
            "value": [
                float(base_curve["hit_rate_pct"].mean()),
                float(ppo_curve["hit_rate_pct"].mean()),
                base_last,
                ppo_last,
                delta,
            ],
        }
    )
    summary_path = final_plot_path.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"wrote: {final_plot_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
