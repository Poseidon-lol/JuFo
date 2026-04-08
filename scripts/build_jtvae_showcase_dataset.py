#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _pick_smiles_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Requested smiles column '{explicit}' not found.")
        return explicit
    for candidate in ("smiles", "smile"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No smiles column found. Expected one of: 'smiles', 'smile'.")


def _parse_target_columns(raw: str) -> List[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("target columns list is empty.")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight JT-VAE showcase dataset (CPU/GPU friendly)."
    )
    parser.add_argument(
        "--input",
        default="data/processed/opv_db_red_gap_top5k_around_680.csv",
        help="Input CSV with smiles + target columns.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/jtvae_showcase_thin.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Optional explicit smiles column name. Auto-detects smiles/smile when omitted.",
    )
    parser.add_argument(
        "--target-columns",
        default="homo,lumo,gap",
        help="Comma-separated target columns for conditioning.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1200,
        help="Maximum number of rows in the showcase dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    target_cols = _parse_target_columns(args.target_columns)
    df = pd.read_csv(in_path)
    smiles_col = _pick_smiles_column(df, args.smiles_column)

    required = [smiles_col, *target_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[required].copy()
    if smiles_col != "smiles":
        work = work.rename(columns={smiles_col: "smiles"})
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[(work["smiles"] != "") & (work["smiles"].str.lower() != "nan")]
    work = work.dropna(subset=["smiles", *target_cols])
    work = work.drop_duplicates(subset=["smiles"], keep="first")

    max_rows = max(1, int(args.max_rows))
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=int(args.seed), replace=False)

    # Stable order for reproducible dashboards.
    work = work.sort_values("smiles").reset_index(drop=True)
    work.to_csv(out_path, index=False)

    print(f"wrote: {out_path}")
    print(f"rows: {len(work)}")
    print(f"columns: {list(work.columns)}")


if __name__ == "__main__":
    main()

