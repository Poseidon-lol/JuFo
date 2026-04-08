#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight labelled seed set for active-loop showcase runs."
    )
    parser.add_argument(
        "--input",
        default="data/processed/opv_db_red_gap_650_780.csv",
        help="Source CSV containing smile/homo/lumo/gap columns.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/opv_seed_showcase_300.csv",
        help="Output seed CSV path.",
    )
    parser.add_argument("--rows", type=int, default=300, help="Number of seed rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "smiles" not in df.columns and "smile" in df.columns:
        df = df.rename(columns={"smile": "smiles"})
    required = ["smiles", "homo", "lumo", "gap"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {in_path}: {missing}")

    work = df[required].copy()
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[(work["smiles"] != "") & (work["smiles"].str.lower() != "nan")]
    work = work.dropna(subset=["smiles", "homo", "lumo", "gap"])
    work = work.drop_duplicates(subset=["smiles"], keep="first")

    n = max(1, int(args.rows))
    if len(work) > n:
        work = work.sample(n=n, random_state=int(args.seed), replace=False)
    work = work.sort_values("smiles").reset_index(drop=True)
    work.to_csv(out_path, index=False)

    print(f"wrote: {out_path}")
    print(f"rows: {len(work)}")
    print(f"columns: {list(work.columns)}")


if __name__ == "__main__":
    main()

