#!/usr/bin/env python3
"""
Stage-A shortlist builder for red-gap candidates.

Ranks molecules by cheap proxies and optionally enforces diversity before ORCA.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RDKit_IMPORT_ERROR: Optional[Exception] = None
Chem = None
DataStructs = None
AllChem = None
MaxMinPicker = None
try:
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    try:
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker  # type: ignore
    except Exception:
        MaxMinPicker = None
except Exception as exc:
    RDKit_IMPORT_ERROR = exc


def _resolve_smiles_col(columns: Sequence[str], preferred: Optional[str]) -> str:
    if preferred and preferred in columns:
        return preferred
    for name in ("smiles", "smile", "SMILES"):
        if name in columns:
            return name
    raise KeyError(f"No SMILES column found. Available: {list(columns)}")


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).astype(float)


def _score_table(df: pd.DataFrame, optical_lumo_target: float) -> pd.DataFrame:
    work = df.copy()

    score_gap = pd.Series(0.5, index=work.index, dtype=float)
    if "red_gap_score" in work.columns:
        red = pd.to_numeric(work["red_gap_score"], errors="coerce")
        score_gap = _pct_rank(red.fillna(red.median() if red.notna().any() else 0.0))
    elif "dist_to_680_nm" in work.columns:
        dist = pd.to_numeric(work["dist_to_680_nm"], errors="coerce")
        score_gap = _pct_rank((-dist).fillna((-dist).median() if dist.notna().any() else 0.0))
    elif "lambda_from_gap_nm" in work.columns:
        lmb = pd.to_numeric(work["lambda_from_gap_nm"], errors="coerce")
        score_gap = _pct_rank((-np.abs(lmb - 680.0)).fillna(0.0))

    score_overlap = pd.Series(0.5, index=work.index, dtype=float)
    if "spectral_overlap" in work.columns:
        so = pd.to_numeric(work["spectral_overlap"], errors="coerce")
        so = np.log1p(so.clip(lower=0).fillna(0.0))
        score_overlap = _pct_rank(so)

    score_opt_lumo = pd.Series(0.5, index=work.index, dtype=float)
    if "optical_lumo" in work.columns:
        ol = pd.to_numeric(work["optical_lumo"], errors="coerce")
        score_opt_lumo = _pct_rank((-np.abs(ol - float(optical_lumo_target))).fillna(0.0))

    # Emphasize red-gap quality first, then overlap, then optical_lumo proximity.
    work["stagea_score"] = 0.60 * score_gap + 0.25 * score_overlap + 0.15 * score_opt_lumo
    return work.sort_values("stagea_score", ascending=False).reset_index(drop=True)


def _greedy_maxmin_pick(
    fps: Sequence[object],
    n_pick: int,
    *,
    seed: int,
    start_index: int = 0,
) -> List[int]:
    n_total = len(fps)
    n_pick = min(int(n_pick), n_total)
    if n_pick <= 0:
        return []
    if n_pick == n_total:
        return list(range(n_total))

    rng = np.random.default_rng(int(seed))
    chosen = [int(max(0, min(start_index, n_total - 1)))]
    remaining = np.ones(n_total, dtype=bool)
    remaining[chosen[0]] = False
    fps_list = list(fps)

    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_list[chosen[0]], fps_list), dtype=np.float32)
    min_dist = 1.0 - sims
    min_dist[~remaining] = -1.0

    while len(chosen) < n_pick:
        nxt = int(np.argmax(min_dist))
        if nxt < 0 or min_dist[nxt] < 0:
            candidates = np.where(remaining)[0]
            if candidates.size == 0:
                break
            nxt = int(rng.choice(candidates))
        chosen.append(nxt)
        remaining[nxt] = False
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_list[nxt], fps_list), dtype=np.float32)
        dist = 1.0 - sims
        min_dist = np.minimum(min_dist, dist)
        min_dist[~remaining] = -1.0
    return chosen


def _diverse_pick(smiles: Sequence[str], n_pick: int, *, seed: int, radius: int, nbits: int) -> List[int]:
    if Chem is None or DataStructs is None or AllChem is None:
        raise RuntimeError(f"RDKit missing for diversity picking: {RDKit_IMPORT_ERROR!r}")
    fps = []
    idx_map = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(nbits))
        fps.append(fp)
        idx_map.append(i)
    if not fps:
        return []
    n_pick = min(int(n_pick), len(fps))
    if n_pick <= 0:
        return []
    if MaxMinPicker is not None:
        try:
            picker = MaxMinPicker()
            picked_local = list(picker.LazyBitVectorPick(fps, len(fps), n_pick, int(seed)))
            return [idx_map[int(i)] for i in picked_local]
        except Exception:
            pass
    picked_local = _greedy_maxmin_pick(fps, n_pick=n_pick, seed=int(seed), start_index=0)
    return [idx_map[int(i)] for i in picked_local]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stage-A shortlist for red-gap ORCA labeling.")
    p.add_argument("--input", default="data/processed/opv_db_red_gap_650_780.csv")
    p.add_argument("--output", default="data/processed/opv_db_red_gap_650_780_stageA_top3000.csv")
    p.add_argument("--smiles-col", default=None, help="Optional explicit smiles column.")
    p.add_argument("--top-k", type=int, default=3000, help="Final shortlist size.")
    p.add_argument("--prefilter-factor", type=float, default=3.0, help="Keep top_k * factor before diversity.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--optical-lumo-target", type=float, default=-3.0)
    p.add_argument("--disable-diversity", action="store_true", help="Skip diversity pick and take pure top-k.")
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-nbits", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    smiles_col = _resolve_smiles_col(df.columns.tolist(), args.smiles_col)

    before = len(df)
    smiles = df[smiles_col].astype(str).str.strip()
    good = (~smiles.isna()) & (~smiles.str.lower().isin(["", "nan", "none"]))
    df = df.loc[good].copy()
    df[smiles_col] = smiles.loc[good]
    df = df.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)
    print(f"[info] rows: {before} -> {len(df)} after smiles cleanup + dedupe")

    scored = _score_table(df, optical_lumo_target=float(args.optical_lumo_target))

    top_k = max(1, int(args.top_k))
    pre_n = min(len(scored), max(top_k, int(math.ceil(top_k * float(args.prefilter_factor)))))
    pre = scored.head(pre_n).copy()
    print(f"[info] stage-A prefilter size: {pre_n}")

    if args.disable_diversity:
        out = pre.head(top_k).copy()
        print("[info] diversity disabled; using pure score ranking.")
    else:
        picked = _diverse_pick(
            pre[smiles_col].tolist(),
            n_pick=top_k,
            seed=int(args.seed),
            radius=int(args.fp_radius),
            nbits=int(args.fp_nbits),
        )
        if not picked:
            out = pre.head(top_k).copy()
            print("[warn] diversity pick returned empty; fallback to pure top-k.")
        else:
            out = pre.iloc[picked].copy().sort_values("stagea_score", ascending=False).reset_index(drop=True)
            print(f"[info] diversity-picked rows: {len(out)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()

