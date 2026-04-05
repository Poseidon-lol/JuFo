#!/usr/bin/env python3
"""
Build a focused, diverse shortlist for optical QC labeling.

Pipeline:
1. Load OPV database and normalize SMILES.
2. Apply proxy windows on HOMO/LUMO/gap.
3. Apply structural quality filters (neutrality, aromaticity, conjugation, flexibility).
4. Rank by optical proxy (`optical_lumo` by default), keep top prefilter slice.
5. Select a diverse subset with fingerprint max-min picking.

Default output is a 10k shortlist for TD-DFT labeling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IMPORT_ERROR: Optional[Exception] = None
MaxMinPicker = None
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdMolDescriptors
    try:
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker  # type: ignore
    except Exception:
        MaxMinPicker = None
except Exception as exc:  # pragma: no cover - environment-dependent
    IMPORT_ERROR = exc
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    AllChem = None  # type: ignore
    rdMolDescriptors = None  # type: ignore


def _require_rdkit() -> None:
    if IMPORT_ERROR is not None or Chem is None or DataStructs is None or AllChem is None or rdMolDescriptors is None:
        raise RuntimeError(
            "RDKit is required for build_rotlicht_funnel.py. "
            f"Import error: {IMPORT_ERROR!r}"
        )


def _resolve_smiles_col(columns: Iterable[str], preferred: Optional[str]) -> str:
    available = list(columns)
    if preferred and preferred in available:
        return preferred
    for candidate in ("smiles", "smile", "SMILES"):
        if candidate in available:
            return candidate
    raise KeyError(f"No SMILES column found. Available columns: {available}")


def _between(series: pd.Series, lo: Optional[float], hi: Optional[float]) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if lo is not None:
        mask &= series >= float(lo)
    if hi is not None:
        mask &= series <= float(hi)
    return mask


def _conjugated_bond_count(mol: "Chem.Mol") -> int:
    return int(sum(1 for b in mol.GetBonds() if b.GetIsConjugated()))


def _charged_atom_count(mol: "Chem.Mol") -> int:
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0))


def _sanitize_and_filter_structures(
    frame: pd.DataFrame,
    smiles_col: str,
    *,
    require_neutral: bool,
    max_charged_atoms: Optional[int],
    min_aromatic_rings: Optional[int],
    max_rotatable_bonds: Optional[int],
    min_conjugated_bonds: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "missing_smiles": 0,
        "invalid_smiles": 0,
        "charged_molecule": 0,
        "too_many_charged_atoms": 0,
        "too_few_aromatic_rings": 0,
        "too_many_rotatable": 0,
        "too_few_conjugated": 0,
    }

    canonical_smiles: List[Optional[str]] = [None] * len(frame)
    keep = np.zeros(len(frame), dtype=bool)
    smiles_values = frame[smiles_col].tolist()
    for i, raw in enumerate(smiles_values):
        if raw is None:
            stats["missing_smiles"] += 1
            continue
        smi = str(raw).strip()
        if not smi or smi.lower() in {"nan", "none"}:
            stats["missing_smiles"] += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_smiles"] += 1
            continue

        if require_neutral and Chem.GetFormalCharge(mol) != 0:
            stats["charged_molecule"] += 1
            continue
        if max_charged_atoms is not None and _charged_atom_count(mol) > int(max_charged_atoms):
            stats["too_many_charged_atoms"] += 1
            continue
        if min_aromatic_rings is not None:
            aromatic = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            if aromatic < int(min_aromatic_rings):
                stats["too_few_aromatic_rings"] += 1
                continue
        if max_rotatable_bonds is not None:
            rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
            if rot > int(max_rotatable_bonds):
                stats["too_many_rotatable"] += 1
                continue
        if min_conjugated_bonds is not None:
            conj = _conjugated_bond_count(mol)
            if conj < int(min_conjugated_bonds):
                stats["too_few_conjugated"] += 1
                continue

        canonical_smiles[i] = Chem.MolToSmiles(mol, isomericSmiles=True)
        keep[i] = True

    out = frame.loc[keep].copy()
    out["smiles"] = [canonical_smiles[i] for i, flag in enumerate(keep) if flag]
    out = out.drop_duplicates(subset="smiles").reset_index(drop=True)
    return out, stats


def _optical_prefilter(
    frame: pd.DataFrame,
    *,
    optical_col: str,
    optical_mode: str,
    optical_target: Optional[float],
    prefilter_fraction: float,
    target_size: int,
) -> pd.DataFrame:
    if optical_col not in frame.columns:
        print(f"[warn] Optical column '{optical_col}' not present. Skipping optical prefilter.")
        return frame.copy().reset_index(drop=True)

    values = pd.to_numeric(frame[optical_col], errors="coerce")
    work = frame.loc[values.notna()].copy()
    if work.empty:
        print(f"[warn] Optical column '{optical_col}' is empty after numeric coercion. Skipping optical prefilter.")
        return frame.copy().reset_index(drop=True)

    work["_optical_proxy"] = values.loc[work.index].astype(float)
    mode = optical_mode.lower()
    if mode == "min":
        work = work.sort_values("_optical_proxy", ascending=True)
    elif mode == "max":
        work = work.sort_values("_optical_proxy", ascending=False)
    elif mode == "target":
        if optical_target is None:
            raise ValueError("optical_mode='target' requires --optical-target.")
        work["_optical_proxy_dist"] = (work["_optical_proxy"] - float(optical_target)).abs()
        work = work.sort_values("_optical_proxy_dist", ascending=True)
    else:
        raise ValueError(f"Unsupported optical mode: {optical_mode}")

    keep_n = int(math.ceil(len(work) * prefilter_fraction))
    keep_n = max(int(target_size), keep_n)
    keep_n = min(len(work), keep_n)
    return work.head(keep_n).reset_index(drop=True)


def _fingerprints_from_smiles(smiles: Sequence[str], *, radius: int, nbits: int) -> Tuple[List[object], List[int]]:
    fps: List[object] = []
    idx_map: List[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(nbits))
        fps.append(fp)
        idx_map.append(i)
    return fps, idx_map


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
    start_index = int(max(0, min(start_index, n_total - 1)))
    chosen = [start_index]
    fps_list = list(fps)

    remaining = np.ones(n_total, dtype=bool)
    remaining[start_index] = False
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_list[start_index], fps_list), dtype=np.float32)
    min_dist = 1.0 - sims
    min_dist[~remaining] = -1.0

    while len(chosen) < n_pick:
        next_idx = int(np.argmax(min_dist))
        if next_idx < 0 or min_dist[next_idx] < 0:
            candidates = np.where(remaining)[0]
            if candidates.size == 0:
                break
            next_idx = int(rng.choice(candidates))

        chosen.append(next_idx)
        remaining[next_idx] = False

        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps_list[next_idx], fps_list), dtype=np.float32)
        dist = 1.0 - sims
        min_dist = np.minimum(min_dist, dist)
        min_dist[~remaining] = -1.0

        if len(chosen) % 500 == 0 or len(chosen) == n_pick:
            print(f"[info] greedy maxmin progress: {len(chosen)}/{n_pick}")

    return chosen


def _pick_diverse_subset(
    fps: Sequence[object],
    n_pick: int,
    *,
    seed: int,
    force_greedy: bool,
) -> List[int]:
    n_total = len(fps)
    n_pick = min(int(n_pick), n_total)
    if n_pick <= 0:
        return []
    if n_pick == n_total:
        return list(range(n_total))

    if not force_greedy and MaxMinPicker is not None:
        try:
            picker = MaxMinPicker()
            picked = list(picker.LazyBitVectorPick(fps, n_total, n_pick, int(seed)))
            return [int(i) for i in picked]
        except Exception as exc:
            print(f"[warn] MaxMinPicker failed ({exc}); falling back to greedy max-min.")

    return _greedy_maxmin_pick(fps, n_pick, seed=seed, start_index=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a focused + diverse optical shortlist.")
    parser.add_argument("--input", default="data/raw/opv_db.csv", help="Input CSV path.")
    parser.add_argument(
        "--output",
        default="data/processed/opv_optical_shortlist_10k.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--smiles-col", default=None, help="SMILES column name (optional).")
    parser.add_argument("--target-size", type=int, default=10000, help="Final shortlist size.")
    parser.add_argument(
        "--prefilter-fraction",
        type=float,
        default=0.20,
        help="Fraction kept after optical proxy ranking before diversity sampling.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")

    parser.add_argument("--homo-min", type=float, default=-6.1)
    parser.add_argument("--homo-max", type=float, default=-4.9)
    parser.add_argument("--lumo-min", type=float, default=-3.8)
    parser.add_argument("--lumo-max", type=float, default=-2.4)
    parser.add_argument("--gap-min", type=float, default=1.2)
    parser.add_argument("--gap-max", type=float, default=2.8)

    parser.add_argument("--optical-col", type=str, default="optical_lumo")
    parser.add_argument(
        "--optical-mode",
        choices=("min", "max", "target"),
        default="min",
        help="How optical proxy is ranked before diversity sampling.",
    )
    parser.add_argument(
        "--optical-target",
        type=float,
        default=None,
        help="Required when --optical-mode target.",
    )

    parser.add_argument(
        "--allow-charged",
        action="store_true",
        help="Allow non-neutral molecules. Default is neutral-only.",
    )
    parser.add_argument("--max-charged-atoms", type=int, default=0)
    parser.add_argument("--min-aromatic-rings", type=int, default=2)
    parser.add_argument("--max-rotatable-bonds", type=int, default=10)
    parser.add_argument("--min-conjugated-bonds", type=int, default=8)

    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-nbits", type=int, default=2048)
    parser.add_argument(
        "--force-greedy-maxmin",
        action="store_true",
        help="Use greedy max-min picker even when RDKit MaxMinPicker is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_rdkit()

    if args.target_size <= 0:
        raise ValueError("--target-size must be > 0")
    if not (0.0 < float(args.prefilter_fraction) <= 1.0):
        raise ValueError("--prefilter-fraction must be in (0, 1].")
    if args.optical_mode == "target" and args.optical_target is None:
        raise ValueError("--optical-target is required when --optical-mode target.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")
    smiles_col = _resolve_smiles_col(df.columns, args.smiles_col)

    required_cols = ("homo", "lumo", "gap")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required proxy columns in input CSV: {missing}")

    for col in ("homo", "lumo", "gap", args.optical_col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[info] loaded rows: {len(df)}")
    proxy_mask = (
        _between(df["homo"], args.homo_min, args.homo_max)
        & _between(df["lumo"], args.lumo_min, args.lumo_max)
        & _between(df["gap"], args.gap_min, args.gap_max)
    )
    work = df.loc[proxy_mask].copy()
    print(f"[info] after proxy windows (homo/lumo/gap): {len(work)}")
    if work.empty:
        raise RuntimeError("No rows left after proxy windows.")

    work, struct_stats = _sanitize_and_filter_structures(
        work,
        smiles_col,
        require_neutral=(not bool(args.allow_charged)),
        max_charged_atoms=args.max_charged_atoms,
        min_aromatic_rings=args.min_aromatic_rings,
        max_rotatable_bonds=args.max_rotatable_bonds,
        min_conjugated_bonds=args.min_conjugated_bonds,
    )
    print(f"[info] after structural + canonical filters: {len(work)}")
    for key, val in struct_stats.items():
        if int(val) > 0:
            print(f"[info] dropped {key}: {val}")
    if work.empty:
        raise RuntimeError("No rows left after structural/canonical filters.")

    before_prefilter = len(work)
    work = _optical_prefilter(
        work,
        optical_col=args.optical_col,
        optical_mode=args.optical_mode,
        optical_target=args.optical_target,
        prefilter_fraction=float(args.prefilter_fraction),
        target_size=int(args.target_size),
    )
    print(f"[info] after optical prefilter: {len(work)} (from {before_prefilter})")
    if work.empty:
        raise RuntimeError("No rows left after optical prefilter.")

    fps, idx_map = _fingerprints_from_smiles(
        work["smiles"].astype(str).tolist(),
        radius=int(args.fp_radius),
        nbits=int(args.fp_nbits),
    )
    if not fps:
        raise RuntimeError("Could not build fingerprints for any shortlisted molecule.")
    if len(idx_map) != len(work):
        missing_fp = len(work) - len(idx_map)
        print(f"[warn] dropping {missing_fp} rows without fingerprint.")
        work = work.iloc[idx_map].reset_index(drop=True)

    pick_size = min(int(args.target_size), len(work))
    picked_local = _pick_diverse_subset(
        fps,
        pick_size,
        seed=int(args.seed),
        force_greedy=bool(args.force_greedy_maxmin),
    )
    selected = work.iloc[picked_local].copy().reset_index(drop=True)

    selected["funnel_stage"] = "optical_prefilter_diverse"
    selected["funnel_target_size"] = int(args.target_size)
    selected["funnel_seed"] = int(args.seed)
    selected.to_csv(output_path, index=False)

    print(f"[done] wrote {len(selected)} rows to {output_path}")


if __name__ == "__main__":
    main()
