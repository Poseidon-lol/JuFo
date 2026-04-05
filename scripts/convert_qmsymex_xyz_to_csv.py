#!/usr/bin/env python3
"""
Convert QM-symex XYZ files to a training CSV for the optical surrogate.

Output columns (default):
    dataset_id,file,smiles,mol,lambda_max_nm,oscillator_strength,best_state_index,n_atoms,source_tag,status,error

Notes:
- QM-symex transition lines contain singlet/triplet data per state. This script
  follows the same lambda/f selection logic as the ORCA parser in this repo:
  choose the state with the highest oscillator strength.
- SMILES are reconstructed from XYZ coordinates using RDKit bond perception.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _parse_transitions(lines: Iterable[str]) -> List[Dict[str, float | int]]:
    states: List[Dict[str, float | int]] = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        left, _, rest = line.partition("|")
        idx = left.strip()
        if not idx.isdigit():
            continue
        singlet_block = rest.split("|", 1)[0].strip()
        parts = singlet_block.split()
        # Expected: "<symmetry> <energy_eV> <wavelength_nm> <osc_strength> <spin>"
        if len(parts) < 4:
            continue
        ev = _to_float(parts[1])
        nm = _to_float(parts[2])
        fval = _to_float(parts[3])
        if nm is None:
            continue
        states.append(
            {
                "idx": int(idx),
                "ev": ev if ev is not None else float("nan"),
                "nm": nm,
                "f": fval if fval is not None else 0.0,
            }
        )
    return states


def _in_range(value: float, lower: Optional[float], upper: Optional[float]) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _select_transition(
    states: List[Dict[str, float | int]],
    *,
    mode: str,
    lambda_min: Optional[float],
    lambda_max: Optional[float],
) -> Optional[Dict[str, float | int]]:
    if not states:
        return None

    if mode == "best_f":
        return max(states, key=lambda s: float(s["f"]))

    if mode == "best_in_range":
        if lambda_min is None and lambda_max is None:
            return max(states, key=lambda s: float(s["f"]))
        in_range = [s for s in states if _in_range(float(s["nm"]), lambda_min, lambda_max)]
        if not in_range:
            return None
        return max(in_range, key=lambda s: float(s["f"]))

    raise ValueError(f"Unsupported transition selection mode: {mode}")


def _build_mol_from_xyz(lines: List[str], n_atoms: int, *, charge: int = 0) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    atom_rows: List[str] = []
    for row in lines[2 : 2 + n_atoms]:
        tokens = row.split()
        # QM-symex rows can have an extra atomic-charge token; keep only elem,x,y,z.
        if len(tokens) < 4:
            return None, f"invalid_atom_row:{row}"
        atom_rows.append(" ".join(tokens[:4]))
    xyz_block = "\n".join([str(n_atoms), "converted_qm_symex", *atom_rows]) + "\n"
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        return None, "rdkit_mol_from_xyz_failed"
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception as exc:
        return None, f"determine_bonds_failed:{exc}"
    return mol, None


def _parse_file(
    path: Path,
    *,
    charge: int = 0,
    include_mol: bool = True,
    transition_mode: str = "best_f",
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "dataset_id": path.stem,
        "file": str(path),
        "smiles": None,
        "mol": None,
        "lambda_max_nm": None,
        "oscillator_strength": None,
        "best_state_index": None,
        "n_atoms": None,
        "source_tag": "qm_symex_xyz",
        "status": "error",
        "error": None,
    }

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        record["error"] = f"read_failed:{exc}"
        return record
    if len(lines) < 3:
        record["error"] = "file_too_short"
        return record

    try:
        n_atoms = int(lines[0].strip())
    except Exception:
        record["error"] = "invalid_atom_count"
        return record
    record["n_atoms"] = n_atoms

    if len(lines) < 2 + n_atoms:
        record["error"] = "incomplete_xyz_block"
        return record

    mol, mol_error = _build_mol_from_xyz(lines, n_atoms, charge=charge)
    if mol is None:
        record["error"] = mol_error
        return record

    try:
        mol_no_h = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol_no_h, canonical=True, isomericSmiles=True)
        mol_block = Chem.MolToMolBlock(mol) if include_mol else ""
    except Exception as exc:
        record["error"] = f"mol_to_smiles_or_block_failed:{exc}"
        return record

    tail = lines[2 + n_atoms :]
    if tail and tail[0].strip().upper().startswith("HOMO"):
        tail = tail[1:]
    states = _parse_transitions(tail)
    selected = _select_transition(
        states,
        mode=transition_mode,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    if selected is None:
        if transition_mode == "best_in_range" and (lambda_min is not None or lambda_max is not None):
            record["status"] = "filtered"
            record["error"] = "no_transition_in_requested_range"
        else:
            record["error"] = "no_transition_data"
        return record

    record.update(
        {
            "smiles": smiles,
            "mol": mol_block,
            "lambda_max_nm": float(selected["nm"]),
            "oscillator_strength": float(selected["f"]),
            "best_state_index": int(selected["idx"]),
            "status": "ok",
            "error": "",
        }
    )
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert QM-symex XYZ files to CSV.")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw/QM_symex"))
    p.add_argument("--output", type=Path, default=Path("data/processed/qm_symex_optical.csv"))
    p.add_argument("--max-files", type=int, default=None, help="Optional cap for quick tests.")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--charge", type=int, default=0, help="Net charge for bond perception (default: 0).")
    p.add_argument("--lambda-min", type=float, default=None, help="Optional wavelength lower bound (nm).")
    p.add_argument("--lambda-max", type=float, default=None, help="Optional wavelength upper bound (nm).")
    p.add_argument("--f-min", type=float, default=None, help="Optional oscillator strength lower bound.")
    p.add_argument(
        "--transition-mode",
        choices=("best_f", "best_in_range"),
        default="best_f",
        help=(
            "How to pick the transition written to lambda_max_nm/f. "
            "'best_f' = strongest transition (default). "
            "'best_in_range' = strongest transition within [lambda-min,lambda-max] if present."
        ),
    )
    p.add_argument("--dedupe-smiles", action="store_true", help="Drop duplicate canonical SMILES.")
    p.add_argument(
        "--omit-mol",
        action="store_true",
        help="Do not store MolBlock text in the output (much smaller CSV).",
    )
    p.add_argument(
        "--keep-failed",
        action="store_true",
        help="Also write failed parse rows (without mol/smiles/targets).",
    )
    return p.parse_args()


def _passes_filters(
    record: Dict[str, object],
    *,
    lambda_min: Optional[float],
    lambda_max: Optional[float],
    f_min: Optional[float],
) -> bool:
    lam = record.get("lambda_max_nm")
    fval = record.get("oscillator_strength")
    if lam is None or fval is None:
        return False
    lam = float(lam)
    fval = float(fval)
    if lambda_min is not None and lam < lambda_min:
        return False
    if lambda_max is not None and lam > lambda_max:
        return False
    if f_min is not None and fval < f_min:
        return False
    return True


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.xyz"))
    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset_id",
        "file",
        "smiles",
        "mol",
        "lambda_max_nm",
        "oscillator_strength",
        "best_state_index",
        "n_atoms",
        "source_tag",
        "status",
        "error",
    ]

    seen_smiles = set()
    total = len(files)
    ok_count = 0
    fail_count = 0
    filtered_count = 0
    dedup_count = 0

    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, path in enumerate(files, start=1):
            rec = _parse_file(
                path,
                charge=int(args.charge),
                include_mol=not bool(args.omit_mol),
                transition_mode=str(args.transition_mode),
                lambda_min=args.lambda_min,
                lambda_max=args.lambda_max,
            )
            if rec.get("status") != "ok":
                if rec.get("status") == "filtered":
                    filtered_count += 1
                else:
                    fail_count += 1
                if args.keep_failed:
                    writer.writerow(rec)
                continue

            if not _passes_filters(
                rec,
                lambda_min=args.lambda_min,
                lambda_max=args.lambda_max,
                f_min=args.f_min,
            ):
                filtered_count += 1
                continue

            smiles = str(rec.get("smiles") or "").strip()
            if args.dedupe_smiles:
                if smiles in seen_smiles:
                    dedup_count += 1
                    continue
                seen_smiles.add(smiles)

            writer.writerow(rec)
            ok_count += 1

            if args.progress_every > 0 and i % args.progress_every == 0:
                print(
                    f"[progress] {i}/{total} files | written={ok_count} failed={fail_count} "
                    f"filtered={filtered_count} deduped={dedup_count}"
                )

    print(f"[done] input files: {total}")
    print(f"[done] written rows: {ok_count}")
    print(f"[done] failed parse: {fail_count}")
    print(f"[done] filtered out: {filtered_count}")
    print(f"[done] deduped: {dedup_count}")
    print(f"[done] output: {args.output}")


if __name__ == "__main__":
    main()
