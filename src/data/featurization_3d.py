"""3D featurization helpers for SchNet-style surrogate models.

This module converts MolBlocks or SMILES into PyG ``Data`` objects with:
- ``z``   atomic numbers, shape [N]
- ``pos`` 3D coordinates, shape [N, 3]
- ``y``   optional targets, shape [1, D]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import contextlib
import logging

import pandas as pd
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as exc:  # pragma: no cover - environment dependent
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
    from rdkit.Chem import rdDetermineBonds


logger = logging.getLogger(__name__)


def _ensure_rdkit() -> None:
    if Chem is None or AllChem is None:
        raise ImportError(
            "RDKit is required for 3D featurization. Install rdkit before using featurization_3d."
        ) from _IMPORT_ERROR


def _clean_smiles(smiles: object) -> Optional[str]:
    if smiles is None:
        return None
    if isinstance(smiles, float) and pd.isna(smiles):
        return None
    text = str(smiles).strip()
    return text if text else None


def _mol_from_molblock(mol_block: object) -> Optional["Chem.Mol"]:
    _ensure_rdkit()
    if mol_block is None:
        return None
    if isinstance(mol_block, float) and pd.isna(mol_block):
        return None
    text = str(mol_block).strip()
    if not text:
        return None

    # First try normal sanitization; then retry relaxed parsing.
    mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False, strictParsing=False)
    if mol is not None:
        return mol
    mol = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False, strictParsing=False)
    if mol is None:
        return None
    with contextlib.suppress(Exception):
        Chem.SanitizeMol(mol)
    return mol


def _embed_from_smiles(smiles: str) -> Optional["Chem.Mol"]:
    _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    code = AllChem.EmbedMolecule(mol, params)
    if code != 0:
        code = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D)
    if code != 0:
        return None

    with contextlib.suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _targets_to_tensor(targets: Optional[Sequence[float]]) -> Optional[torch.Tensor]:
    if targets is None:
        return None
    if len(targets) == 0:
        return None
    return torch.tensor(list(targets), dtype=torch.float).view(1, -1)


def _mol_to_data(
    mol: "Chem.Mol",
    *,
    y: Optional[Sequence[float]] = None,
    smiles: Optional[str] = None,
) -> Optional[Data]:
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer(0)
    zs: List[int] = []
    poss: List[List[float]] = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = conf.GetAtomPosition(atom_idx)
        zs.append(int(atom.GetAtomicNum()))
        poss.append([float(pos.x), float(pos.y), float(pos.z)])

    data = Data(
        z=torch.tensor(zs, dtype=torch.long),
        pos=torch.tensor(poss, dtype=torch.float),
    )
    y_tensor = _targets_to_tensor(y)
    if y_tensor is not None:
        data.y = y_tensor
    if smiles:
        data.smiles = smiles
    return data


def molblock_to_data(
    mol_block: object,
    *,
    smiles: Optional[str] = None,
    y: Optional[Sequence[float]] = None,
) -> Optional[Data]:
    """Convert a MolBlock (or fallback SMILES) to a 3D ``Data`` sample."""

    _ensure_rdkit()
    clean_smiles = _clean_smiles(smiles)
    mol = _mol_from_molblock(mol_block)

    # Fallback to conformer generation from SMILES if MolBlock is missing/invalid.
    if mol is None and clean_smiles:
        mol = _embed_from_smiles(clean_smiles)
    elif mol is not None and mol.GetNumConformers() == 0 and clean_smiles:
        fallback = _embed_from_smiles(clean_smiles)
        if fallback is not None:
            mol = fallback

    if mol is None:
        return None
    return _mol_to_data(mol, y=y, smiles=clean_smiles)


def dataframe_to_3d_dataset(
    df: pd.DataFrame,
    *,
    mol_col: str = "mol",
    smiles_col: str = "smiles",
    target_cols: Optional[Iterable[str]] = None,
) -> List[Data]:
    """Convert a dataframe into a list of 3D PyG ``Data`` samples."""

    _ensure_rdkit()
    target_cols = list(target_cols or [])

    missing = [col for col in target_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing target columns in dataframe: {missing}")
    if mol_col not in df.columns and smiles_col not in df.columns:
        raise KeyError(
            f"Expected at least one of columns '{mol_col}' or '{smiles_col}' in dataframe."
        )

    dataset: List[Data] = []
    has_mol = mol_col in df.columns
    has_smiles = smiles_col in df.columns

    for _, row in df.iterrows():
        if target_cols:
            if any(pd.isna(row[col]) for col in target_cols):
                continue
            y = [float(row[col]) for col in target_cols]
        else:
            y = None

        mol_block = row[mol_col] if has_mol else None
        smiles_val = _clean_smiles(row[smiles_col]) if has_smiles else None
        data = molblock_to_data(mol_block, smiles=smiles_val, y=y)
        if data is None:
            continue
        dataset.append(data)

    return dataset


def _qmsymex_to_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _qmsymex_parse_transitions(lines: Iterable[str]) -> List[Dict[str, float | int]]:
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
        # expected: "<symmetry> <energy_eV> <wavelength_nm> <osc_strength> <spin>"
        if len(parts) < 4:
            continue
        ev = _qmsymex_to_float(parts[1])
        nm = _qmsymex_to_float(parts[2])
        fval = _qmsymex_to_float(parts[3])
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


def _qmsymex_in_range(value: float, lower: Optional[float], upper: Optional[float]) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _qmsymex_select_transition(
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
        in_range = [s for s in states if _qmsymex_in_range(float(s["nm"]), lambda_min, lambda_max)]
        if not in_range:
            return None
        return max(in_range, key=lambda s: float(s["f"]))
    raise ValueError(f"Unsupported transition mode: {mode}")


def qmsymex_xyz_dir_to_3d_dataset(
    xyz_dir: Path | str,
    *,
    target_cols: Sequence[str],
    transition_mode: str = "best_f",
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
    f_min: Optional[float] = None,
    charge: int = 0,
    max_files: Optional[int] = None,
    dedupe_smiles: bool = False,
    progress_every: int = 2000,
) -> List[Data]:
    """Build a 3D dataset directly from a QM-symex XYZ folder.

    Supported target columns are aliases of:
    - lambda_max_nm
    - oscillator_strength
    """

    _ensure_rdkit()
    xyz_dir = Path(xyz_dir)
    if not xyz_dir.exists() or not xyz_dir.is_dir():
        raise FileNotFoundError(f"QM-symex xyz_dir not found or not a directory: {xyz_dir}")

    target_map: List[str] = []
    for col in target_cols:
        key = str(col).strip().lower()
        if key in {"lambda_max_nm", "lambda_max", "lambda_nm"}:
            target_map.append("lambda_max_nm")
        elif key in {"oscillator_strength", "f", "f_osc"}:
            target_map.append("oscillator_strength")
        else:
            raise ValueError(
                f"Unsupported QM-symex target column '{col}'. "
                "Allowed: lambda_max_nm/lambda_max, oscillator_strength/f."
            )

    files = sorted(xyz_dir.glob("*.xyz"))
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]

    dataset: List[Data] = []
    seen_smiles: set[str] = set()
    n_failed = 0
    n_filtered = 0
    n_dup = 0

    for i, path in enumerate(files, start=1):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) < 3:
                n_failed += 1
                continue
            n_atoms = int(lines[0].strip())
            if len(lines) < 2 + n_atoms:
                n_failed += 1
                continue
            atom_rows: List[str] = []
            for row in lines[2 : 2 + n_atoms]:
                toks = row.split()
                if len(toks) < 4:
                    atom_rows = []
                    break
                atom_rows.append(" ".join(toks[:4]))  # drop optional partial-charge token
            if not atom_rows:
                n_failed += 1
                continue

            xyz_block = "\n".join([str(n_atoms), "qmsymex_xyz", *atom_rows]) + "\n"
            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                n_failed += 1
                continue
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
            except Exception:
                n_failed += 1
                continue

            mol_no_h = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol_no_h, canonical=True, isomericSmiles=True)
            if dedupe_smiles:
                if smiles in seen_smiles:
                    n_dup += 1
                    continue
                seen_smiles.add(smiles)

            tail = lines[2 + n_atoms :]
            if tail and tail[0].strip().upper().startswith("HOMO"):
                tail = tail[1:]
            states = _qmsymex_parse_transitions(tail)
            selected = _qmsymex_select_transition(
                states,
                mode=str(transition_mode),
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            )
            if selected is None:
                n_filtered += 1
                continue
            lam = float(selected["nm"])
            fval = float(selected["f"])
            if f_min is not None and fval < float(f_min):
                n_filtered += 1
                continue

            y: List[float] = []
            for mapped in target_map:
                if mapped == "lambda_max_nm":
                    y.append(lam)
                elif mapped == "oscillator_strength":
                    y.append(fval)
                else:
                    raise RuntimeError(f"Internal target mapping error: {mapped}")

            data = _mol_to_data(mol, y=y, smiles=smiles)
            if data is None:
                n_failed += 1
                continue
            dataset.append(data)
        except Exception:
            n_failed += 1
            continue

        if progress_every > 0 and i % int(progress_every) == 0:
            logger.info(
                "QM-symex loader progress: %d/%d files | dataset=%d failed=%d filtered=%d deduped=%d",
                i,
                len(files),
                len(dataset),
                n_failed,
                n_filtered,
                n_dup,
            )

    logger.info(
        "QM-symex loader done: files=%d dataset=%d failed=%d filtered=%d deduped=%d",
        len(files),
        len(dataset),
        n_failed,
        n_filtered,
        n_dup,
    )
    return dataset
