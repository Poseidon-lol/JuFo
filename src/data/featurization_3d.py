"""3D featurization helpers for SchNet-style surrogate models.

This module converts MolBlocks or SMILES into PyG ``Data`` objects with:
- ``z``   atomic numbers, shape [N]
- ``pos`` 3D coordinates, shape [N, 3]
- ``y``   optional targets, shape [1, D]
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import contextlib

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

