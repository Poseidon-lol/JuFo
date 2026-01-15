"""featurisierung von smiles zu pyg graphen"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

# feature listen
ATOM_LIST = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "H"]
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

def one_hot_encoding(x, allowable_set):
    """liefert one hot fuer x im set"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

# atom features
def atom_features(atom):
    return torch.tensor(
        one_hot_encoding(atom.GetSymbol(), ATOM_LIST)
        + [atom.GetDegree(), atom.GetTotalNumHs(), atom.GetIsAromatic()],
        dtype=torch.float
    )

# bond features
def bond_features(bond):
    bt = [0, 0, 0, 0]
    bt[BOND_TYPES[bond.GetBondType()]] = 1
    return torch.tensor(bt, dtype=torch.float)

# hauptfunktion smiles -> graph
def mol_to_graph(smiles: str, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # atome
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # bonds
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES)), dtype=torch.float)

    # target(optional) 
    y_tensor = None
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float).view(1, -1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)

# kleines beispiel zum testen
if __name__ == "__main__":
    smiles = "c1ccccc1"  # benzene
    data = mol_to_graph(smiles, y=[-5.5, -2.5])  # bsp. HOMO/LUMO
    print(data)
    print("Node features:", data.x.shape)
    print("Edge index:", data.edge_index.shape)
