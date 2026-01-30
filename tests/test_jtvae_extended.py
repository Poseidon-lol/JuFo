import os
import torch

from src.models.jtvae_extended import JTVAE, JTVDataset, train_jtvae, sample_conditional, _evaluate_jtvae
from src.utils.device import get_device


def _dummy_dataset(n=4, cond_dim=1, max_nodes=1):
    examples = []
    for i in range(n):
        tree_x = torch.randn(max_nodes, 4)
        tree_edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph_x = torch.randn(2, 5)
        graph_edge_index = torch.zeros((2, 0), dtype=torch.long)
        target_frag_idxs = torch.zeros((max_nodes,), dtype=torch.long)
        tree_adj = torch.zeros((max_nodes, max_nodes))
        cond = torch.full((cond_dim,), float(i))
        examples.append(
            {
                "tree_x": tree_x,
                "tree_edge_index": tree_edge_index,
                "graph_x": graph_x,
                "graph_edge_index": graph_edge_index,
                "target_frag_idxs": target_frag_idxs,
                "tree_adj": tree_adj,
                "cond": cond,
                "smiles": "C",
            }
        )
    return JTVDataset(examples)


def test_jtvae_train_and_metrics(tmp_path):
    fragment_vocab = {"C": 0}
    dataset = _dummy_dataset(max_nodes=1)
    model = JTVAE(
        tree_feat_dim=4,
        graph_feat_dim=5,
        fragment_vocab_size=len(fragment_vocab),
        z_dim=8,
        hidden_dim=16,
        cond_dim=1,
        max_tree_nodes=1,
    )
    model = train_jtvae(
        model,
        dataset,
        fragment_vocab,
        device="cpu",
        epochs=2,
        batch_size=2,
        lr=1e-2,
        save_dir=str(tmp_path),
        kl_weight=0.1,
        property_weight=0.1,
        adj_weight=0.0,
        scheduler_patience=1,
        scheduler_factor=0.5,
        max_grad_norm=1.0,
        cond_stats={"mean": [0.0], "std": [1.0]},
    )
    assert (tmp_path / "jtvae_best.pt").exists()
    # Metrics helper
    device_spec = get_device("cpu")
    metrics = _evaluate_jtvae(model, dataset, fragment_vocab, device_spec, train_smiles={"C"})
    assert "recon_accuracy" in metrics
    # Sampling should run and return requested count
    samples = sample_conditional(
        model,
        fragment_vocab,
        n_samples=4,
        cond_stats={"mean": [0.0], "std": [1.0]},
        assemble_kwargs={"max_tree_nodes": 1},
        device="cpu",
    )
    assert len(samples) == 4
