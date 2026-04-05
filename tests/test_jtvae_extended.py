import math
import torch

from src.models.jtvae_extended import (
    JTVAE,
    JTVDataset,
    train_jtvae,
    sample_conditional,
    _evaluate_jtvae,
    train_jtvae_rl_step,
)
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


def _build_rl_trace(model: JTVAE, n_samples: int = 6):
    samples, trace = model.sample_with_trace(
        n_samples=n_samples,
        max_tree_nodes=1,
        fragment_idx_to_smiles={0: "C"},
        device="cpu",
        assemble_kwargs={"adjacency_threshold": 0.5},
        temperature=1.0,
    )
    assert len(samples) == n_samples
    assert "log_prob" in trace
    assert "old_log_prob" in trace
    assert "old_value" in trace
    assert "frag_actions" in trace
    assert "adj_actions" in trace
    return samples, trace


def test_jtvae_policy_gradient_rl_step():
    model = JTVAE(
        tree_feat_dim=4,
        graph_feat_dim=5,
        fragment_vocab_size=1,
        z_dim=8,
        hidden_dim=16,
        cond_dim=0,
        max_tree_nodes=1,
    )
    _, trace = _build_rl_trace(model, n_samples=6)
    rewards = torch.linspace(0.0, 1.0, steps=6)
    baseline = {"value": 0.0}
    metrics, optimizer = train_jtvae_rl_step(
        model,
        trace,
        rewards,
        optimizer=None,
        lr=1e-3,
        algorithm="policy_gradient",
        entropy_weight=0.01,
        baseline_state=baseline,
        value_loss_weight=0.5,
        max_grad_norm=1.0,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metrics["algorithm"] == "policy_gradient"
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "total_loss" in metrics
    assert "grad_norm" in metrics
    assert math.isfinite(float(metrics["total_loss"]))
    assert math.isfinite(float(metrics["grad_norm"]))
    assert baseline["value"] > 0.0


def test_jtvae_ppo_rl_step():
    model = JTVAE(
        tree_feat_dim=4,
        graph_feat_dim=5,
        fragment_vocab_size=1,
        z_dim=8,
        hidden_dim=16,
        cond_dim=0,
        max_tree_nodes=1,
    )
    _, trace = _build_rl_trace(model, n_samples=8)
    rewards = torch.linspace(-0.2, 1.2, steps=8)
    metrics, optimizer = train_jtvae_rl_step(
        model,
        trace,
        rewards,
        optimizer=None,
        lr=1e-3,
        algorithm="ppo",
        entropy_weight=0.01,
        value_loss_weight=0.5,
        ppo_clip_ratio=0.2,
        ppo_epochs=2,
        ppo_minibatch_size=4,
        ppo_target_kl=1.0,
        max_grad_norm=1.0,
    )
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert metrics["algorithm"] == "ppo"
    assert float(metrics["ppo_update_steps"]) >= 1.0
    assert float(metrics["ppo_epochs_ran"]) >= 1.0
    assert "approx_kl" in metrics
    assert math.isfinite(float(metrics["approx_kl"]))
    assert math.isfinite(float(metrics["total_loss"]))
