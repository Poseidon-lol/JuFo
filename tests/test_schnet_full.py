import os
import pytest


def _try_import():
    try:
        import torch  # noqa: F401
        from torch_geometric.data import Data  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _try_import(), reason="torch_geometric not available")


def test_schnet_full_forward_minimal():
    import torch
    from torch_geometric.data import Data
    from src.models.schnet_full import RealSchNetModel, RealSchNetConfig

    z = torch.tensor([6, 6, 8], dtype=torch.long)  # C C O
    pos = torch.randn(3, 3)
    batch = torch.zeros(3, dtype=torch.long)
    y = torch.tensor([[0.0, 1.0]])  # two targets
    data = Data(z=z, pos=pos, y=y, batch=batch)

    model = RealSchNetModel(RealSchNetConfig(hidden_channels=32, num_filters=32, num_interactions=2, num_gaussians=10), out_dim=2)
    out = model(data.z, data.pos, data.batch)
    assert out.shape == (1, 2)


def _make_dummy_dataset(n: int = 10, target_dim: int = 2):
    import torch
    from torch_geometric.data import Data

    torch.manual_seed(0)
    ds = []
    for i in range(n):
        z = torch.tensor([6, 1, 8], dtype=torch.long)
        pos = torch.randn(3, 3)
        y = torch.full((1, target_dim), float(i + 1))
        ds.append(Data(z=z, pos=pos, y=y))
    return ds


def test_train_schnet_full_saves_metrics_and_config(tmp_path):
    import json
    from src.models.schnet_full import RealSchNetConfig, train_schnet_full, load_schnet_full

    train_ds = _make_dummy_dataset(8, target_dim=2)
    val_ds = _make_dummy_dataset(2, target_dim=2)
    save_path = tmp_path / "schnet_full.pt"
    cfg = RealSchNetConfig(
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=6,
        cutoff=5.0,
        readout="add",
        lr=5e-3,
        batch_size=4,
        epochs=3,
        patience=2,
        scheduler_patience=1,
        use_amp=True,
        grad_clip=0.1,
        head_dropout=0.1,
        interaction_dropout=0.1,
        device="cpu",
        save_dir=tmp_path,
    )
    model, hist = train_schnet_full(train_ds, val_ds, target_dim=2, config=cfg, save_path=save_path, seed=123)
    assert save_path.exists()
    metrics_path = save_path.with_suffix(".metrics.json")
    config_path = save_path.with_suffix(".config.json")
    assert metrics_path.exists()
    assert config_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "train_metrics_raw" in metrics and "val_metrics_raw" in metrics
    assert metrics["config"]["head_dropout"] == cfg.head_dropout
    loaded = load_schnet_full(save_path, target_dim=2, cfg=None)
    out = loaded(train_ds[0].z, train_ds[0].pos, getattr(train_ds[0], "batch", None))
    assert out.shape == (1, 2)
    assert len(hist) >= 1


def test_load_schnet_full_recompute_stats(tmp_path):
    import torch
    from src.models.schnet_full import RealSchNetModel, RealSchNetConfig, load_schnet_full

    cfg = RealSchNetConfig(hidden_channels=8, num_filters=8, num_interactions=1, num_gaussians=4, device="cpu", save_dir=tmp_path)
    model = RealSchNetModel(cfg, out_dim=1)
    # simulate old checkpoint missing target stats
    state = model.state_dict()
    state.pop("target_mean", None)
    state.pop("target_std", None)
    ckpt = tmp_path / "old_schnet.pt"
    torch.save(state, ckpt)
    ref_ds = _make_dummy_dataset(4, target_dim=1)
    loaded = load_schnet_full(ckpt, target_dim=1, cfg=cfg, ref_dataset=ref_ds)
    assert loaded.target_mean.abs().sum() > 0
    assert loaded.target_std.min() > 0
