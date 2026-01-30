"""echtes schnet surrogate mit pyg schnet"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import json

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet as PygSchNet
from torch.nn.utils import clip_grad_norm_

from src.utils.device import get_device
import random
import numpy as np


@dataclass
class RealSchNetConfig:
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 10.0
    readout: str = "add"  # add | mean | max
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    epochs: int = 80
    patience: int = 10
    scheduler_patience: int = 10
    use_amp: bool = False
    grad_clip: float = 0.0
    loss: str = "l1"  # l1 | smooth_l1
    target_weights: Optional[List[float]] = None
    head_dropout: float = 0.0
    interaction_dropout: float = 0.0
    device: str = "auto"
    save_dir: Path = Path("models/surrogate_3d_full")
    n_models: int = 1
    seed: int = 1337


class RealSchNetModel(torch.nn.Module):
    """wrapper um pyg schnet mit multi target head"""

    def __init__(self, cfg: RealSchNetConfig, out_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        base = PygSchNet(
            hidden_channels=cfg.hidden_channels,
            num_filters=cfg.num_filters,
            num_interactions=cfg.num_interactions,
            num_gaussians=cfg.num_gaussians,
            cutoff=cfg.cutoff,
            readout=cfg.readout,
        )
        if cfg.interaction_dropout > 0:
            base.lin1 = torch.nn.Sequential(base.lin1, torch.nn.Dropout(cfg.interaction_dropout))
        # Replace final linear layer to support arbitrary target dimensions.
        head_in = base.lin1.out_features
        if cfg.head_dropout > 0:
            base.lin2 = torch.nn.Sequential(torch.nn.Dropout(cfg.head_dropout), torch.nn.Linear(head_in, out_dim))
        else:
            base.lin2 = torch.nn.Linear(head_in, out_dim)
        self.model = base
        # Buffers store target scaling for normalisation/denormalisation.
        self.register_buffer("target_mean", torch.zeros(out_dim))
        self.register_buffer("target_std", torch.ones(out_dim))

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.view(-1)
        std = std.view(-1).clamp(min=1e-6)
        if mean.numel() != self.target_mean.numel() or std.numel() != self.target_std.numel():
            raise ValueError("Target stats shape mismatch.")
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)

    def forward(self, z, pos, batch=None, *, normalized: bool = False):
        pred_norm = self.model(z, pos, batch)
        if pred_norm.dim() == 1:
            pred_norm = pred_norm.view(-1, 1)
        if normalized:
            return pred_norm
        return pred_norm * self.target_std + self.target_mean


def _resolve_device(device_spec: str):
    dev = get_device(device_spec)
    # torch_scatter passt nicht zu directml also cpu fallback
    if dev.type == "directml":
        dev = get_device("cpu")
    return dev


def train_schnet_full(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[RealSchNetConfig] = None,
    *,
    save_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple[RealSchNetModel, List[float]]:
    cfg = config or RealSchNetConfig()
    device = _resolve_device(cfg.device)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # compute target stats for normalization
    target_mean, target_std = _compute_target_stats(train_ds, target_dim)
    model = RealSchNetModel(cfg, out_dim=target_dim)
    model.set_target_stats(target_mean, target_std)
    model = model.to(device.target)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=cfg.scheduler_patience, factor=0.5, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.use_amp) and device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        if val_ds is not None
        else None
    )

    loss_is_l1 = str(cfg.loss).lower() == "l1"
    target_weights = None
    if cfg.target_weights is not None:
        target_weights = torch.tensor(cfg.target_weights, dtype=torch.float, device=device.target).view(-1)
        if target_weights.numel() != target_dim:
            raise ValueError(f"target_weights length {target_weights.numel()} != target_dim {target_dim}")

    history: List[float] = []
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device.target)
            opt.zero_grad()
            with torch.autocast(
                device_type=device.type if device.type in {"cuda", "cpu", "mps"} else "cpu",
                enabled=cfg.use_amp,
            ):
                pred_norm = model(batch.z, batch.pos, getattr(batch, "batch", None), normalized=True)
                y = batch.y.view(pred_norm.size(0), -1)
                y_norm = (y - model.target_mean) / model.target_std.clamp(min=1e-6)
                if loss_is_l1:
                    loss_raw = F.l1_loss(pred_norm, y_norm, reduction="none")
                else:
                    loss_raw = F.smooth_l1_loss(pred_norm, y_norm, reduction="none")
                if target_weights is not None:
                    loss_raw = loss_raw * target_weights.view(1, -1)
                loss = loss_raw.mean()
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * y_norm.size(0)
            count += y_norm.size(0)
        train_loss = total / max(1, count)

        val_loss = None
        val_metrics = None
        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            vcount = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device.target)
                    with torch.autocast(
                        device_type=device.type if device.type in {"cuda", "cpu", "mps"} else "cpu",
                        enabled=cfg.use_amp,
                    ):
                        pred_norm = model(batch.z, batch.pos, getattr(batch, "batch", None), normalized=True)
                        y = batch.y.view(pred_norm.size(0), -1)
                        y_norm = (y - model.target_mean) / model.target_std.clamp(min=1e-6)
                        if loss_is_l1:
                            loss_raw = F.l1_loss(pred_norm, y_norm, reduction="none")
                        else:
                            loss_raw = F.smooth_l1_loss(pred_norm, y_norm, reduction="none")
                        if target_weights is not None:
                            loss_raw = loss_raw * target_weights.view(1, -1)
                        loss = loss_raw.mean()
                    vtotal += loss.item() * y_norm.size(0)
                    vcount += y_norm.size(0)
            val_loss = vtotal / max(1, vcount)
            val_metrics = _compute_metrics(model, val_loader, device, target_dim)
        train_metrics = _compute_metrics(model, train_loader, device, target_dim, limit_batches=5)
        scheduler_metric = val_loss if val_loss is not None else train_loss
        scheduler.step(scheduler_metric)

        metric = val_loss if val_loss is not None else train_loss
        history.append(metric)

        if metric < best_loss:
            best_loss = metric
            bad_epochs = 0
            target_path = save_path if save_path is not None else (cfg.save_dir / "schnet_full.pt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target_path)
            metrics_payload = {
                "epoch": epoch + 1,
                "train_loss_norm": train_loss,
                "val_loss_norm": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "train_metrics_raw": _compute_metrics(model, train_loader, device, target_dim, denormalized=True, limit_batches=5),
                "val_metrics_raw": _compute_metrics(model, val_loader, device, target_dim, denormalized=True) if val_loader is not None else None,
                "target_dim": target_dim,
                "target_mean": model.target_mean.cpu().tolist(),
                "target_std": model.target_std.cpu().tolist(),
                "config": _config_to_dict(cfg),
                "device": device.type,
            }
            metrics_path = target_path.with_suffix(".metrics.json")
            with metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(metrics_payload, fh, indent=2)
            config_path = target_path.with_suffix(".config.json")
            with config_path.open("w", encoding="utf-8") as fh:
                json.dump(_config_to_dict(cfg), fh, indent=2)
        # lightweight live logging for raw-scale metrics
        if train_metrics and train_metrics.get("mae") is not None:
            train_mae = ", ".join(f"{m:.4f}" for m in train_metrics.get("mae", []))
            val_mae = None
            if val_metrics and val_metrics.get("mae") is not None:
                val_mae = ", ".join(f"{m:.4f}" for m in val_metrics.get("mae", []))
            msg = f"[epoch {epoch+1}] train_mae(norm): {train_loss:.4f} | train_mae(raw): {train_mae}"
            if val_loss is not None:
                msg += f" | val_mae(norm): {val_loss:.4f}"
            if val_mae:
                msg += f" | val_mae(raw): {val_mae}"
            print(msg)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    return model, history


def load_schnet_full(
    path: Path,
    target_dim: int,
    cfg: Optional[RealSchNetConfig] = None,
    *,
    ref_dataset=None,
) -> RealSchNetModel:
    cfg = cfg or _load_config_sidecar(path) or RealSchNetConfig()
    model = RealSchNetModel(cfg, out_dim=target_dim)
    state = torch.load(path, map_location="cpu")
    # tolerate checkpoints saved before target stats buffers were added
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and ref_dataset is not None:
        mean, std = _compute_target_stats(ref_dataset, target_dim)
        model.set_target_stats(mean, std)
    elif missing:
        # fallback: default stats, but warn user by raising informative error
        raise RuntimeError(
            "Checkpoint is missing target stats. Provide ref_dataset to load_schnet_full to recompute mean/std."
        )
    return model


def train_schnet_full_ensemble(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[RealSchNetConfig] = None,
) -> Tuple[List[RealSchNetModel], List[List[float]]]:
    cfg = config or RealSchNetConfig()
    models: List[RealSchNetModel] = []
    histories: List[List[float]] = []
    base_seed = getattr(cfg, "seed", 1337)
    n_models = max(1, int(getattr(cfg, "n_models", 1)))
    for idx in range(n_models):
        member_seed = base_seed + idx
        save_path = cfg.save_dir / f"schnet_full_member_{idx:02d}.pt"
        model, hist = train_schnet_full(
            train_ds,
            val_ds,
            target_dim=target_dim,
            config=cfg,
            save_path=save_path,
            seed=member_seed,
        )
        models.append(model)
        histories.append(hist)
    return models, histories


def _compute_target_stats(dataset, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = []
    for data in dataset:
        if getattr(data, "y", None) is None:
            continue
        arr = data.y.view(-1, target_dim).detach().cpu()
        ys.append(arr)
    if not ys:
        return torch.zeros(target_dim), torch.ones(target_dim)
    stacked = torch.cat(ys, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp(min=1e-6)
    return mean, std


def _compute_metrics(
    model: RealSchNetModel,
    loader: DataLoader,
    device,
    target_dim: int,
    *,
    limit_batches: Optional[int] = None,
    denormalized: bool = True,
) -> Optional[dict]:
    if loader is None:
        return None
    was_training = model.training
    model.eval()
    sum_abs = torch.zeros(target_dim, device="cpu")
    sum_sq = torch.zeros(target_dim, device="cpu")
    sum_err = torch.zeros(target_dim, device="cpu")
    count = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device.target)
            pred = model(batch.z, batch.pos, getattr(batch, "batch", None), normalized=not denormalized)
            if not denormalized:
                pred = pred * model.target_std + model.target_mean
            y = batch.y.view(pred.size(0), -1)
            diff = (pred - y).cpu()
            sum_abs += diff.abs().sum(dim=0)
            sum_sq += (diff ** 2).sum(dim=0)
            sum_err += diff.sum(dim=0)
            count += diff.size(0)
            if limit_batches is not None and idx + 1 >= limit_batches:
                break
    if was_training:
        model.train()
    if count == 0:
        return None
    mae = (sum_abs / count).tolist()
    rmse = torch.sqrt(sum_sq / count).tolist()
    bias = (sum_err / count).tolist()
    return {"mae": mae, "rmse": rmse, "bias": bias, "count": count}


def _config_to_dict(cfg: RealSchNetConfig) -> dict:
    raw = asdict(cfg)
    for k, v in list(raw.items()):
        if isinstance(v, Path):
            raw[k] = str(v)
    return raw


def _load_config_sidecar(path: Path) -> Optional[RealSchNetConfig]:
    cfg_path = path.with_suffix(".config.json")
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return RealSchNetConfig(**data)
    except Exception:
        return None
