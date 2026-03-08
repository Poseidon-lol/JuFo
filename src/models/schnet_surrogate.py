"""leichter 3d surrogate ohne schwere schnet deps"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import time

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from src.utils.dashboard_server import start_dashboard_server
from src.utils.schnet_dashboard import write_schnet_live_dashboard


@dataclass
class SchNetConfig:
    hidden_channels: int = 128
    num_filters: int = 128  # legacy feld unused
    num_interactions: int = 6  # legacy feld unused
    num_gaussians: int = 50  # legacy feld unused
    cutoff: float = 10.0  # legacy feld unused
    readout: str = "mean"
    num_embeddings: int = 120  # max atomnummer fuer embedding
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 80
    patience: int = 10
    device: str = "cpu"  # auto cpu cuda oder explizit
    save_dir: Path = Path("models/surrogate_3d")
    target_names: Optional[List[str]] = None
    live_dashboard: bool = False
    live_dashboard_path: Optional[Path] = None
    live_dashboard_refresh_ms: int = 900
    live_dashboard_local_view: bool = False
    live_dashboard_host: str = "127.0.0.1"
    live_dashboard_port: int = 0
    live_dashboard_open_browser: bool = True


class SchNetModel(torch.nn.Module):
    """simpler point cloud encoder fuer 3d molekuele"""

    def __init__(self, cfg: SchNetConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.atom_emb = nn.Embedding(cfg.num_embeddings, cfg.hidden_channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_channels),
            nn.SiLU(),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
        )
        self.head = nn.Linear(cfg.hidden_channels, out_dim)

    def forward(self, z, pos, batch=None, mask=None):
        # clip embedding range falls exotische atome
        z_clamped = torch.clamp(z, max=self.cfg.num_embeddings - 1)
        h = self.atom_emb(z_clamped) + self.pos_mlp(pos)
        h = self.encoder(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        pooled = global_mean_pool(h, batch)
        return self.head(pooled)


def train_schnet(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[SchNetConfig] = None,
) -> Tuple[SchNetModel, List[float]]:
    cfg = config or SchNetConfig()
    device = _resolve_device(cfg.device)
    model = _build_schnet(cfg, target_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = float("inf")
    bad_epochs = 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None else None
    history: List[float] = []
    live_history: List[dict] = []
    live_lines: List[str] = []
    live_server = None
    live_url = None
    live_enabled = bool(getattr(cfg, "live_dashboard", False))
    live_path = (
        Path(cfg.live_dashboard_path)
        if getattr(cfg, "live_dashboard_path", None)
        else (cfg.save_dir / "schnet_live_dashboard.html")
    )
    live_refresh_ms = max(200, int(getattr(cfg, "live_dashboard_refresh_ms", 900)))
    started_at = time.time()

    def _append_live(msg: str) -> None:
        if not live_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        live_lines.append(f"[{stamp}] {msg}")
        if len(live_lines) > 240:
            del live_lines[:-240]

    if live_enabled:
        write_schnet_live_dashboard(
            live_path,
            title="SchNet Live Dashboard",
            epoch=0,
            total_epochs=cfg.epochs,
            best_metric=float(best_loss),
            lr=float(opt.param_groups[0]["lr"]),
            history=[],
            refresh_ms=live_refresh_ms,
            started_at=started_at,
            target_names=cfg.target_names,
            cli_lines=live_lines,
        )
        if bool(getattr(cfg, "live_dashboard_local_view", False)):
            try:
                live_server, live_url = start_dashboard_server(
                    live_path,
                    host=str(getattr(cfg, "live_dashboard_host", "127.0.0.1")),
                    port=int(getattr(cfg, "live_dashboard_port", 0)),
                    open_browser=bool(getattr(cfg, "live_dashboard_open_browser", True)),
                )
                _append_live(f"Local view: {live_url}")
            except Exception as exc:
                _append_live(f"Local view failed: {exc}")

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        sum_abs_train = torch.zeros(target_dim, dtype=torch.float)
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
            y = batch.y.view(pred.size(0), -1)
            loss = torch.nn.functional.l1_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
            sum_abs_train += (pred.detach().cpu() - y.detach().cpu()).abs().sum(dim=0)
        train_loss = total / max(1, count)
        train_mae = (sum_abs_train / max(1, count)).tolist()
        val_loss = None
        val_mae = None
        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            vcount = 0
            sum_abs_val = torch.zeros(target_dim, dtype=torch.float)
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
                    y = batch.y.view(pred.size(0), -1)
                    loss = torch.nn.functional.l1_loss(pred, y)
                    vtotal += loss.item() * y.size(0)
                    vcount += y.size(0)
                    sum_abs_val += (pred.detach().cpu() - y.detach().cpu()).abs().sum(dim=0)
            val_loss = vtotal / max(1, vcount)
            val_mae = (sum_abs_val / max(1, vcount)).tolist()
        history.append(train_loss if val_loss is None else val_loss)
        metric = val_loss if val_loss is not None else train_loss
        if metric < best_loss:
            best_loss = metric
            bad_epochs = 0
            cfg.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.save_dir / "schnet.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

        lr_now = float(opt.param_groups[0]["lr"])
        live_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "metric": float(metric),
                "best": float(best_loss),
                "lr": lr_now,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )
        msg = (
            f"[epoch {epoch + 1}] train={train_loss:.4f}"
            + (f" | val={val_loss:.4f}" if val_loss is not None else "")
            + f" | best={best_loss:.4f} | lr={lr_now:.2e}"
        )
        print(msg)
        _append_live(msg)
        if live_enabled:
            write_schnet_live_dashboard(
                live_path,
                title="SchNet Live Dashboard",
                epoch=epoch + 1,
                total_epochs=cfg.epochs,
                best_metric=float(best_loss),
                lr=lr_now,
                history=live_history,
                refresh_ms=live_refresh_ms,
                started_at=started_at,
                target_names=cfg.target_names,
                cli_lines=live_lines,
            )

    if live_server is not None and live_url:
        _append_live(f"Training done. Dashboard still reachable: {live_url}")
    return model, history


def load_schnet(path: Path, target_dim: int, cfg: Optional[SchNetConfig] = None) -> SchNetModel:
    cfg = cfg or SchNetConfig()
    model = _build_schnet(cfg, target_dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _build_schnet(cfg: SchNetConfig, target_dim: int) -> SchNetModel:
    return SchNetModel(cfg, target_dim)


def _resolve_device(device_spec: str | torch.device) -> torch.device:
    if isinstance(device_spec, torch.device):
        return device_spec
    if isinstance(device_spec, str) and device_spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)
