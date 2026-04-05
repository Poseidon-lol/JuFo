from __future__ import annotations

import sys
import re
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Sequence, TYPE_CHECKING

import pandas as pd
import torch

# Ensure the project root (with src/) is on sys.path
PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate project root containing src/")


from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.log import setup_logging

if TYPE_CHECKING:
    from src.models.jtvae_extended import JTVAE, JTVDataset
    from src.models.schnet_full import RealSchNetModel


def train_surrogate(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    logger.warning(
        "SchNet-only mode aktiv: 'train-surrogate' wird auf full SchNet umgeleitet."
    )
    cfg_path = str(getattr(args, "config", "") or "")
    if cfg_path.endswith("train_conf.yaml"):
        args.config = "configs/train_conf_3d_full.yaml"
    train_surrogate_3d_full(args)

def train_surrogate_3d(args: argparse.Namespace) -> None:
    """Backward-compatible alias for full SchNet training."""
    logger = logging.getLogger(__name__)
    logger.warning(
        "SchNet-only mode aktiv: 'train-surrogate-3d' wird auf full SchNet umgeleitet."
    )
    cfg_path = str(getattr(args, "config", "") or "")
    if cfg_path.endswith("train_conf_3d.yaml"):
        args.config = "configs/train_conf_3d_full.yaml"
    train_surrogate_3d_full(args)


def train_surrogate_3d_full(args: argparse.Namespace) -> None:
    """Train a full SchNet (PyG) surrogate on MolBlock geometries."""
    import pandas as pd
    import numpy as np
    from src.data.featurization_3d import dataframe_to_3d_dataset, qmsymex_xyz_dir_to_3d_dataset
    from src.models.schnet_full import RealSchNetConfig, train_schnet_full

    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg.training.device = args.device
    data_cfg = cfg.dataset
    target_columns = list(getattr(data_cfg, "target_columns", []))
    if not target_columns:
        raise ValueError("Config dataset.target_columns must list at least one property for surrogate training.")
    data_path = Path(getattr(data_cfg, "path"))
    data_source = str(getattr(data_cfg, "source", "") or "").strip().lower()
    if data_path.is_dir() or data_source in {"qmsymex", "qm_symex", "qmsymex_xyz"}:
        ds = qmsymex_xyz_dir_to_3d_dataset(
            data_path,
            target_cols=target_columns,
            transition_mode=str(getattr(data_cfg, "transition_mode", "best_f")),
            lambda_min=getattr(data_cfg, "lambda_min", None),
            lambda_max=getattr(data_cfg, "lambda_max", None),
            f_min=getattr(data_cfg, "f_min", None),
            charge=int(getattr(data_cfg, "charge", 0) or 0),
            max_files=getattr(data_cfg, "max_files", None),
            dedupe_smiles=bool(getattr(data_cfg, "dedupe_smiles", False)),
            progress_every=int(getattr(data_cfg, "progress_every", 2000) or 2000),
        )
    else:
        df = pd.read_csv(data_path)
        mol_col = getattr(data_cfg, "mol_column", "mol")
        smi_col = getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile"))
        ds = dataframe_to_3d_dataset(df, mol_col=mol_col, smiles_col=smi_col, target_cols=target_columns)
    if len(ds) == 0:
        raise ValueError("No valid 3D entries parsed from dataset; check mol_column/smiles_column.")
    # split
    val_fraction = float(getattr(data_cfg, "val_fraction", 0.1))
    seed = int(getattr(cfg.training, "seed", 1337))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    n_val = int(len(ds) * val_fraction)
    val_idx = set(perm[:n_val].tolist())
    train_ds = [ds[i] for i in perm if i not in val_idx]
    val_ds = [ds[i] for i in val_idx] if n_val > 0 else None
    train_cfg = cfg.training
    model_cfg = cfg.model
    live_cfg = getattr(train_cfg, "live_dashboard", None)
    sch_cfg = RealSchNetConfig(
        hidden_channels=model_cfg.hidden_channels,
        num_filters=model_cfg.num_filters,
        num_interactions=model_cfg.num_interactions,
        num_gaussians=model_cfg.num_gaussians,
        cutoff=model_cfg.cutoff,
        readout=model_cfg.readout,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        batch_size=train_cfg.batch_size,
        num_workers=getattr(train_cfg, "num_workers", RealSchNetConfig.num_workers),
        pin_memory=getattr(train_cfg, "pin_memory", False),
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        scheduler_patience=getattr(train_cfg, "scheduler_patience", getattr(train_cfg, "patience", 10)),
        use_amp=bool(getattr(train_cfg, "use_amp", False)),
        grad_clip=float(getattr(train_cfg, "grad_clip", 0.0)),
        loss=getattr(train_cfg, "loss", "l1"),
        target_weights=getattr(train_cfg, "target_weights", None),
        head_dropout=getattr(model_cfg, "head_dropout", 0.0),
        interaction_dropout=getattr(model_cfg, "interaction_dropout", 0.0),
        device=getattr(train_cfg, "device", "cpu"),
        save_dir=Path(train_cfg.save_dir),
        target_names=target_columns,
        live_dashboard=bool(getattr(live_cfg, "enabled", False)) if live_cfg is not None else False,
        live_dashboard_path=(
            Path(getattr(live_cfg, "path"))
            if (live_cfg is not None and getattr(live_cfg, "path", None))
            else None
        ),
        live_dashboard_refresh_ms=int(getattr(live_cfg, "refresh_ms", 900)) if live_cfg is not None else 900,
        live_dashboard_local_view=bool(getattr(live_cfg, "local_view_enabled", False)) if live_cfg is not None else False,
        live_dashboard_host=getattr(live_cfg, "local_view_host", "127.0.0.1") if live_cfg is not None else "127.0.0.1",
        live_dashboard_port=int(getattr(live_cfg, "local_view_port", 0)) if live_cfg is not None else 0,
        live_dashboard_open_browser=bool(getattr(live_cfg, "open_browser", True)) if live_cfg is not None else True,
    )
    model, hist = train_schnet_full(train_ds, val_ds, target_dim=len(target_columns), config=sch_cfg)
    print(f"Full SchNet surrogate trained; best model saved to {sch_cfg.save_dir/'schnet_full.pt'}")

def _load_jtvae_from_ckpt(ckpt: Path, fragment_vocab_size: int, cond_dim: int) -> JTVAE:
    from src.models.jtvae_extended import JTVAE

    raw = torch.load(ckpt, map_location="cpu")

    # Support both plain state_dict checkpoints and training bundles.
    if isinstance(raw, dict):
        for container_key in ("state_dict", "model_state_dict", "generator_state_dict", "model"):
            nested = raw.get(container_key)
            if isinstance(nested, dict):
                raw = nested
                break
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format at {ckpt}: {type(raw)!r}")

    def _normalize_keys(sd: dict) -> dict:
        prefixes = ("module.", "_orig_mod.", "model.", "generator.")
        normalized = {}
        for key, value in sd.items():
            if not isinstance(key, str):
                continue
            k = key.replace("._orig_mod.", ".")
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if k.startswith(prefix):
                        k = k[len(prefix) :]
                        changed = True
            normalized[k] = value
        return normalized

    state = _normalize_keys(raw)

    def _resolve_key(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in state:
                return cand
        for cand in candidates:
            for key in state.keys():
                if key.endswith(cand):
                    return key
        return None

    tree_proj_key = _resolve_key("encoder.tree_encoder.input_proj.weight", "tree_encoder.input_proj.weight")
    graph_proj_key = _resolve_key("encoder.graph_encoder.input_proj.weight", "graph_encoder.input_proj.weight")
    fc_mu_key = _resolve_key("encoder.fc_mu.weight", "fc_mu.weight")
    if tree_proj_key is None or graph_proj_key is None or fc_mu_key is None:
        sample_keys = ", ".join(list(state.keys())[:20])
        raise KeyError(
            "JT-VAE checkpoint keys not recognized. "
            f"Missing one of tree/graph/fc_mu keys in {ckpt}. Sample keys: {sample_keys}"
        )

    hidden_dim = state[tree_proj_key].shape[0]
    node_feat_dim = state[tree_proj_key].shape[1]
    graph_feat_dim = state[graph_proj_key].shape[1]
    z_dim = state[fc_mu_key].shape[0]
    # Infer cond_dim in checkpoint for compatibility logging.
    cond_dim_from_state = None
    property_head_key = _resolve_key("property_head.2.weight")
    if property_head_key is not None:
        cond_dim_from_state = state[property_head_key].shape[0]
    else:
        fused_dim = state[fc_mu_key].shape[1]
        cond_dim_from_state = max(fused_dim - 2 * hidden_dim, 0)
    requested_cond_dim = int(cond_dim)
    if cond_dim_from_state is not None and int(cond_dim_from_state) != requested_cond_dim:
        logging.getLogger(__name__).info(
            "JT-VAE checkpoint cond_dim=%s, requested cond_dim=%s. "
            "Keeping requested cond_dim and loading only shape-compatible weights.",
            int(cond_dim_from_state),
            requested_cond_dim,
        )
    positional_key = None
    for key in state.keys():
        if key.endswith("decoder.positional"):
            positional_key = key
            break
    max_tree_nodes = state[positional_key].shape[0] if positional_key else 12
    encoder_layer_indices = set()
    pattern = re.compile(r"(?:^|.*\.)tree_encoder\.layers\.(\d+)\.")
    for key in state.keys():
        match = pattern.match(key)
        if match:
            encoder_layer_indices.add(int(match.group(1)))
    encoder_layers = (max(encoder_layer_indices) + 1) if encoder_layer_indices else 3
    model = JTVAE(
        tree_feat_dim=node_feat_dim,
        graph_feat_dim=graph_feat_dim,
        fragment_vocab_size=fragment_vocab_size,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        cond_dim=requested_cond_dim,
        max_tree_nodes=max_tree_nodes,
        encoder_layers=encoder_layers,
    )
    model_state = model.state_dict()
    loadable_state = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        loadable_state[key] = value

    missing_after_load, unexpected_after_load = model.load_state_dict(loadable_state, strict=False)
    logger = logging.getLogger(__name__)
    if skipped_shape:
        preview = ", ".join(k for k, _, _ in skipped_shape[:6])
        logger.warning(
            "Skipped %d JT-VAE checkpoint tensors due to shape mismatch (likely cond_dim change): %s%s",
            len(skipped_shape),
            preview,
            "..." if len(skipped_shape) > 6 else "",
        )
    if skipped_missing:
        logger.info(
            "Ignored %d checkpoint tensors not present in current JT-VAE graph.",
            len(skipped_missing),
        )
    if unexpected_after_load:
        logger.info(
            "JT-VAE load returned %d unexpected tensor keys after filtering.",
            len(unexpected_after_load),
        )
    if missing_after_load:
        logger.info(
            "JT-VAE initialized %d tensors from scratch (no compatible checkpoint tensor).",
            len(missing_after_load),
        )
    return model


def train_generator(args: argparse.Namespace) -> None:
    from src.data.jt_preprocess import JTPreprocessConfig, build_fragment_vocab, prepare_jtvae_examples
    from src.models.jtvae_extended import JTVAE, JTVDataset, train_jtvae

    cfg = load_config(args.config)
    if args.device:
        cfg.training.device = args.device
    if args.amp is not None:
        cfg.training.use_amp = bool(args.amp)
    if args.compile is not None:
        cfg.training.compile = bool(args.compile)
    if args.compile_mode:
        cfg.training.compile_mode = args.compile_mode
    if args.compile_fullgraph is not None:
        cfg.training.compile_fullgraph = bool(args.compile_fullgraph)
    logger = logging.getLogger(__name__)
    data_cfg = cfg.dataset
    logger.info("Loading JT-VAE dataset from %s", data_cfg.path)
    df = pd.read_csv(data_cfg.path)
    logger.info("Loaded %d molecules for JT-VAE training.", len(df))
    if "smiles" not in df.columns:
        if "smile" in df.columns:
            df = df.rename(columns={"smile": "smiles"})
            logger.info("Renamed 'smile' column to 'smiles' for JT-VAE training.")
        else:
            raise KeyError("JT-VAE dataset must contain a 'smiles' (or 'smile') column.")
    min_count = getattr(data_cfg, "fragment_min_count", 1)
    fragment_method = getattr(data_cfg, "fragment_method", "ring_scaffold")
    min_fragment_heavy_atoms = getattr(data_cfg, "min_fragment_heavy_atoms", 1)
    frag2idx, idx2frag = build_fragment_vocab(
        df["smiles"],
        min_count=min_count,
        fragment_method=fragment_method,
        min_fragment_heavy_atoms=min_fragment_heavy_atoms,
    )
    if len(frag2idx) == 0 and min_count > 1:
        logger.warning(
            "Fragment vocabulary is empty with min_count=%s; lowering to 1 and rebuilding.", min_count
        )
        min_count = 1
        frag2idx, idx2frag = build_fragment_vocab(
            df["smiles"],
            min_count=min_count,
            fragment_method=fragment_method,
            min_fragment_heavy_atoms=min_fragment_heavy_atoms,
        )
    if len(frag2idx) == 0:
        raise ValueError(
            "Fragment vocabulary is empty. Provide more data or lower dataset.fragment_min_count."
        )
    logger.info(
        "Fragment vocabulary size: %d (min_count=%s, method=%s, min_heavy=%s)",
        len(frag2idx),
        min_count,
        fragment_method,
        min_fragment_heavy_atoms,
    )
    cond_cols = cfg.dataset.target_columns
    jt_config = JTPreprocessConfig(
        max_fragments=cfg.dataset.max_fragments,
        condition_columns=cond_cols,
        fragment_method=fragment_method,
        min_fragment_heavy_atoms=min_fragment_heavy_atoms,
    )
    max_heavy_atoms = getattr(data_cfg, "max_heavy_atoms", 80)
    logger.info(
        "Preparing JT-VAE examples (max_fragments=%s, condition_columns=%s, max_heavy_atoms=%s)...",
        cfg.dataset.max_fragments,
        cond_cols,
        max_heavy_atoms,
    )
    examples = prepare_jtvae_examples(
        df,
        frag2idx,
        config=jt_config,
        max_heavy_atoms=max_heavy_atoms,
    )
    logger.info("Prepared %d JT-VAE examples.", len(examples))
    dataset = JTVDataset(examples)
    logger.info("Constructed JTVDataset with %d entries.", len(dataset))
    cond_dim_value = len(cond_cols) if getattr(cfg.model, "cond_dim", None) is None else cfg.model.cond_dim
    if cond_dim_value != len(cond_cols):
        logger.warning(
            "Configured cond_dim (%s) differs from number of condition columns (%s). "
            "Using %s for property head.",
            cond_dim_value,
            len(cond_cols),
            len(cond_cols),
        )
        cond_dim_value = len(cond_cols)
    tree_feat_dim = examples[0]["tree_x"].size(1)
    graph_feat_dim = examples[0]["graph_x"].size(1)
    logger.info("Tree feature dim: %s | Graph feature dim: %s", tree_feat_dim, graph_feat_dim)
    encoder_layers = int(getattr(cfg.model, "encoder_layers", 3))
    resume_ckpt = getattr(cfg.training, "resume_ckpt", None)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"JT-VAE resume checkpoint not found: {ckpt_path}")
        logger.info("Loading JT-VAE checkpoint from %s", ckpt_path)
        model = _load_jtvae_from_ckpt(ckpt_path, len(frag2idx), cond_dim_value)
        logger.info("Loaded JT-VAE from checkpoint (encoder depth inferred from checkpoint).")
    else:
        model = JTVAE(
            tree_feat_dim=tree_feat_dim,
            graph_feat_dim=graph_feat_dim,
            fragment_vocab_size=len(frag2idx),
            z_dim=cfg.model.z_dim,
            hidden_dim=cfg.model.hidden_dim,
            cond_dim=cond_dim_value,
            max_tree_nodes=cfg.dataset.max_fragments,
            encoder_layers=encoder_layers,
        )
        logger.info("Initialized JT-VAE with encoder_layers=%s.", encoder_layers)
    kl_weight = getattr(cfg.training, "kl_weight", 0.5)
    property_weight = getattr(cfg.training, "property_loss_weight", 0.0)
    adjacency_weight = getattr(cfg.training, "adjacency_loss_weight", 1.0)
    device_override = getattr(cfg.training, "device", None)
    use_amp = bool(getattr(cfg.training, "use_amp", False))
    compile_flag = bool(getattr(cfg.training, "compile", False))
    compile_mode = getattr(cfg.training, "compile_mode", "default")
    compile_fullgraph = bool(getattr(cfg.training, "compile_fullgraph", False))
    max_grad_norm = getattr(cfg.training, "max_grad_norm", None)
    scheduler_patience = int(getattr(cfg.training, "scheduler_patience", 10))
    scheduler_factor = float(getattr(cfg.training, "scheduler_factor", 0.5))
    live_cfg = getattr(cfg.training, "live_decode", None)
    live_decode_kwargs = {}
    if live_cfg is not None:
        live_decode_kwargs = {
            "live_decode": bool(getattr(live_cfg, "enabled", False)),
            "live_decode_path": getattr(live_cfg, "path", None),
            "live_decode_refresh_ms": int(getattr(live_cfg, "refresh_ms", 1000)),
            "live_decode_step_delay": float(getattr(live_cfg, "step_delay", 0.0)),
            "live_decode_topk": int(getattr(live_cfg, "topk", 5)),
            "live_decode_max_steps": getattr(live_cfg, "max_steps", None),
            "live_decode_temperature": float(getattr(live_cfg, "temperature", 1.0)),
            "live_decode_every_n_epochs": int(getattr(live_cfg, "every_n_epochs", 1)),
            "live_decode_adjacency_threshold": float(
                getattr(live_cfg, "adjacency_threshold", 0.5)
            ),
            "live_decode_exhibition_mode": bool(getattr(live_cfg, "exhibition_mode", False)),
            "live_decode_local_view": bool(getattr(live_cfg, "local_view_enabled", False)),
            "live_decode_host": getattr(live_cfg, "local_view_host", "127.0.0.1"),
            "live_decode_port": int(getattr(live_cfg, "local_view_port", 0)),
            "live_decode_open_browser": bool(getattr(live_cfg, "open_browser", True)),
        }
    resume_epoch = getattr(cfg.training, "resume_epoch", None)
    if resume_epoch is None and resume_ckpt:
        match = re.search(r"epoch_(\d+)", str(resume_ckpt))
        if match:
            resume_epoch = int(match.group(1))
    start_epoch = 1
    if resume_epoch is not None:
        start_epoch = max(1, int(resume_epoch) + 1)
    logger.info(
        "Starting JT-VAE training: epochs=%s start_epoch=%s batch_size=%s lr=%s (kl=%s, prop=%s, adj=%s) device=%s amp=%s compile=%s max_grad_norm=%s",
        cfg.training.epochs,
        start_epoch,
        cfg.training.batch_size,
        cfg.training.lr,
        kl_weight,
        property_weight,
        adjacency_weight,
        device_override or "auto",
        use_amp,
        compile_flag,
        max_grad_norm,
    )
    train_jtvae(
        model,
        dataset,
        frag2idx,
        device=device_override,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        save_dir=cfg.save_dir,
        kl_weight=kl_weight,
        property_weight=property_weight,
        adj_weight=adjacency_weight,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        use_amp=use_amp,
        compile=compile_flag,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        max_grad_norm=max_grad_norm,
        start_epoch=start_epoch,
        cond_stats=jt_config.condition_stats,
        **live_decode_kwargs,
    )
    vocab_path = Path(cfg.save_dir) / "fragment_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(idx2frag, f, indent=2)
    if jt_config.condition_stats:
        stats_path = Path(cfg.save_dir) / "condition_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(jt_config.condition_stats, f, indent=2)
    print(f"Generator checkpoints and vocab saved to {cfg.save_dir}")


def train_generator_3d(args: argparse.Namespace) -> None:
    """Train the 3D VAE generator on opv_db MolBlocks."""
    import pandas as pd
    from src.data.featurization_3d_gen import build_gen3d_dataset
    from src.models.vae3d import VAE3DConfig, train_vae3d

    cfg = load_config(args.config)
    data_cfg = cfg.dataset
    train_cfg = cfg.training
    model_cfg = cfg.model

    df = pd.read_csv(data_cfg.path)
    max_atoms = int(getattr(data_cfg, "max_atoms", 100))
    dataset = build_gen3d_dataset(
        df,
        mol_col=getattr(data_cfg, "mol_column", "mol"),
        smiles_col=getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile")),
        max_atoms=max_atoms,
    )
    device = getattr(train_cfg, "device", "cpu")
    config = VAE3DConfig(
        max_atoms=max_atoms,
        z_dim=model_cfg.z_dim,
        hidden_dim=model_cfg.hidden_dim,
        lr=train_cfg.lr,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        device=device,
        save_path=train_cfg.save_path,
    )
    train_vae3d(dataset, config)
    print(f"3D generator trained and saved to {config.save_path}")


def run_active_loop(args: argparse.Namespace) -> None:
    from src.active_learn.acq import AcquisitionConfig
    from src.active_learn.loop import ActiveLearningLoop, LoopConfig
    from src.active_learn.objectives import (
        apply_objective_profile,
        load_objective_profile,
        normalize_objective_mode,
    )
    from src.active_learn.sched import SchedulerConfig
    from src.models.jtvae_extended import JTVAE
    import pandas as pd
    from torch_geometric.loader import DataLoader

    cfg = load_config(args.config)
    objective_mode_raw = getattr(args, "objective_mode", None)
    if objective_mode_raw is None:
        objective_mode_raw = getattr(cfg.loop, "objective_mode", None)
    objective_profile_raw = getattr(args, "objective_profile", None)
    if objective_profile_raw is None:
        objective_profile_raw = getattr(cfg.loop, "objective_profile_path", None)
    if objective_mode_raw is not None or objective_profile_raw is not None:
        objective_mode = normalize_objective_mode(objective_mode_raw or "red")
        objective_profile, resolved_objective_path = load_objective_profile(
            objective_mode,
            objective_profile_raw,
        )
        applied = apply_objective_profile(
            cfg,
            objective_mode,
            objective_profile,
            resolved_objective_path,
        )
        logging.getLogger(__name__).info(
            "Applied objective profile mode=%s from %s (sections=%s).",
            objective_mode,
            resolved_objective_path,
            ",".join(sorted(k for k in applied.keys() if k not in {"mode", "profile_path"})) or "none",
        )
    acq_cfg = AcquisitionConfig(**cfg.acquisition)
    sched_cfg = SchedulerConfig(**cfg.scheduler)
    loop_device_cfg = getattr(cfg.loop, "device", None)
    surrogate_device = getattr(cfg.loop, "surrogate_device", loop_device_cfg)
    generator_device = getattr(cfg.loop, "generator_device", loop_device_cfg)
    if getattr(args, "device", None):
        surrogate_device = generator_device = args.device
    if getattr(args, "surrogate_device", None):
        surrogate_device = args.surrogate_device
    if getattr(args, "generator_device", None):
        generator_device = args.generator_device

    red_score_columns_cfg = getattr(cfg.loop, "red_score_columns", None)
    if red_score_columns_cfg is None:
        red_score_columns_cfg = getattr(cfg.loop, "objective_score_columns", LoopConfig.red_score_columns)
    red_score_targets_cfg = getattr(cfg.loop, "red_score_targets", None)
    if red_score_targets_cfg is None:
        red_score_targets_cfg = getattr(cfg.loop, "objective_score_targets", LoopConfig.red_score_targets)
    red_score_tolerances_cfg = getattr(cfg.loop, "red_score_tolerances", None)
    if red_score_tolerances_cfg is None:
        red_score_tolerances_cfg = getattr(cfg.loop, "objective_score_tolerances", LoopConfig.red_score_tolerances)
    red_score_weights_cfg = getattr(cfg.loop, "red_score_weights", None)
    if red_score_weights_cfg is None:
        red_score_weights_cfg = getattr(cfg.loop, "objective_score_weights", LoopConfig.red_score_weights)
    red_score_missing_penalty_cfg = getattr(cfg.loop, "red_score_missing_penalty", None)
    if red_score_missing_penalty_cfg is None:
        red_score_missing_penalty_cfg = getattr(
            cfg.loop,
            "objective_score_missing_penalty",
            LoopConfig.red_score_missing_penalty,
        )
    red_score_pass_threshold_cfg = getattr(cfg.loop, "red_score_pass_threshold", None)
    if red_score_pass_threshold_cfg is None:
        red_score_pass_threshold_cfg = getattr(
            cfg.loop,
            "objective_score_pass_threshold",
            LoopConfig.red_score_pass_threshold,
        )
    red_score_require_qc_success_cfg = getattr(cfg.loop, "red_score_require_qc_success", None)
    if red_score_require_qc_success_cfg is None:
        red_score_require_qc_success_cfg = getattr(
            cfg.loop,
            "objective_score_require_qc_success",
            LoopConfig.red_score_require_qc_success,
        )

    objective_mode_cfg = str(getattr(cfg.loop, "objective_mode", "red") or "red")
    objective_profile_path_cfg = getattr(cfg.loop, "objective_profile_path", None)
    rl_cfg_raw = getattr(cfg.loop, "rl", {}) or {}
    rl_cfg = dict(rl_cfg_raw)
    if getattr(args, "rl_enabled", None) is not None:
        rl_cfg["enabled"] = bool(args.rl_enabled)
    if getattr(args, "rl_every_n_iterations", None) is not None:
        rl_cfg["every_n_iterations"] = int(args.rl_every_n_iterations)
    if getattr(args, "rl_steps_per_update", None) is not None:
        rl_cfg["steps_per_update"] = int(args.rl_steps_per_update)
    if getattr(args, "rl_batch_size", None) is not None:
        rl_cfg["batch_size"] = int(args.rl_batch_size)
    if getattr(args, "rl_lr", None) is not None:
        rl_cfg["lr"] = float(args.rl_lr)
    if getattr(args, "rl_algorithm", None) is not None:
        rl_cfg["algorithm"] = str(args.rl_algorithm)
    if getattr(args, "rl_entropy_weight", None) is not None:
        rl_cfg["entropy_weight"] = float(args.rl_entropy_weight)
    if getattr(args, "rl_baseline_momentum", None) is not None:
        rl_cfg["baseline_momentum"] = float(args.rl_baseline_momentum)
    if getattr(args, "rl_reward_clip", None) is not None:
        rl_cfg["reward_clip"] = float(args.rl_reward_clip)
    if getattr(args, "rl_normalize_advantage", None) is not None:
        rl_cfg["normalize_advantage"] = bool(args.rl_normalize_advantage)
    if getattr(args, "rl_use_qc_top_k", None) is not None:
        rl_cfg["use_qc_top_k"] = bool(args.rl_use_qc_top_k)
    if getattr(args, "rl_qc_top_k", None) is not None:
        rl_cfg["qc_top_k"] = int(args.rl_qc_top_k)
    if getattr(args, "rl_warmup_iterations", None) is not None:
        rl_cfg["warmup_iterations"] = int(args.rl_warmup_iterations)
    if getattr(args, "rl_max_grad_norm", None) is not None:
        rl_cfg["max_grad_norm"] = float(args.rl_max_grad_norm)
    if getattr(args, "rl_checkpoint_every", None) is not None:
        rl_cfg["checkpoint_every"] = int(args.rl_checkpoint_every)
    if getattr(args, "rl_value_loss_weight", None) is not None:
        rl_cfg["value_loss_weight"] = float(args.rl_value_loss_weight)
    if getattr(args, "rl_ppo_clip_ratio", None) is not None:
        rl_cfg["ppo_clip_ratio"] = float(args.rl_ppo_clip_ratio)
    if getattr(args, "rl_ppo_epochs", None) is not None:
        rl_cfg["ppo_epochs"] = int(args.rl_ppo_epochs)
    if getattr(args, "rl_ppo_minibatch_size", None) is not None:
        rl_cfg["ppo_minibatch_size"] = int(args.rl_ppo_minibatch_size)
    if getattr(args, "rl_ppo_target_kl", None) is not None:
        rl_cfg["ppo_target_kl"] = float(args.rl_ppo_target_kl)
    rl_baseline_momentum_cfg = rl_cfg.get("baseline_momentum", LoopConfig.rl_baseline_momentum)
    rl_entropy_weight_cfg = rl_cfg.get("entropy_weight", LoopConfig.rl_entropy_weight)
    rl_lr_cfg = rl_cfg.get("lr", LoopConfig.rl_lr)
    rl_reward_clip_cfg = rl_cfg.get("reward_clip", LoopConfig.rl_reward_clip)

    loop_cfg = LoopConfig(
        batch_size=cfg.loop.batch_size,
        acquisition=acq_cfg,
        scheduler=sched_cfg,
        target_columns=tuple(cfg.loop.target_columns),
        maximise=tuple(cfg.loop.maximise),
        generator_samples=cfg.loop.generator_samples,
        generator_attempts=int(getattr(cfg.loop, "generator_attempts", LoopConfig.generator_attempts)),
        seed=(args.seed if getattr(args, "seed", None) is not None else getattr(cfg.loop, "seed", None)),
        results_dir=Path(cfg.loop.results_dir),
        assemble=dict(getattr(cfg, "assemble", {})),
        diversity_threshold=float(getattr(cfg.loop, "diversity_threshold", 0.0)),
        diversity_metric=getattr(cfg.loop, "diversity_metric", "tanimoto"),
        generator_refresh=dict(getattr(cfg.loop, "generator_refresh", {})),
        property_aliases=dict(getattr(cfg.loop, "property_aliases", {})),
        rl_enabled=bool(rl_cfg.get("enabled", LoopConfig.rl_enabled)),
        rl_every_n_iterations=max(
            1,
            int(rl_cfg.get("every_n_iterations", LoopConfig.rl_every_n_iterations) or 1),
        ),
        rl_steps_per_update=max(
            1,
            int(rl_cfg.get("steps_per_update", LoopConfig.rl_steps_per_update) or 1),
        ),
        rl_batch_size=max(
            1,
            int(rl_cfg.get("batch_size", LoopConfig.rl_batch_size) or 1),
        ),
        rl_lr=float(LoopConfig.rl_lr if rl_lr_cfg is None else rl_lr_cfg),
        rl_algorithm=str(rl_cfg.get("algorithm", LoopConfig.rl_algorithm) or LoopConfig.rl_algorithm),
        rl_entropy_weight=float(
            LoopConfig.rl_entropy_weight if rl_entropy_weight_cfg is None else rl_entropy_weight_cfg
        ),
        rl_baseline_momentum=float(
            LoopConfig.rl_baseline_momentum
            if rl_baseline_momentum_cfg is None
            else rl_baseline_momentum_cfg
        ),
        rl_reward_clip=(
            None
            if rl_reward_clip_cfg is None
            else float(rl_reward_clip_cfg)
        ),
        rl_normalize_advantage=bool(
            rl_cfg.get("normalize_advantage", LoopConfig.rl_normalize_advantage)
        ),
        rl_use_qc_top_k=bool(rl_cfg.get("use_qc_top_k", LoopConfig.rl_use_qc_top_k)),
        rl_qc_top_k=max(0, int(rl_cfg.get("qc_top_k", LoopConfig.rl_qc_top_k) or 0)),
        rl_warmup_iterations=max(
            0,
            int(rl_cfg.get("warmup_iterations", LoopConfig.rl_warmup_iterations) or 0),
        ),
        rl_max_grad_norm=(
            None
            if rl_cfg.get("max_grad_norm", LoopConfig.rl_max_grad_norm) is None
            else float(rl_cfg.get("max_grad_norm", LoopConfig.rl_max_grad_norm))
        ),
        rl_checkpoint_every=max(
            1,
            int(rl_cfg.get("checkpoint_every", LoopConfig.rl_checkpoint_every) or 1),
        ),
        rl_value_loss_weight=float(
            rl_cfg.get("value_loss_weight", LoopConfig.rl_value_loss_weight)
            if rl_cfg.get("value_loss_weight", LoopConfig.rl_value_loss_weight) is not None
            else LoopConfig.rl_value_loss_weight
        ),
        rl_actor_lr=(
            None
            if rl_cfg.get("actor_lr", LoopConfig.rl_actor_lr) is None
            else float(rl_cfg.get("actor_lr", LoopConfig.rl_actor_lr))
        ),
        rl_critic_lr=(
            None
            if rl_cfg.get("critic_lr", LoopConfig.rl_critic_lr) is None
            else float(rl_cfg.get("critic_lr", LoopConfig.rl_critic_lr))
        ),
        rl_anchor_weight=float(
            rl_cfg.get("anchor_weight", LoopConfig.rl_anchor_weight)
            if rl_cfg.get("anchor_weight", LoopConfig.rl_anchor_weight) is not None
            else LoopConfig.rl_anchor_weight
        ),
        rl_reward_running_norm=bool(
            rl_cfg.get("reward_running_norm", LoopConfig.rl_reward_running_norm)
        ),
        rl_reward_norm_eps=float(
            rl_cfg.get("reward_norm_eps", LoopConfig.rl_reward_norm_eps)
            if rl_cfg.get("reward_norm_eps", LoopConfig.rl_reward_norm_eps) is not None
            else LoopConfig.rl_reward_norm_eps
        ),
        rl_reward_norm_clip=(
            None
            if rl_cfg.get("reward_norm_clip", LoopConfig.rl_reward_norm_clip) is None
            else float(rl_cfg.get("reward_norm_clip", LoopConfig.rl_reward_norm_clip))
        ),
        rl_entropy_weight_start=(
            None
            if rl_cfg.get("entropy_weight_start", LoopConfig.rl_entropy_weight_start) is None
            else float(rl_cfg.get("entropy_weight_start", LoopConfig.rl_entropy_weight_start))
        ),
        rl_entropy_weight_end=(
            None
            if rl_cfg.get("entropy_weight_end", LoopConfig.rl_entropy_weight_end) is None
            else float(rl_cfg.get("entropy_weight_end", LoopConfig.rl_entropy_weight_end))
        ),
        rl_entropy_decay_updates=max(
            1,
            int(rl_cfg.get("entropy_decay_updates", LoopConfig.rl_entropy_decay_updates) or 1),
        ),
        rl_ppo_clip_ratio=float(
            rl_cfg.get("ppo_clip_ratio", LoopConfig.rl_ppo_clip_ratio)
            if rl_cfg.get("ppo_clip_ratio", LoopConfig.rl_ppo_clip_ratio) is not None
            else LoopConfig.rl_ppo_clip_ratio
        ),
        rl_ppo_epochs=max(
            1,
            int(rl_cfg.get("ppo_epochs", LoopConfig.rl_ppo_epochs) or 1),
        ),
        rl_ppo_minibatch_size=(
            None
            if rl_cfg.get("ppo_minibatch_size", LoopConfig.rl_ppo_minibatch_size) is None
            else max(1, int(rl_cfg.get("ppo_minibatch_size", LoopConfig.rl_ppo_minibatch_size)))
        ),
        rl_ppo_target_kl=(
            None
            if rl_cfg.get("ppo_target_kl", LoopConfig.rl_ppo_target_kl) is None
            else float(rl_cfg.get("ppo_target_kl", LoopConfig.rl_ppo_target_kl))
        ),
        rl_ppo_value_clip_range=(
            None
            if rl_cfg.get("ppo_value_clip_range", LoopConfig.rl_ppo_value_clip_range) is None
            else float(rl_cfg.get("ppo_value_clip_range", LoopConfig.rl_ppo_value_clip_range))
        ),
        rl_ppo_adaptive_kl=bool(
            rl_cfg.get("ppo_adaptive_kl", LoopConfig.rl_ppo_adaptive_kl)
        ),
        rl_ppo_adaptive_kl_high_mult=float(
            rl_cfg.get("ppo_adaptive_kl_high_mult", LoopConfig.rl_ppo_adaptive_kl_high_mult)
            if rl_cfg.get("ppo_adaptive_kl_high_mult", LoopConfig.rl_ppo_adaptive_kl_high_mult) is not None
            else LoopConfig.rl_ppo_adaptive_kl_high_mult
        ),
        rl_ppo_adaptive_kl_low_mult=float(
            rl_cfg.get("ppo_adaptive_kl_low_mult", LoopConfig.rl_ppo_adaptive_kl_low_mult)
            if rl_cfg.get("ppo_adaptive_kl_low_mult", LoopConfig.rl_ppo_adaptive_kl_low_mult) is not None
            else LoopConfig.rl_ppo_adaptive_kl_low_mult
        ),
        rl_ppo_lr_down_factor=float(
            rl_cfg.get("ppo_lr_down_factor", LoopConfig.rl_ppo_lr_down_factor)
            if rl_cfg.get("ppo_lr_down_factor", LoopConfig.rl_ppo_lr_down_factor) is not None
            else LoopConfig.rl_ppo_lr_down_factor
        ),
        rl_ppo_lr_up_factor=float(
            rl_cfg.get("ppo_lr_up_factor", LoopConfig.rl_ppo_lr_up_factor)
            if rl_cfg.get("ppo_lr_up_factor", LoopConfig.rl_ppo_lr_up_factor) is not None
            else LoopConfig.rl_ppo_lr_up_factor
        ),
        rl_ppo_clip_down_factor=float(
            rl_cfg.get("ppo_clip_down_factor", LoopConfig.rl_ppo_clip_down_factor)
            if rl_cfg.get("ppo_clip_down_factor", LoopConfig.rl_ppo_clip_down_factor) is not None
            else LoopConfig.rl_ppo_clip_down_factor
        ),
        rl_ppo_clip_up_factor=float(
            rl_cfg.get("ppo_clip_up_factor", LoopConfig.rl_ppo_clip_up_factor)
            if rl_cfg.get("ppo_clip_up_factor", LoopConfig.rl_ppo_clip_up_factor) is not None
            else LoopConfig.rl_ppo_clip_up_factor
        ),
        rl_ppo_actor_lr_min=(
            None
            if rl_cfg.get("ppo_actor_lr_min", LoopConfig.rl_ppo_actor_lr_min) is None
            else float(rl_cfg.get("ppo_actor_lr_min", LoopConfig.rl_ppo_actor_lr_min))
        ),
        rl_ppo_actor_lr_max=(
            None
            if rl_cfg.get("ppo_actor_lr_max", LoopConfig.rl_ppo_actor_lr_max) is None
            else float(rl_cfg.get("ppo_actor_lr_max", LoopConfig.rl_ppo_actor_lr_max))
        ),
        rl_ppo_clip_ratio_min=float(
            rl_cfg.get("ppo_clip_ratio_min", LoopConfig.rl_ppo_clip_ratio_min)
            if rl_cfg.get("ppo_clip_ratio_min", LoopConfig.rl_ppo_clip_ratio_min) is not None
            else LoopConfig.rl_ppo_clip_ratio_min
        ),
        rl_ppo_clip_ratio_max=float(
            rl_cfg.get("ppo_clip_ratio_max", LoopConfig.rl_ppo_clip_ratio_max)
            if rl_cfg.get("ppo_clip_ratio_max", LoopConfig.rl_ppo_clip_ratio_max) is not None
            else LoopConfig.rl_ppo_clip_ratio_max
        ),
        objective_mode=objective_mode_cfg,
        objective_profile_path=(
            str(objective_profile_path_cfg) if objective_profile_path_cfg is not None else None
        ),
        max_pool_eval=getattr(cfg.loop, "max_pool_eval", None),
        predict_batch_size=getattr(cfg.loop, "predict_batch_size", None),
        predict_mc_samples=getattr(cfg.loop, "predict_mc_samples", None),
        max_generated_heavy_atoms=getattr(cfg.loop, "max_generated_heavy_atoms", None),
        max_generated_smiles_len=getattr(cfg.loop, "max_generated_smiles_len", None),
        generated_smiles_len_factor=(
            None
            if getattr(cfg.loop, "generated_smiles_len_factor", LoopConfig.generated_smiles_len_factor) is None
            else float(getattr(cfg.loop, "generated_smiles_len_factor", LoopConfig.generated_smiles_len_factor))
        ),
        exclude_smiles_paths=tuple(getattr(cfg.loop, "exclude_smiles_paths", ())),
        min_pi_conjugated_fraction=getattr(cfg.loop, "min_pi_conjugated_fraction", None),
        require_conjugation=bool(getattr(cfg.loop, "require_conjugation", LoopConfig.require_conjugation)),
        min_conjugated_bonds=int(
            getattr(cfg.loop, "min_conjugated_bonds", LoopConfig.min_conjugated_bonds)
        ),
        min_alternating_conjugated_bonds=int(
            getattr(
                cfg.loop,
                "min_alternating_conjugated_bonds",
                LoopConfig.min_alternating_conjugated_bonds,
            )
        ),
        min_aromatic_rings=int(getattr(cfg.loop, "min_aromatic_rings", LoopConfig.min_aromatic_rings)),
        max_rotatable_bonds=getattr(cfg.loop, "max_rotatable_bonds", LoopConfig.max_rotatable_bonds),
        max_rotatable_bonds_conjugated=getattr(
            cfg.loop,
            "max_rotatable_bonds_conjugated",
            LoopConfig.max_rotatable_bonds_conjugated,
        ),
        max_branch_points=getattr(cfg.loop, "max_branch_points", None),
        max_branch_degree=getattr(cfg.loop, "max_branch_degree", None),
        max_charged_atoms=getattr(cfg.loop, "max_charged_atoms", None),
        property_filters=dict(getattr(cfg.loop, "property_filters", {})),
        qc_extra_properties=tuple(getattr(cfg.loop, "qc_extra_properties", ())),
        objective_gate_min_lambda_max_nm=getattr(cfg.loop, "objective_gate_min_lambda_max_nm", None),
        objective_gate_max_lambda_max_nm=getattr(cfg.loop, "objective_gate_max_lambda_max_nm", None),
        objective_gate_min_oscillator_strength=getattr(
            cfg.loop,
            "objective_gate_min_oscillator_strength",
            None,
        ),
        objective_gate_max_oscillator_strength=getattr(
            cfg.loop,
            "objective_gate_max_oscillator_strength",
            None,
        ),
        max_lambda_max_nm=getattr(cfg.loop, "max_lambda_max_nm", None),
        max_oscillator_strength=getattr(cfg.loop, "max_oscillator_strength", None),
        min_lambda_max_nm=getattr(cfg.loop, "min_lambda_max_nm", None),
        min_oscillator_strength=getattr(cfg.loop, "min_oscillator_strength", None),
        require_neutral=bool(getattr(cfg.loop, "require_neutral", True)),
        sa_score_max=getattr(cfg.loop, "sa_score_max", None),
        physchem_filters=dict(getattr(cfg.loop, "physchem_filters", {})),
        scaffold_unique=bool(getattr(cfg.loop, "scaffold_unique", False)),
        live_dashboard=dict(getattr(cfg.loop, "live_dashboard", {})),
        auto_relax_filters=bool(getattr(cfg.loop, "auto_relax_filters", LoopConfig.auto_relax_filters)),
        save_diagnostics=bool(getattr(cfg.loop, "save_diagnostics", False)),
        diagnostics_every=int(getattr(cfg.loop, "diagnostics_every", 0)),
        diagnostics_max_points=int(getattr(cfg.loop, "diagnostics_max_points", 12000)),
        optical_score_weight=float(getattr(cfg.loop, "optical_score_weight", LoopConfig.optical_score_weight) or 0.0),
        optical_target_columns=tuple(
            getattr(cfg.loop, "optical_target_columns", LoopConfig.optical_target_columns)
        ),
        optical_targets=(
            None
            if getattr(cfg.loop, "optical_targets", LoopConfig.optical_targets) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "optical_targets", LoopConfig.optical_targets)
            )
        ),
        optical_tolerances=(
            None
            if getattr(cfg.loop, "optical_tolerances", LoopConfig.optical_tolerances) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "optical_tolerances", LoopConfig.optical_tolerances)
            )
        ),
        optical_weights=(
            None
            if getattr(cfg.loop, "optical_weights", LoopConfig.optical_weights) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "optical_weights", LoopConfig.optical_weights)
            )
        ),
        optical_beta=float(getattr(cfg.loop, "optical_beta", LoopConfig.optical_beta) or 0.0),
        optical_weight_schedule=tuple(
            getattr(cfg.loop, "optical_weight_schedule", LoopConfig.optical_weight_schedule)
        ),
        optical_predict_batch_size=getattr(cfg.loop, "optical_predict_batch_size", None),
        optical_predict_mc_samples=getattr(cfg.loop, "optical_predict_mc_samples", None),
        min_labels_for_optical=int(getattr(cfg.loop, "min_labels_for_optical", 0) or 0),
        optical_retrain_every=int(getattr(cfg.loop, "optical_retrain_every", 0) or 0),
        optical_retrain_min_labels=int(
            getattr(cfg.loop, "optical_retrain_min_labels", LoopConfig.optical_retrain_min_labels)
            or LoopConfig.optical_retrain_min_labels
        ),
        optical_retrain_val_fraction=float(
            getattr(
                cfg.loop,
                "optical_retrain_val_fraction",
                LoopConfig.optical_retrain_val_fraction,
            )
            or LoopConfig.optical_retrain_val_fraction
        ),
        optical_retrain_on_success_only=bool(
            getattr(
                cfg.loop,
                "optical_retrain_on_success_only",
                LoopConfig.optical_retrain_on_success_only,
            )
        ),
        optical_incremental_path=getattr(
            cfg.loop,
            "optical_incremental_path",
            LoopConfig.optical_incremental_path,
        ),
        optical_incremental_dedupe_on=tuple(
            getattr(
                cfg.loop,
                "optical_incremental_dedupe_on",
                LoopConfig.optical_incremental_dedupe_on,
            )
        ),
        optical_incremental_require_all_targets=bool(
            getattr(
                cfg.loop,
                "optical_incremental_require_all_targets",
                LoopConfig.optical_incremental_require_all_targets,
            )
        ),
        oscillator_score_weight=float(
            getattr(cfg.loop, "oscillator_score_weight", LoopConfig.oscillator_score_weight) or 0.0
        ),
        oscillator_target_columns=tuple(
            getattr(cfg.loop, "oscillator_target_columns", LoopConfig.oscillator_target_columns)
        ),
        oscillator_targets=(
            None
            if getattr(cfg.loop, "oscillator_targets", LoopConfig.oscillator_targets) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "oscillator_targets", LoopConfig.oscillator_targets)
            )
        ),
        oscillator_tolerances=(
            None
            if getattr(cfg.loop, "oscillator_tolerances", LoopConfig.oscillator_tolerances) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "oscillator_tolerances", LoopConfig.oscillator_tolerances)
            )
        ),
        oscillator_weights=(
            None
            if getattr(cfg.loop, "oscillator_weights", LoopConfig.oscillator_weights) is None
            else tuple(
                float(v) for v in getattr(cfg.loop, "oscillator_weights", LoopConfig.oscillator_weights)
            )
        ),
        oscillator_beta=float(getattr(cfg.loop, "oscillator_beta", LoopConfig.oscillator_beta) or 0.0),
        oscillator_predict_batch_size=getattr(cfg.loop, "oscillator_predict_batch_size", None),
        oscillator_predict_mc_samples=getattr(cfg.loop, "oscillator_predict_mc_samples", None),
        min_labels_for_oscillator=int(getattr(cfg.loop, "min_labels_for_oscillator", 0) or 0),
        oscillator_retrain_every=int(getattr(cfg.loop, "oscillator_retrain_every", 0) or 0),
        oscillator_retrain_min_labels=int(
            getattr(cfg.loop, "oscillator_retrain_min_labels", LoopConfig.oscillator_retrain_min_labels)
            or LoopConfig.oscillator_retrain_min_labels
        ),
        oscillator_retrain_val_fraction=float(
            getattr(
                cfg.loop,
                "oscillator_retrain_val_fraction",
                LoopConfig.oscillator_retrain_val_fraction,
            )
            or LoopConfig.oscillator_retrain_val_fraction
        ),
        oscillator_retrain_on_success_only=bool(
            getattr(
                cfg.loop,
                "oscillator_retrain_on_success_only",
                LoopConfig.oscillator_retrain_on_success_only,
            )
        ),
        objective_score_columns=tuple(
            getattr(cfg.loop, "objective_score_columns", red_score_columns_cfg)
        ),
        objective_score_targets=(
            None
            if getattr(cfg.loop, "objective_score_targets", red_score_targets_cfg) is None
            else tuple(
                float(v)
                for v in getattr(cfg.loop, "objective_score_targets", red_score_targets_cfg)
            )
        ),
        objective_score_tolerances=(
            None
            if getattr(cfg.loop, "objective_score_tolerances", red_score_tolerances_cfg) is None
            else tuple(
                float(v)
                for v in getattr(cfg.loop, "objective_score_tolerances", red_score_tolerances_cfg)
            )
        ),
        objective_score_weights=(
            None
            if getattr(cfg.loop, "objective_score_weights", red_score_weights_cfg) is None
            else tuple(
                float(v)
                for v in getattr(cfg.loop, "objective_score_weights", red_score_weights_cfg)
            )
        ),
        objective_score_missing_penalty=float(
            red_score_missing_penalty_cfg
            if getattr(cfg.loop, "objective_score_missing_penalty", red_score_missing_penalty_cfg) is None
            else getattr(cfg.loop, "objective_score_missing_penalty", red_score_missing_penalty_cfg)
        ),
        objective_score_pass_threshold=getattr(
            cfg.loop,
            "objective_score_pass_threshold",
            red_score_pass_threshold_cfg,
        ),
        objective_score_require_qc_success=bool(
            getattr(
                cfg.loop,
                "objective_score_require_qc_success",
                red_score_require_qc_success_cfg,
            )
        ),
        red_score_columns=tuple(red_score_columns_cfg),
        red_score_targets=(
            None
            if red_score_targets_cfg is None
            else tuple(
                float(v) for v in red_score_targets_cfg
            )
        ),
        red_score_tolerances=(
            None
            if red_score_tolerances_cfg is None
            else tuple(
                float(v)
                for v in red_score_tolerances_cfg
            )
        ),
        red_score_weights=(
            None
            if red_score_weights_cfg is None
            else tuple(
                float(v) for v in red_score_weights_cfg
            )
        ),
        red_score_missing_penalty=float(
            LoopConfig.red_score_missing_penalty
            if red_score_missing_penalty_cfg is None
            else red_score_missing_penalty_cfg
        ),
        red_score_pass_threshold=red_score_pass_threshold_cfg,
        red_score_require_qc_success=bool(red_score_require_qc_success_cfg),
        early_stop_no_improve_iterations=int(
            getattr(
                cfg.loop,
                "early_stop_no_improve_iterations",
                LoopConfig.early_stop_no_improve_iterations,
            )
            or LoopConfig.early_stop_no_improve_iterations
        ),
        early_stop_min_delta=float(
            getattr(cfg.loop, "early_stop_min_delta", LoopConfig.early_stop_min_delta)
            or LoopConfig.early_stop_min_delta
        ),
        export_top_k=int(getattr(cfg.loop, "export_top_k", LoopConfig.export_top_k) or LoopConfig.export_top_k),
        export_sort_column=str(
            getattr(cfg.loop, "export_sort_column", LoopConfig.export_sort_column)
            or LoopConfig.export_sort_column
        ),
        export_require_qc_success=bool(
            getattr(cfg.loop, "export_require_qc_success", LoopConfig.export_require_qc_success)
        ),
        export_require_red_pass=bool(
            getattr(cfg.loop, "export_require_red_pass", LoopConfig.export_require_red_pass)
        ),
        export_top_candidates_path=getattr(
            cfg.loop,
            "export_top_candidates_path",
            LoopConfig.export_top_candidates_path,
        ),
    )

    labelled = pd.read_csv(cfg.data.labelled)
    pool = pd.read_csv(cfg.data.pool)

    def _ensure_smiles(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "smiles" not in df.columns:
            if "smile" in df.columns:
                df = df.rename(columns={"smile": "smiles"})
            else:
                raise KeyError(f"{name} dataframe must contain a 'smiles' column (or 'smile').")
        return df

    labelled = _ensure_smiles(labelled, "labelled")
    pool = _ensure_smiles(pool, "pool")

    surrogate_device_runtime = surrogate_device
    surrogate = None
    surrogate_path = Path(args.surrogate_dir)
    dft_job_defaults: Dict[str, object] = {}

    class SchNetSurrogateWrapper:
        def __init__(self, model, device: str, target_columns, train_params=None):
            import numpy as np

            if isinstance(model, (list, tuple)):
                self.models = list(model)
            else:
                self.models = [model]
            self.device = torch.device(device)
            self.target_columns = list(target_columns)
            self.batch_size = cfg.loop.batch_size
            self.is_schnet = True
            self.train_params = train_params or {}
            self.mc_samples_default = int(self.train_params.get("mc_samples", 0))
            self.predict_num_workers = max(0, int(getattr(cfg.loop, "predict_num_workers", 0) or 0))
            self.predict_pin_memory = bool(
                getattr(cfg.loop, "predict_pin_memory", self.device.type == "cuda")
            )
            self.predict_amp = bool(getattr(cfg.loop, "predict_amp", self.device.type == "cuda"))

            # move models to device
            moved = []
            for m in self.models:
                moved.append(m.to(self.device))
            self.models = moved
            if self.mc_samples_default <= 0 and self.models:
                base_cfg = getattr(self.models[0], "cfg", None)
                if base_cfg and (
                    getattr(base_cfg, "head_dropout", 0.0) > 0 or getattr(base_cfg, "interaction_dropout", 0.0) > 0
                ):
                    self.mc_samples_default = 5

        def predict(self, graphs, batch_size=None, mc_samples: int | None = None, **kwargs):
            import numpy as np

            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            loader_kwargs = {
                "batch_size": batch_size or self.batch_size,
                "shuffle": False,
                "num_workers": self.predict_num_workers,
                "pin_memory": self.predict_pin_memory,
            }
            if self.predict_num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4
            loader = DataLoader(graphs, **loader_kwargs)
            mc = mc_samples
            if mc is None:
                mc = self.mc_samples_default
            mc = max(1, int(mc))
            all_passes = []
            for model in self.models:
                is_training = model.training
                try:
                    model.train(mc > 1)
                    with torch.inference_mode():
                        for _ in range(mc):
                            preds = []
                            for batch in loader:
                                batch = batch.to(
                                    self.device,
                                    non_blocking=bool(self.predict_pin_memory and self.device.type == "cuda"),
                                )
                                if self.predict_amp and self.device.type == "cuda":
                                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                                        out = model(batch.z, batch.pos, getattr(batch, "batch", None))
                                else:
                                    out = model(batch.z, batch.pos, getattr(batch, "batch", None))
                                preds.append(out.detach().cpu())
                            all_passes.append(torch.cat(preds, dim=0))
                finally:
                    if not is_training:
                        model.eval()
            stacked = torch.stack(all_passes, dim=0)  # [n_passes, N, D]
            mean = stacked.mean(dim=0).numpy()
            std = stacked.std(dim=0, unbiased=True).numpy() if stacked.shape[0] > 1 else np.zeros_like(mean)
            return mean, std, None

        def fit(self, train_df, val_df=None):
            from src.data.featurization_3d import dataframe_to_3d_dataset
            from src.models.schnet_full import RealSchNetConfig, train_schnet_full

            target_cols = [c for c in self.target_columns if c in train_df.columns]
            ds_train = dataframe_to_3d_dataset(train_df, mol_col="mol", smiles_col="smiles", target_cols=target_cols)
            ds_val = None
            if val_df is not None and len(val_df) > 0:
                ds_val = dataframe_to_3d_dataset(val_df, mol_col="mol", smiles_col="smiles", target_cols=target_cols)
            if len(ds_train) == 0:
                logger.warning("SchNet retrain skipped: no valid 3D entries in training split.")
                return

            n_models = int(self.train_params.get("n_models", len(self.models)))
            base_cfg = None
            if hasattr(self.models[0], "cfg"):
                base_cfg = getattr(self.models[0], "cfg")

            new_models = []
            for idx in range(max(1, n_models)):
                params = {
                    "hidden_channels": getattr(base_cfg, "hidden_channels", 128) if base_cfg else 128,
                    "num_filters": getattr(base_cfg, "num_filters", 128) if base_cfg else 128,
                    "num_interactions": getattr(base_cfg, "num_interactions", 6) if base_cfg else 6,
                    "num_gaussians": getattr(base_cfg, "num_gaussians", 50) if base_cfg else 50,
                    "cutoff": getattr(base_cfg, "cutoff", 10.0) if base_cfg else 10.0,
                    "readout": getattr(base_cfg, "readout", "add") if base_cfg else "add",
                    "lr": self.train_params.get("lr", 1e-3),
                    "weight_decay": self.train_params.get("weight_decay", 0.0),
                    "batch_size": self.train_params.get("batch_size", 16),
                    "epochs": self.train_params.get("epochs", 30),
                    "patience": self.train_params.get("patience", 5),
                    "scheduler_patience": self.train_params.get(
                        "scheduler_patience", getattr(base_cfg, "scheduler_patience", 10)
                    ),
                    "use_amp": bool(self.train_params.get("use_amp", getattr(base_cfg, "use_amp", False))),
                    "grad_clip": float(self.train_params.get("grad_clip", getattr(base_cfg, "grad_clip", 0.0))),
                    "loss": self.train_params.get("loss", getattr(base_cfg, "loss", "l1")),
                    "target_weights": self.train_params.get("target_weights", getattr(base_cfg, "target_weights", None)),
                    "head_dropout": float(self.train_params.get("head_dropout", getattr(base_cfg, "head_dropout", 0.0))),
                    "interaction_dropout": float(self.train_params.get("interaction_dropout", getattr(base_cfg, "interaction_dropout", 0.0))),
                    "num_workers": int(self.train_params.get("num_workers", getattr(base_cfg, "num_workers", 0))),
                    "pin_memory": bool(self.train_params.get("pin_memory", getattr(base_cfg, "pin_memory", False))),
                    "device": str(self.device),
                    "save_dir": Path(self.train_params.get("save_dir", getattr(base_cfg, "save_dir", "models/surrogate_3d_full"))),
                }
                sch_cfg = RealSchNetConfig(**params)
                new_model, _ = train_schnet_full(
                    ds_train,
                    ds_val,
                    target_dim=len(target_cols),
                    config=sch_cfg,
                    save_path=Path(sch_cfg.save_dir) / f"schnet_full_member_{idx:02d}.pt",
                    seed=int(self.train_params.get("seed", 1337)) + idx,
                )
                new_models.append(new_model.to(self.device))
            self.models = new_models

    def _load_surrogate_runtime(
        ckpt_or_dir: Path,
        *,
        target_columns: Sequence[str],
    ):
        runtime_device = surrogate_device
        if not ckpt_or_dir.exists():
            raise FileNotFoundError(f"Surrogate path does not exist: {ckpt_or_dir}")
        from src.models.schnet_full import load_schnet_full

        if ckpt_or_dir.is_file():
            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            runtime_device = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            # SchNet-only: interpret every provided checkpoint as full SchNet.
            model = load_schnet_full(ckpt_or_dir, target_dim=len(target_columns))
            loaded_surrogate = SchNetSurrogateWrapper([model], runtime_device, target_columns)
            loaded_surrogate.is_schnet_full = True  # marker
            return loaded_surrogate, runtime_device

        # Directory path: require full SchNet artifacts.
        maybe_full = ckpt_or_dir / "schnet_full.pt"
        if maybe_full.exists():
            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            runtime_device = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            model = load_schnet_full(maybe_full, target_dim=len(target_columns))
            loaded_surrogate = SchNetSurrogateWrapper([model], runtime_device, target_columns)
            loaded_surrogate.is_schnet_full = True
            return loaded_surrogate, runtime_device

        # check for ensemble of full SchNet members
        members = sorted(ckpt_or_dir.glob("schnet_full_member_*.pt"))
        if members:
            device_spec = get_device(surrogate_device or loop_device_cfg or "cpu")
            runtime_device = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
            loaded = [load_schnet_full(m, target_dim=len(target_columns)) for m in members]
            loaded_surrogate = SchNetSurrogateWrapper(loaded, runtime_device, target_columns)
            loaded_surrogate.is_schnet_full = True
            return loaded_surrogate, runtime_device

        raise FileNotFoundError(
            "SchNet-only mode: surrogate directory must contain 'schnet_full.pt' "
            "or at least one 'schnet_full_member_*.pt' checkpoint. "
            f"Got: {ckpt_or_dir}"
        )

    surrogate, surrogate_device_runtime = _load_surrogate_runtime(
        surrogate_path,
        target_columns=tuple(loop_cfg.target_columns),
    )

    optical_surrogate = None
    optical_surrogate_path = getattr(args, "optical_surrogate_dir", None)
    if optical_surrogate_path is None:
        optical_surrogate_path = getattr(cfg.loop, "optical_surrogate_dir", None)
    if optical_surrogate_path:
        if not tuple(loop_cfg.optical_target_columns):
            raise ValueError(
                "loop.optical_target_columns must be non-empty when an optical surrogate is configured."
            )
        optical_surrogate, _ = _load_surrogate_runtime(
            Path(optical_surrogate_path),
            target_columns=tuple(loop_cfg.optical_target_columns),
        )
        logging.getLogger(__name__).info(
            "Loaded optical surrogate from %s with targets=%s",
            optical_surrogate_path,
            tuple(loop_cfg.optical_target_columns),
        )
    elif abs(float(getattr(loop_cfg, "optical_score_weight", 0.0) or 0.0)) > 1e-12:
        logging.getLogger(__name__).warning(
            "optical_score_weight=%.3f but no optical surrogate provided; optical score will be ignored.",
            float(getattr(loop_cfg, "optical_score_weight", 0.0) or 0.0),
        )

    oscillator_surrogate = None
    oscillator_surrogate_path = getattr(args, "oscillator_surrogate_dir", None)
    if oscillator_surrogate_path is None:
        oscillator_surrogate_path = getattr(cfg.loop, "oscillator_surrogate_dir", None)
    if oscillator_surrogate_path:
        if not tuple(loop_cfg.oscillator_target_columns):
            raise ValueError(
                "loop.oscillator_target_columns must be non-empty when an oscillator surrogate is configured."
            )
        oscillator_surrogate, _ = _load_surrogate_runtime(
            Path(oscillator_surrogate_path),
            target_columns=tuple(loop_cfg.oscillator_target_columns),
        )
        logging.getLogger(__name__).info(
            "Loaded oscillator surrogate from %s with targets=%s",
            oscillator_surrogate_path,
            tuple(loop_cfg.oscillator_target_columns),
        )
    elif abs(float(getattr(loop_cfg, "oscillator_score_weight", 0.0) or 0.0)) > 1e-12:
        logging.getLogger(__name__).warning(
            "oscillator_score_weight=%.3f but no oscillator surrogate provided; oscillator score will be ignored.",
            float(getattr(loop_cfg, "oscillator_score_weight", 0.0) or 0.0),
        )

    generator = None
    generator3d = None
    generator3d_template = None
    fragment_vocab: Optional[Dict[str, int]] = None
    generator_device_runtime: Optional[str] = None
    if args.generator_ckpt:
        ckpt_path = Path(args.generator_ckpt)
        configured_vocab = Path(cfg.data.fragment_vocab) if getattr(cfg.data, "fragment_vocab", None) else None
        adjacent_vocab = ckpt_path.parent / "fragment_vocab.json"
        vocab_candidates = []
        if adjacent_vocab.exists():
            vocab_candidates.append(adjacent_vocab)
        if configured_vocab is not None:
            vocab_candidates.append(configured_vocab)

        vocab_path = None
        for candidate in vocab_candidates:
            if candidate.exists():
                vocab_path = candidate
                break
        if vocab_path is None:
            raise FileNotFoundError(
                "No fragment vocab found for generator checkpoint. "
                f"Checked: {[str(p) for p in vocab_candidates]}"
            )
        if configured_vocab is not None and vocab_path != configured_vocab:
            logging.getLogger(__name__).info(
                "Using fragment vocab next to checkpoint (%s) instead of configured path (%s).",
                vocab_path,
                configured_vocab,
            )

        with vocab_path.open("r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        if not isinstance(raw_vocab, dict) or not raw_vocab:
            raise ValueError(f"Fragment vocab at {vocab_path} is empty or not a mapping.")

        def _intlike(x: object) -> bool:
            try:
                int(x)
                return True
            except Exception:
                return False

        sample_key, sample_val = next(iter(raw_vocab.items()))
        # Case A: frag -> idx mapping (values int-like)
        if _intlike(sample_val):
            fragment_vocab = {k: int(v) for k, v in raw_vocab.items()}
        # Case B: idx -> frag mapping (keys int-like, values strings)
        elif _intlike(sample_key):
            fragment_vocab = {v: int(k) for k, v in raw_vocab.items()}
            logging.getLogger(__name__).info(
                "Reversed fragment vocab idx->frag mapping from %s.", vocab_path
            )
        else:
            raise ValueError(f"Unrecognised fragment_vocab format in {vocab_path}")

        generator = _load_jtvae_from_ckpt(
            ckpt_path,
            len(fragment_vocab),
            cond_dim=len(loop_cfg.target_columns),
        )
        if generator_device:
            device_spec = get_device(generator_device)
            generator = generator.to(device_spec.target)
            generator_device_runtime = (
                f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
            )
        else:
            generator_device_runtime = None
    if getattr(args, "generator_3d_ckpt", None):
        from src.models.vae3d import VAE3D
        from src.data.featurization_3d_gen import build_gen3d_dataset
        data_cfg = cfg.data
        max_atoms = getattr(getattr(cfg, "generator3d", {}), "max_atoms", 100)
        gen3d_ckpt = Path(args.generator_3d_ckpt)
        generator3d = VAE3D(max_atoms=max_atoms)
        generator3d.load_state_dict(torch.load(gen3d_ckpt, map_location="cpu"))
        if generator_device:
            device_spec = get_device(generator_device)
            generator3d = generator3d.to(device_spec.target)
            if generator_device_runtime is None:
                generator_device_runtime = (
                    f"{device_spec.type}:{device_spec.index}" if device_spec.index is not None else device_spec.type
                )
        df_template = pd.read_csv(data_cfg.labelled)
        template_ds = build_gen3d_dataset(
            df_template,
            mol_col=getattr(data_cfg, "mol_column", "mol"),
            smiles_col=getattr(data_cfg, "smiles_column", getattr(data_cfg, "smile_column", "smile")),
            max_atoms=max_atoms,
        )
        if len(template_ds) > 0:
            import numpy as np

            gen3d_cfg = getattr(cfg, "generator3d", {}) or {}
            pool_size = int(getattr(gen3d_cfg, "template_pool_size", 256))
            seed = int(getattr(gen3d_cfg, "template_seed", 1337))
            z_pool = template_ds.zs
            mask_pool = template_ds.mask
            if pool_size > 0 and pool_size < len(template_ds):
                rng = np.random.default_rng(seed)
                idxs = rng.choice(len(template_ds), size=pool_size, replace=False)
                idxs = torch.as_tensor(idxs, dtype=torch.long)
                z_pool = z_pool.index_select(0, idxs)
                mask_pool = mask_pool.index_select(0, idxs)
            generator3d_template = {
                "z_pool": z_pool,
                "mask_pool": mask_pool,
            }
            logging.getLogger(__name__).info(
                "Loaded %d 3D templates for generator sampling.",
                int(z_pool.size(0)),
            )

    dft = None
    qc_store = None
    qc_manager = None
    if args.use_pseudo_dft:
        from src.data.dft_int import DFTInterface  # lazy import

        dft = DFTInterface()
    else:
        qc_cfg = getattr(cfg, "qc", None)
        if qc_cfg:
            from src.data.dft_int import DFTInterface  # lazy import
            from src.qc.config import GeometryConfig, QuantumTaskConfig, PipelineConfig
            from src.qc.pipeline import QCPipeline, AsyncQCManager
            from src.qc.storage import QCResultStore

            pipeline_data = {}
            pipeline_config_path = getattr(qc_cfg, "pipeline_config", None)
            if pipeline_config_path:
                pipeline_path = Path(pipeline_config_path)
                if not pipeline_path.exists():
                    raise FileNotFoundError(f"QC pipeline config not found: {pipeline_path}")
                with pipeline_path.open("r", encoding="utf-8") as fh:
                    pipeline_data = yaml.safe_load(fh) or {}
            defaults = PipelineConfig()
            geometry_cfg = GeometryConfig(**pipeline_data.get("geometry", {}))
            quantum_cfg = QuantumTaskConfig(**pipeline_data.get("quantum", {}))
            pipeline_kwargs = {
                "geometry": geometry_cfg,
                "quantum": quantum_cfg,
                "work_dir": Path(pipeline_data.get("work_dir", defaults.work_dir)),
                "max_workers": pipeline_data.get("max_workers", defaults.max_workers),
                "poll_interval": pipeline_data.get("poll_interval", defaults.poll_interval),
                "cleanup_workdir": pipeline_data.get("cleanup_workdir", defaults.cleanup_workdir),
                "store_metadata": pipeline_data.get("store_metadata", defaults.store_metadata),
                "allow_fallback": pipeline_data.get("allow_fallback", defaults.allow_fallback),
                "tracked_properties": tuple(pipeline_data.get("tracked_properties", defaults.tracked_properties)),
                "walltime_limit": pipeline_data.get("walltime_limit", defaults.walltime_limit),
            }
            pipeline_config = PipelineConfig(**pipeline_kwargs)
            if hasattr(qc_cfg, "engine"):
                pipeline_config.quantum.engine = qc_cfg.engine
            if hasattr(qc_cfg, "method"):
                pipeline_config.quantum.method = qc_cfg.method
            if hasattr(qc_cfg, "basis"):
                pipeline_config.quantum.basis = qc_cfg.basis
            if hasattr(qc_cfg, "properties"):
                pipeline_config.quantum.properties = tuple(qc_cfg.properties)
            if isinstance(pipeline_config.work_dir, str):
                pipeline_config.work_dir = Path(pipeline_config.work_dir)
            if isinstance(pipeline_config.quantum.scratch_dir, str):
                pipeline_config.quantum.scratch_dir = Path(pipeline_config.quantum.scratch_dir)
            pipeline_config.quantum.properties = tuple(pipeline_config.quantum.properties)
            pipeline_config.tracked_properties = tuple(pipeline_config.tracked_properties)
            store_path = getattr(qc_cfg, "result_store", None)
            if store_path:
                qc_store = QCResultStore(Path(store_path))
            pipeline = QCPipeline(pipeline_config, result_store=qc_store)
            qc_manager = AsyncQCManager(pipeline, max_workers=pipeline_config.max_workers)
            dft = DFTInterface(executor=qc_manager)
            dft_job_defaults = {
                "charge": pipeline_config.quantum.charge,
                "multiplicity": pipeline_config.quantum.multiplicity,
                "metadata": {
                    "engine": pipeline_config.quantum.engine,
                    "level_of_theory": pipeline_config.quantum.level_of_theory
                    or f"{pipeline_config.quantum.method}/{pipeline_config.quantum.basis}",
                },
            }

    loop = ActiveLearningLoop(
        surrogate=surrogate,
        optical_surrogate=optical_surrogate,
        oscillator_surrogate=oscillator_surrogate,
        labelled=labelled,
        pool=pool,
        config=loop_cfg,
        generator=generator,
        fragment_vocab=fragment_vocab,
        dft=dft,
        generator_device=generator_device_runtime,
        dft_job_defaults=dft_job_defaults,
    )
    loop.run(args.iterations)
    loop.save_history()
    if qc_manager is not None:
        qc_manager.shutdown()
    print("Active learning completed.")


def main() -> None:
    # Parse log level first so it is accepted regardless of position (before/after subcommands)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", default="INFO", help="Global log level (case-insensitive).")
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="OSC discovery toolkit", parents=[pre_parser])
    parser.set_defaults(log_level=pre_args.log_level)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-surrogate")
    train_parser.add_argument(
        "--config",
        default="configs/train_conf_3d_full.yaml",
        help="SchNet-only alias command; defaults to full SchNet config.",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--device",
        default=None,
        help="Device for surrogate training (e.g. 'auto', 'cuda', 'cuda:0', 'cpu').",
    )
    train_parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision (default: taken from config).",
    )
    train_parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.compile kernel fusion (default: from config).",
    )
    train_parser.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode (default inherits from config).",
    )
    train_parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request fullgraph compilation when using torch.compile.",
    )

    train3d_parser = subparsers.add_parser("train-surrogate-3d")
    train3d_parser.add_argument(
        "--config",
        default="configs/train_conf_3d_full.yaml",
        help="SchNet-only alias command; defaults to full SchNet config.",
    )
    train3d_parser.add_argument(
        "--device",
        default=None,
        help="Device for 3D surrogate training (e.g., 'auto', 'cuda', 'cpu'). Overrides config.",
    )

    train3d_full_parser = subparsers.add_parser("train-surrogate-3d-full")
    train3d_full_parser.add_argument("--config", default="configs/train_conf_3d_full.yaml")
    train3d_full_parser.add_argument(
        "--device",
        default=None,
        help="Device for full SchNet surrogate training (e.g., 'auto', 'cuda', 'cpu'). Overrides config.",
    )
    train3d_full_parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision for full SchNet (default: config).",
    )

    gen_parser = subparsers.add_parser("train-generator")
    gen_parser.add_argument("--config", default="configs/gen_conf.yaml")
    gen_parser.add_argument(
        "--device",
        default=None,
        help="Device for JT-VAE training (e.g. 'auto', 'cuda', 'cpu').",
    )
    gen_parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable torch.compile for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode for JT-VAE (default: config).",
    )
    gen_parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fullgraph toggle for torch.compile when training JT-VAE.",
    )

    gen3d_parser = subparsers.add_parser("train-generator-3d")
    gen3d_parser.add_argument("--config", default="configs/gen_conf_3d.yaml")
    gen3d_parser.add_argument(
        "--device",
        default=None,
        help="Device for 3D generator training (e.g., 'auto', 'cuda', 'cpu').",
    )

    al_parser = subparsers.add_parser("active-loop")
    al_parser.add_argument("--config", default="configs/active_learn.yaml")
    al_parser.add_argument(
        "--objective-mode",
        default=None,
        choices=["red", "blue", "general"],
        help="Optional objective preset mode (overrides loop.objective_mode in config).",
    )
    al_parser.add_argument(
        "--objective-profile",
        default=None,
        help="Path to objective profile YAML (default from config or configs/objectives.yaml).",
    )
    al_parser.add_argument(
        "--rl-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RL update path (overrides loop.rl.enabled).",
    )
    al_parser.add_argument(
        "--rl-every-n-iterations",
        type=int,
        default=None,
        help="Run RL update every N active-loop iterations.",
    )
    al_parser.add_argument(
        "--rl-steps-per-update",
        type=int,
        default=None,
        help="Number of RL gradient steps per update.",
    )
    al_parser.add_argument(
        "--rl-batch-size",
        type=int,
        default=None,
        help="RL sampling batch size.",
    )
    al_parser.add_argument(
        "--rl-lr",
        type=float,
        default=None,
        help="RL optimizer learning rate.",
    )
    al_parser.add_argument(
        "--rl-algorithm",
        default=None,
        choices=["reinforce", "policy_gradient", "ppo"],
        help="RL algorithm for generator updates.",
    )
    al_parser.add_argument(
        "--rl-entropy-weight",
        type=float,
        default=None,
        help="Entropy regularization weight for RL policy loss.",
    )
    al_parser.add_argument(
        "--rl-baseline-momentum",
        type=float,
        default=None,
        help="EMA momentum for reward baseline (0..1).",
    )
    al_parser.add_argument(
        "--rl-reward-clip",
        type=float,
        default=None,
        help="Absolute reward clip value used in RL loss.",
    )
    al_parser.add_argument(
        "--rl-normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable advantage normalization.",
    )
    al_parser.add_argument(
        "--rl-use-qc-top-k",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable QC override for top-K RL candidates.",
    )
    al_parser.add_argument(
        "--rl-qc-top-k",
        type=int,
        default=None,
        help="Top-K candidates for QC reward override in RL mode.",
    )
    al_parser.add_argument(
        "--rl-warmup-iterations",
        type=int,
        default=None,
        help="Skip RL updates for the first N completed loop iterations.",
    )
    al_parser.add_argument(
        "--rl-max-grad-norm",
        type=float,
        default=None,
        help="Gradient clipping max-norm for RL updates.",
    )
    al_parser.add_argument(
        "--rl-checkpoint-every",
        type=int,
        default=None,
        help="Write RL latest checkpoint every N RL updates.",
    )
    al_parser.add_argument(
        "--rl-value-loss-weight",
        type=float,
        default=None,
        help="Value-function loss weight for policy-gradient/PPO updates.",
    )
    al_parser.add_argument(
        "--rl-ppo-clip-ratio",
        type=float,
        default=None,
        help="PPO clipping epsilon.",
    )
    al_parser.add_argument(
        "--rl-ppo-epochs",
        type=int,
        default=None,
        help="Number of PPO epochs per RL update.",
    )
    al_parser.add_argument(
        "--rl-ppo-minibatch-size",
        type=int,
        default=None,
        help="Mini-batch size for PPO updates.",
    )
    al_parser.add_argument(
        "--rl-ppo-target-kl",
        type=float,
        default=None,
        help="Optional PPO early-stop KL target (set 0 or negative to disable).",
    )
    al_parser.add_argument(
        "--surrogate-dir",
        default="models/surrogate_3d_full",
        help="Path to full SchNet surrogate checkpoint or directory containing schnet_full*.pt artifacts.",
    )
    al_parser.add_argument(
        "--optical-surrogate-dir",
        default=None,
        help="Optional second surrogate path for optical targets (e.g. lambda_max_nm, oscillator_strength).",
    )
    al_parser.add_argument(
        "--oscillator-surrogate-dir",
        default=None,
        help="Optional third surrogate path for oscillator-focused targets (e.g. oscillator_strength).",
    )
    al_parser.add_argument("--generator-ckpt", default=None)
    al_parser.add_argument("--generator-3d-ckpt", default=None)
    al_parser.add_argument("--iterations", type=int, default=5)
    al_parser.add_argument("--seed", type=int, default=None)
    al_parser.add_argument("--use-pseudo-dft", action="store_true")
    al_parser.add_argument(
        "--device",
        default=None,
        help="Device override for both surrogate and generator inference.",
    )
    al_parser.add_argument(
        "--surrogate-device",
        default=None,
        help="Device override for surrogate inference (takes precedence over --device).",
    )
    al_parser.add_argument(
        "--generator-device",
        default=None,
        help="Device override for generator sampling (takes precedence over --device).",
    )

    args = parser.parse_args(remaining)
    log_level_name = str(args.log_level).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    setup_logging(level=log_level)

    if args.command == "train-surrogate":
        train_surrogate(args)
    elif args.command == "train-surrogate-3d":
        train_surrogate_3d(args)
    elif args.command == "train-surrogate-3d-full":
        train_surrogate_3d_full(args)
    elif args.command == "train-generator":
        train_generator(args)
    elif args.command == "train-generator-3d":
        train_generator_3d(args)
    elif args.command == "active-loop":
        run_active_loop(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
