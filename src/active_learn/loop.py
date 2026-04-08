"""
Active learning loop dirigiert surrogate, generator und DFT interface
"""

from __future__ import annotations

import contextlib
from collections import deque
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("project roout nicht auf src/")

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd
import torch

from src.active_learn.acq import AcquisitionConfig, acquisition_score
from src.active_learn.objectives import load_objective_profile, normalize_objective_mode
from src.active_learn.sched import ActiveLearningScheduler, SchedulerConfig
from src.data.dataset import split_dataframe
from src.data.dft_int import DFTInterface, DFTJobSpec, DFTResult
from src.data.featurization_3d import molblock_to_data
from src.models.jtvae_extended import JTVAE, sample_conditional, train_jtvae_rl_step
from src.models.reward import compute_reward_from_objective_profile
from src.utils.active_loop_dashboard import write_active_loop_dashboard
from src.utils.dashboard_server import start_dashboard_server
from src.utils.device import ensure_state_dict_on_cpu, get_device
from src.utils.log import get_logger

try:
    _sa_score = None  # optional SA scorer
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs, Lipinski, rdMolDescriptors, Crippen
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except Exception:  # auch optional
        rdFingerprintGenerator = None  # type: ignore
    try:
        # optionaler synthetic accessibility scorer (nicht in allen RDKits glaub) 
        from rdkit.Chem import rdMolDescriptors as _rdm
        from rdkit.Chem import Descriptors as _desc
        import sascorer as _sascorer  # type: ignore

        def _sa_score(mol: "Chem.Mol") -> float:
            return float(_sascorer.calculateScore(mol))

    except Exception:  # optional wie davor
        _sa_score = None  # type: ignore
    RDKit_AVAILABLE = True
except Exception:  # wieder optional, werden safe nie benutzt
    RDKit_AVAILABLE = False

logger = get_logger(__name__)

PROPERTY_DEFAULT_ALIASES = {
    "HOMO": "HOMO_eV",
    "LUMO": "LUMO_eV",
    "gap": "gap_eV",
    "IE": "IE_eV",
    "EA": "EA_eV",
    "lambda_max": "lambda_max_nm",
    "lambda_max_nm": "lambda_max_nm",
    "oscillator_strength": "oscillator_strength",
    "f_osc": "oscillator_strength",
}



@dataclass
class LoopConfig:
    batch_size: int = 8
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    target_columns: Sequence[str] = ("HOMO", "LUMO")
    maximise: Sequence[bool] = (False, True)
    generator_samples: int = 32
    generator_temperature: float = 1.0
    generator_attempts: int = 5  # batches to sample wenn pool keine candidates hat
    graph_cache_cap: int = 20000
    seed: Optional[int] = None  # seed fuer RNG (diversity, sampling etc)
    results_dir: Path = Path("experiments")
    assemble: Dict[str, object] = field(default_factory=dict)
    diversity_threshold: float = 0.85
    diversity_metric: str = "tanimoto"
    generator_refresh: Dict[str, object] = field(default_factory=dict)
    property_aliases: Dict[str, str] = field(default_factory=dict)
    rl_enabled: bool = False
    rl_every_n_iterations: int = 1
    rl_steps_per_update: int = 1
    rl_batch_size: int = 64
    rl_lr: float = 1e-5
    rl_algorithm: str = "policy_gradient"  # reinforce|policy_gradient|ppo
    rl_entropy_weight: float = 0.01
    rl_baseline_momentum: float = 0.9
    rl_reward_clip: Optional[float] = 5.0
    rl_normalize_advantage: bool = True
    rl_use_qc_top_k: bool = False
    rl_qc_top_k: int = 0
    rl_warmup_iterations: int = 0
    rl_max_grad_norm: Optional[float] = 1.0
    rl_checkpoint_every: int = 1
    rl_value_loss_weight: float = 0.5
    rl_actor_lr: Optional[float] = None
    rl_critic_lr: Optional[float] = None
    rl_anchor_weight: float = 0.0
    rl_reward_running_norm: bool = True
    rl_reward_norm_eps: float = 1e-6
    rl_reward_norm_clip: Optional[float] = 5.0
    rl_entropy_weight_start: Optional[float] = None
    rl_entropy_weight_end: Optional[float] = None
    rl_entropy_decay_updates: int = 200
    rl_ppo_clip_ratio: float = 0.2
    rl_ppo_epochs: int = 4
    rl_ppo_minibatch_size: Optional[int] = None
    rl_ppo_target_kl: Optional[float] = 0.03
    rl_ppo_value_clip_range: Optional[float] = 0.2
    rl_ppo_adaptive_kl: bool = False
    rl_ppo_adaptive_kl_high_mult: float = 1.5
    rl_ppo_adaptive_kl_low_mult: float = 0.5
    rl_ppo_lr_down_factor: float = 0.5
    rl_ppo_lr_up_factor: float = 1.05
    rl_ppo_clip_down_factor: float = 0.9
    rl_ppo_clip_up_factor: float = 1.02
    rl_ppo_actor_lr_min: Optional[float] = 1e-6
    rl_ppo_actor_lr_max: Optional[float] = 1e-3
    rl_ppo_clip_ratio_min: float = 0.05
    rl_ppo_clip_ratio_max: float = 0.4
    objective_mode: str = "red"  # profile switch: red|blue|general
    objective_profile_path: Optional[str] = None
    max_pool_eval: Optional[int] = None  # cap number von pool candidates evaluated pro iteration
    predict_batch_size: Optional[int] = None  # optional batch-size override fuer surrogate inference im pool
    predict_mc_samples: Optional[int] = None  # optional MC-dropout passes fuer surrogate inference
    max_generated_heavy_atoms: Optional[int] = None  # skipt generated SMILES mit zu vielen schweren atome
    max_generated_smiles_len: Optional[int] = None  # skipt generated SMILES mit zu langer Länge
    generated_smiles_len_factor: Optional[float] = 1.5  # fallback length cap als factor von median length wenn max nicht gesetted
    require_conjugation: bool = True  # macht basic OSC conjugation filter auf die generierten SMILES
    min_conjugated_bonds: int = 2  # selbserklärend
    min_alternating_conjugated_bonds: int = 3  # macht das alternating single/double conjugated path von dieser länge sind
    min_pi_conjugated_fraction: Optional[float] = None  # selbsterklärend
    min_aromatic_rings: int = 1  # selbsterklärend
    max_rotatable_bonds: Optional[int] = None  # cap flexibility
    max_rotatable_bonds_conjugated: Optional[int] = None  # cap rotatable bonds mit einem conjugated subgraph
    max_branch_points: Optional[int] = None  # cap anzahl von heavy atoms mit degree >= 3
    max_branch_degree: Optional[int] = None  # cap max heavy-atom degree
    max_charged_atoms: Optional[int] = None  # cap anzahl von geladenen atomen
    property_filters: Dict[str, Sequence[float]] = field(default_factory=dict)  # min/max per property
    qc_extra_properties: Sequence[str] = tuple()  # extra QC properties requested from DFT
    objective_gate_min_lambda_max_nm: Optional[float] = None
    objective_gate_max_lambda_max_nm: Optional[float] = None
    objective_gate_min_oscillator_strength: Optional[float] = None
    objective_gate_max_oscillator_strength: Optional[float] = None
    max_lambda_max_nm: Optional[float] = None  # reserved for future hard-gate support
    max_oscillator_strength: Optional[float] = None  # reserved for future hard-gate support
    min_lambda_max_nm: Optional[float] = None  # optional red-light gate threshold
    min_oscillator_strength: Optional[float] = None  # optional red-light gate threshold
    require_neutral: bool = True  #selbsterklaerend
    sa_score_max: Optional[float] = None  # optional siehe oben
    physchem_filters: Dict[str, Sequence[float]] = field(default_factory=dict)  # also clogp, tpsa, frac_csp3
    scaffold_unique: bool = False  # macht das man unique Murcko scaffolds hat
    exclude_smiles_paths: Sequence[str] = tuple()  # optional CSV/TXT files with SMILES to exclude
    auto_relax_filters: bool = True  #falls nichts generated wird, werden alle filter relaxed
    save_diagnostics: bool = False  # disable expensive matplotlib diagnostics by default (perf mode)
    diagnostics_every: int = 0  # write diagnostics every N iterations; <=0 disables plotting
    diagnostics_max_points: int = 12000  # subsample pool points when plotting diagnostics
    dft_job_defaults: Dict[str, object] = field(default_factory=dict)  # also charge, multiplicity, metadata etc.
    live_dashboard: Dict[str, object] = field(default_factory=dict)
    optical_score_weight: float = 0.0  # weight for optional secondary optical surrogate score
    optical_target_columns: Sequence[str] = ("lambda_max_nm", "oscillator_strength")
    optical_targets: Optional[Sequence[float]] = (650.0, 0.20)
    optical_tolerances: Optional[Sequence[float]] = (50.0, 0.10)
    optical_weights: Optional[Sequence[float]] = (1.0, 1.0)
    optical_beta: float = 0.0  # optional uncertainty bonus for optical score
    optical_weight_schedule: Sequence[Dict[str, float]] = tuple()  # stage schedule [{until_iteration, weight}]
    optical_predict_batch_size: Optional[int] = None
    optical_predict_mc_samples: Optional[int] = None
    min_labels_for_optical: int = 0  # warmup: enable optical acquisition erst ab dieser label-count
    optical_retrain_every: int = 0  # optional cadence (iterations) fuer optical surrogate retrain
    optical_retrain_min_labels: int = 500  # minimum labels with optical targets before retrain
    optical_retrain_val_fraction: float = 0.1
    optical_retrain_on_success_only: bool = True
    optical_incremental_path: Optional[str] = "data/processed/opv_optical_incremental.csv"
    optical_incremental_dedupe_on: Sequence[str] = ("smiles",)
    optical_incremental_require_all_targets: bool = True
    oscillator_score_weight: float = 0.0  # optional tertiary surrogate score for oscillator-only model
    oscillator_target_columns: Sequence[str] = ("oscillator_strength",)
    oscillator_targets: Optional[Sequence[float]] = (0.20,)
    oscillator_tolerances: Optional[Sequence[float]] = (0.10,)
    oscillator_weights: Optional[Sequence[float]] = (1.0,)
    oscillator_beta: float = 0.0
    oscillator_predict_batch_size: Optional[int] = None
    oscillator_predict_mc_samples: Optional[int] = None
    min_labels_for_oscillator: int = 0
    oscillator_retrain_every: int = 0
    oscillator_retrain_min_labels: int = 500
    oscillator_retrain_val_fraction: float = 0.1
    oscillator_retrain_on_success_only: bool = True
    objective_score_columns: Sequence[str] = ("lambda_max_nm", "oscillator_strength", "gap")
    objective_score_targets: Optional[Sequence[float]] = (680.0, 0.25, 1.8)
    objective_score_tolerances: Optional[Sequence[float]] = (60.0, 0.08, 0.45)
    objective_score_weights: Optional[Sequence[float]] = (1.2, 1.6, 0.8)
    objective_score_missing_penalty: float = 3.0
    objective_score_pass_threshold: Optional[float] = -2.0
    objective_score_require_qc_success: bool = True
    red_score_columns: Sequence[str] = ("lambda_max_nm", "oscillator_strength", "gap")
    red_score_targets: Optional[Sequence[float]] = (680.0, 0.25, 1.8)
    red_score_tolerances: Optional[Sequence[float]] = (60.0, 0.08, 0.45)
    red_score_weights: Optional[Sequence[float]] = (1.2, 1.6, 0.8)
    red_score_missing_penalty: float = 3.0
    red_score_pass_threshold: Optional[float] = -2.0
    red_score_require_qc_success: bool = True
    early_stop_no_improve_iterations: int = 0  # optional early stop based on best red_score
    early_stop_min_delta: float = 0.0
    export_top_k: int = 50
    export_sort_column: str = "red_score"
    export_require_qc_success: bool = True
    export_require_red_pass: bool = False
    export_top_candidates_path: Optional[str] = None


@contextlib.contextmanager
def _suppress_rdkit_errors():
    """RDKit Fehler spam unterdrücken."""

    if not RDKit_AVAILABLE:
        yield
        return
    try:
        RDLogger.DisableLog("rdApp.error")
        yield
    finally:
        try:
            RDLogger.EnableLog("rdApp.error")
        except Exception:
            pass


class ActiveLearningLoop:
    @staticmethod
    def _require_schnet_surrogate(surrogate_obj: Optional[object], role: str) -> None:
        if surrogate_obj is None:
            return
        is_schnet = bool(
            getattr(surrogate_obj, "is_schnet", False)
            or getattr(surrogate_obj, "is_schnet_full", False)
            or getattr(surrogate_obj, "schnet_like", False)
        )
        if not is_schnet:
            raise ValueError(
                f"SchNet-only mode active: {role} surrogate must be a full SchNet runtime wrapper."
            )

    def __init__(
        self,
        surrogate: object,
        labelled: pd.DataFrame,
        pool: pd.DataFrame,
        config: LoopConfig,
        *,
        generator: Optional[JTVAE] = None,
        fragment_vocab: Optional[Dict[str, int]] = None,
        optical_surrogate: Optional[object] = None,
        oscillator_surrogate: Optional[object] = None,
        dft: Optional[DFTInterface] = None,
        generator_device: Optional[str] = None,
        dft_job_defaults: Optional[Dict[str, object]] = None,
    ) -> None:
        self.surrogate = surrogate
        self.optical_surrogate = optical_surrogate
        self.oscillator_surrogate = oscillator_surrogate
        self.config = config
        self.labelled = labelled.reset_index(drop=True)
        self.pool = pool.reset_index(drop=True)
        self.generator = generator
        self.generator_device = generator_device
        self.fragment_vocab = fragment_vocab or {}
        self.dft = dft
        self.scheduler = ActiveLearningScheduler(config.scheduler)
        self.history: List[pd.DataFrame] = []
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.assemble_kwargs = dict(config.assemble)
        self.diversity_threshold = float(getattr(config, "diversity_threshold", 0.0))
        self.diversity_metric = getattr(config, "diversity_metric", "tanimoto").lower()
        self.generator_refresh_kwargs = dict(getattr(config, "generator_refresh", {}))
        self.property_aliases = dict(PROPERTY_DEFAULT_ALIASES)
        self.property_aliases.update(getattr(config, "property_aliases", {}))
        self._objective_mode = normalize_objective_mode(
            getattr(config, "objective_mode", "red"),
            default="red",
        )
        self._objective_profile_path = getattr(config, "objective_profile_path", None)
        self._objective_profile: Optional[Dict[str, Any]] = None
        try:
            profile, resolved_profile_path = load_objective_profile(
                self._objective_mode,
                self._objective_profile_path,
            )
            self._objective_profile = profile
            self._objective_profile_path = str(resolved_profile_path)
        except Exception as exc:
            logger.warning(
                "Objective profile could not be loaded for mode=%s path=%s (%s). "
                "Reward fallback defaults will be used.",
                self._objective_mode,
                self._objective_profile_path,
                exc,
            )
            self._objective_profile = None
        self._qc_extra_properties = [
            str(prop).strip()
            for prop in getattr(config, "qc_extra_properties", ())
            if str(prop).strip()
        ]
        self._min_lambda_max_nm = (
            float(config.min_lambda_max_nm) if getattr(config, "min_lambda_max_nm", None) is not None else None
        )
        self._min_oscillator_strength = (
            float(config.min_oscillator_strength)
            if getattr(config, "min_oscillator_strength", None) is not None
            else None
        )
        if self._min_lambda_max_nm is not None:
            self._qc_extra_properties.append("lambda_max")
        if self._min_oscillator_strength is not None:
            self._qc_extra_properties.append("oscillator_strength")
        self._qc_requested_properties = self._build_requested_qc_properties()
        seed = getattr(config, "seed", None)
        self._rng = np.random.default_rng(seed)
        self._smiles_cache: Dict[str, Optional[str]] = {}
        self._fingerprint_cache: Dict[str, Optional[object]] = {}
        self._fingerprints: List[object] = []
        self._morgan_generator = None
        self._graph_cache: Dict[str, object] = {}
        self._graph_cache_keys: deque[str] = deque()
        self._graph_cache_cap: int = int(getattr(config, "graph_cache_cap", 20000))  # soft cap to keep memory in check
        self._target_indices: List[int] = []
        self._refresh_target_indices()
        self._optical_target_columns: Tuple[str, ...] = tuple(
            str(col).strip() for col in getattr(config, "optical_target_columns", ()) if str(col).strip()
        )
        self._optical_target_indices: List[int] = []
        self._optical_score_weight = float(getattr(config, "optical_score_weight", 0.0) or 0.0)
        self._optical_base_weight = float(self._optical_score_weight)
        self._optical_beta = float(getattr(config, "optical_beta", 0.0) or 0.0)
        raw_opt_targets = getattr(config, "optical_targets", None)
        self._optical_targets: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_opt_targets) if raw_opt_targets is not None else None
        )
        raw_opt_tol = getattr(config, "optical_tolerances", None)
        self._optical_tolerances: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_opt_tol) if raw_opt_tol is not None else None
        )
        raw_opt_weights = getattr(config, "optical_weights", None)
        self._optical_weights: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_opt_weights) if raw_opt_weights is not None else None
        )
        self._optical_warned_no_targets = False
        self._require_schnet_surrogate(self.optical_surrogate, "optical")
        if self.optical_surrogate is not None:
            if not self._optical_target_columns:
                self._optical_target_columns = ("lambda_max_nm", "oscillator_strength")
            self._refresh_optical_target_indices()
            self._validate_optical_config()
        self._min_labels_for_optical = max(0, int(getattr(config, "min_labels_for_optical", 0) or 0))
        self._optical_retrain_every = max(0, int(getattr(config, "optical_retrain_every", 0) or 0))
        self._optical_retrain_min_labels = max(
            1,
            int(getattr(config, "optical_retrain_min_labels", 500) or 500),
        )
        self._optical_retrain_val_fraction = float(
            getattr(config, "optical_retrain_val_fraction", 0.1) or 0.1
        )
        self._optical_retrain_on_success_only = bool(
            getattr(config, "optical_retrain_on_success_only", True)
        )
        raw_optical_incremental_path = getattr(config, "optical_incremental_path", None)
        self._optical_incremental_path: Optional[Path] = (
            Path(raw_optical_incremental_path)
            if raw_optical_incremental_path not in (None, "", False)
            else None
        )
        self._optical_incremental_dedupe_on: Tuple[str, ...] = tuple(
            str(col).strip()
            for col in getattr(config, "optical_incremental_dedupe_on", ("smiles",))
            if str(col).strip()
        ) or ("smiles",)
        self._optical_incremental_require_all_targets = bool(
            getattr(config, "optical_incremental_require_all_targets", True)
        )
        self._optical_weight_schedule: List[Tuple[int, float]] = []
        raw_weight_schedule = getattr(config, "optical_weight_schedule", ()) or ()
        if raw_weight_schedule:
            for entry in raw_weight_schedule:
                if not isinstance(entry, dict):
                    continue
                until = int(entry.get("until_iteration", 0) or 0)
                weight = float(entry.get("weight", self._optical_base_weight) or 0.0)
                if until > 0:
                    self._optical_weight_schedule.append((until, weight))
            self._optical_weight_schedule.sort(key=lambda x: x[0])
        self._optical_weight_unlocked = False
        self._oscillator_target_columns: Tuple[str, ...] = tuple(
            str(col).strip() for col in getattr(config, "oscillator_target_columns", ()) if str(col).strip()
        )
        self._oscillator_target_indices: List[int] = []
        self._oscillator_score_weight = float(getattr(config, "oscillator_score_weight", 0.0) or 0.0)
        self._oscillator_base_weight = float(self._oscillator_score_weight)
        self._oscillator_beta = float(getattr(config, "oscillator_beta", 0.0) or 0.0)
        raw_osc_targets = getattr(config, "oscillator_targets", None)
        self._oscillator_targets: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_osc_targets) if raw_osc_targets is not None else None
        )
        raw_osc_tol = getattr(config, "oscillator_tolerances", None)
        self._oscillator_tolerances: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_osc_tol) if raw_osc_tol is not None else None
        )
        raw_osc_weights = getattr(config, "oscillator_weights", None)
        self._oscillator_weights: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_osc_weights) if raw_osc_weights is not None else None
        )
        self._oscillator_warned_no_targets = False
        self._require_schnet_surrogate(self.oscillator_surrogate, "oscillator")
        if self.oscillator_surrogate is not None:
            if not self._oscillator_target_columns:
                self._oscillator_target_columns = ("oscillator_strength",)
            self._refresh_oscillator_target_indices()
            self._validate_oscillator_config()
        self._min_labels_for_oscillator = max(0, int(getattr(config, "min_labels_for_oscillator", 0) or 0))
        self._oscillator_retrain_every = max(0, int(getattr(config, "oscillator_retrain_every", 0) or 0))
        self._oscillator_retrain_min_labels = max(
            1,
            int(getattr(config, "oscillator_retrain_min_labels", 500) or 500),
        )
        self._oscillator_retrain_val_fraction = float(
            getattr(config, "oscillator_retrain_val_fraction", 0.1) or 0.1
        )
        self._oscillator_retrain_on_success_only = bool(
            getattr(config, "oscillator_retrain_on_success_only", True)
        )
        self._oscillator_weight_unlocked = False
        self._red_score_columns: Tuple[str, ...] = tuple(
            str(col).strip() for col in getattr(config, "red_score_columns", ()) if str(col).strip()
        )
        raw_red_targets = getattr(config, "red_score_targets", None)
        self._red_score_targets: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_red_targets) if raw_red_targets is not None else None
        )
        raw_red_tolerances = getattr(config, "red_score_tolerances", None)
        self._red_score_tolerances: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_red_tolerances) if raw_red_tolerances is not None else None
        )
        raw_red_weights = getattr(config, "red_score_weights", None)
        self._red_score_weights: Optional[Tuple[float, ...]] = (
            tuple(float(v) for v in raw_red_weights) if raw_red_weights is not None else None
        )
        self._red_score_missing_penalty = float(
            getattr(config, "red_score_missing_penalty", 3.0) or 3.0
        )
        self._red_score_pass_threshold = getattr(config, "red_score_pass_threshold", None)
        if self._red_score_pass_threshold is not None:
            self._red_score_pass_threshold = float(self._red_score_pass_threshold)
        self._red_score_require_qc_success = bool(
            getattr(config, "red_score_require_qc_success", True)
        )
        self._validate_red_score_config()
        self._early_stop_no_improve_iterations = max(
            0,
            int(getattr(config, "early_stop_no_improve_iterations", 0) or 0),
        )
        self._early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 0.0) or 0.0)
        self._manual_stop_requested = False
        self._best_red_score_seen = float("-inf")
        self._iters_since_red_improve = 0
        self._export_top_k = max(0, int(getattr(config, "export_top_k", 50) or 0))
        self._export_sort_column = str(getattr(config, "export_sort_column", "red_score") or "red_score")
        self._export_require_qc_success = bool(getattr(config, "export_require_qc_success", True))
        self._export_require_red_pass = bool(getattr(config, "export_require_red_pass", False))
        raw_export_path = getattr(config, "export_top_candidates_path", None)
        self._export_top_candidates_path: Optional[Path] = (
            Path(raw_export_path) if raw_export_path not in (None, "", False) else None
        )
        self._filter_indices: Dict[str, int] = {}
        self._init_property_filters()
        self._require_schnet_surrogate(self.surrogate, "primary")
        self._scaffolds_seen: set[str] = set()
        self._excluded_smiles: set[str] = self._load_excluded_smiles(getattr(config, "exclude_smiles_paths", ()))
        self._excluded_smiles = self._canonicalize_smiles_set(self._excluded_smiles, "exclude list")
        self.labelled = self._canonicalize_dataframe(self.labelled, "labelled")
        self.pool = self._canonicalize_dataframe(self.pool, "pool")
        self._filter_pool_overlaps()
        self._median_smiles_len = self._compute_median_smiles_len()
        self._dft_job_defaults: Dict[str, object] = dict(dft_job_defaults or {})
        self._live_cfg = dict(getattr(config, "live_dashboard", {}) or {})
        self._live_enabled = bool(self._live_cfg.get("enabled", False))
        self._live_path = Path(
            self._live_cfg.get("path", self.results_dir / "active_loop_live_dashboard.html")
        )
        self._live_refresh_ms = max(300, int(self._live_cfg.get("refresh_ms", 1200)))
        self._live_top_k = max(1, int(self._live_cfg.get("selected_top_k", 8)))
        self._live_lines: List[str] = []
        self._live_history_rows: List[Dict[str, object]] = []
        self._live_started_at = time.time()
        self._live_total_iterations = max(1, int(getattr(self.scheduler.config, "max_iterations", 1)))
        self._live_server = None
        self._live_url: Optional[str] = None
        self._rl_enabled = bool(getattr(config, "rl_enabled", False))
        self._rl_every_n_iterations = max(1, int(getattr(config, "rl_every_n_iterations", 1) or 1))
        self._rl_steps_per_update = max(1, int(getattr(config, "rl_steps_per_update", 1) or 1))
        self._rl_batch_size = max(1, int(getattr(config, "rl_batch_size", 64) or 64))
        self._rl_lr = float(getattr(config, "rl_lr", 1e-5) or 1e-5)
        self._rl_algorithm = str(getattr(config, "rl_algorithm", "policy_gradient") or "policy_gradient").strip().lower()
        self._rl_entropy_weight = float(getattr(config, "rl_entropy_weight", 0.01) or 0.0)
        self._rl_entropy_weight_start = getattr(config, "rl_entropy_weight_start", None)
        if self._rl_entropy_weight_start is not None:
            self._rl_entropy_weight_start = float(self._rl_entropy_weight_start)
        self._rl_entropy_weight_end = getattr(config, "rl_entropy_weight_end", None)
        if self._rl_entropy_weight_end is not None:
            self._rl_entropy_weight_end = float(self._rl_entropy_weight_end)
        self._rl_entropy_decay_updates = max(1, int(getattr(config, "rl_entropy_decay_updates", 200) or 1))
        self._rl_baseline_momentum = float(getattr(config, "rl_baseline_momentum", 0.9) or 0.0)
        self._rl_reward_clip = getattr(config, "rl_reward_clip", 5.0)
        if self._rl_reward_clip is not None:
            self._rl_reward_clip = float(self._rl_reward_clip)
        self._rl_normalize_advantage = bool(getattr(config, "rl_normalize_advantage", True))
        self._rl_reward_running_norm = bool(getattr(config, "rl_reward_running_norm", True))
        self._rl_reward_norm_eps = float(getattr(config, "rl_reward_norm_eps", 1e-6) or 1e-6)
        self._rl_reward_norm_clip = getattr(config, "rl_reward_norm_clip", 5.0)
        if self._rl_reward_norm_clip is not None:
            self._rl_reward_norm_clip = float(self._rl_reward_norm_clip)
        self._rl_use_qc_top_k = bool(getattr(config, "rl_use_qc_top_k", False))
        self._rl_qc_top_k = max(0, int(getattr(config, "rl_qc_top_k", 0) or 0))
        self._rl_warmup_iterations = max(0, int(getattr(config, "rl_warmup_iterations", 0) or 0))
        self._rl_max_grad_norm = getattr(config, "rl_max_grad_norm", 1.0)
        if self._rl_max_grad_norm is not None:
            self._rl_max_grad_norm = float(self._rl_max_grad_norm)
        self._rl_checkpoint_every = max(1, int(getattr(config, "rl_checkpoint_every", 1) or 1))
        self._rl_value_loss_weight = float(getattr(config, "rl_value_loss_weight", 0.5) or 0.5)
        self._rl_actor_lr = getattr(config, "rl_actor_lr", None)
        if self._rl_actor_lr is not None:
            self._rl_actor_lr = float(self._rl_actor_lr)
        self._rl_critic_lr = getattr(config, "rl_critic_lr", None)
        if self._rl_critic_lr is not None:
            self._rl_critic_lr = float(self._rl_critic_lr)
        self._rl_anchor_weight = float(getattr(config, "rl_anchor_weight", 0.0) or 0.0)
        self._rl_ppo_clip_ratio = float(getattr(config, "rl_ppo_clip_ratio", 0.2) or 0.2)
        self._rl_ppo_epochs = max(1, int(getattr(config, "rl_ppo_epochs", 4) or 4))
        self._rl_ppo_minibatch_size = getattr(config, "rl_ppo_minibatch_size", None)
        if self._rl_ppo_minibatch_size is not None:
            self._rl_ppo_minibatch_size = max(1, int(self._rl_ppo_minibatch_size))
        self._rl_ppo_target_kl = getattr(config, "rl_ppo_target_kl", 0.03)
        if self._rl_ppo_target_kl is not None:
            self._rl_ppo_target_kl = float(self._rl_ppo_target_kl)
            if self._rl_ppo_target_kl <= 0.0:
                self._rl_ppo_target_kl = None
        self._rl_ppo_value_clip_range = getattr(config, "rl_ppo_value_clip_range", 0.2)
        if self._rl_ppo_value_clip_range is not None:
            self._rl_ppo_value_clip_range = abs(float(self._rl_ppo_value_clip_range))
        self._rl_ppo_adaptive_kl = bool(getattr(config, "rl_ppo_adaptive_kl", False))
        self._rl_ppo_adaptive_kl_high_mult = max(
            1.0,
            float(getattr(config, "rl_ppo_adaptive_kl_high_mult", 1.5) or 1.5),
        )
        self._rl_ppo_adaptive_kl_low_mult = float(getattr(config, "rl_ppo_adaptive_kl_low_mult", 0.5) or 0.5)
        self._rl_ppo_adaptive_kl_low_mult = max(
            0.0,
            min(self._rl_ppo_adaptive_kl_low_mult, self._rl_ppo_adaptive_kl_high_mult),
        )
        self._rl_ppo_lr_down_factor = float(getattr(config, "rl_ppo_lr_down_factor", 0.5) or 0.5)
        self._rl_ppo_lr_up_factor = float(getattr(config, "rl_ppo_lr_up_factor", 1.05) or 1.05)
        self._rl_ppo_clip_down_factor = float(getattr(config, "rl_ppo_clip_down_factor", 0.9) or 0.9)
        self._rl_ppo_clip_up_factor = float(getattr(config, "rl_ppo_clip_up_factor", 1.02) or 1.02)
        self._rl_ppo_actor_lr_min = getattr(config, "rl_ppo_actor_lr_min", 1e-6)
        if self._rl_ppo_actor_lr_min is not None:
            self._rl_ppo_actor_lr_min = float(self._rl_ppo_actor_lr_min)
        self._rl_ppo_actor_lr_max = getattr(config, "rl_ppo_actor_lr_max", 1e-3)
        if self._rl_ppo_actor_lr_max is not None:
            self._rl_ppo_actor_lr_max = float(self._rl_ppo_actor_lr_max)
        self._rl_ppo_clip_ratio_min = max(
            1e-4,
            float(getattr(config, "rl_ppo_clip_ratio_min", 0.05) or 0.05),
        )
        self._rl_ppo_clip_ratio_max = max(
            self._rl_ppo_clip_ratio_min,
            float(getattr(config, "rl_ppo_clip_ratio_max", 0.4) or 0.4),
        )
        self._rl_optimizer: Optional[torch.optim.Optimizer] = None
        self._rl_baseline_state: Dict[str, float] = {"value": 0.0}
        self._rl_last_metrics: Dict[str, float] = {}
        self._rl_updates: int = 0
        self._rl_best_reward_mean: float = float("-inf")
        self._rl_reward_stats: Dict[str, float] = {
            "mean": 0.0,
            "var": 1.0,
            "count": 0.0,
        }
        self._rl_dir = self.results_dir / "rl"
        if self._rl_enabled:
            self._rl_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "RL enabled (mode=%s algo=%s every=%d steps=%d batch=%d lr=%g qc_top_k=%s).",
                self._objective_mode,
                self._rl_algorithm,
                self._rl_every_n_iterations,
                self._rl_steps_per_update,
                self._rl_batch_size,
                self._rl_lr,
                self._rl_qc_top_k if self._rl_use_qc_top_k else 0,
            )
        if self.generator is not None:
            self.generator.eval()
            if self.generator_device is None:
                try:
                    self.generator_device = str(next(self.generator.parameters()).device)
                except StopIteration:
                    self.generator_device = None
        if RDKit_AVAILABLE and self.diversity_threshold > 0:
            logger.info(
                "Precomputet diversity fingerprints (threshold=%.2f) fuer %d molecules...",
                self.diversity_threshold,
                len(pd.concat(
                    [self.labelled.get("smiles", pd.Series(dtype=str)), self.pool.get("smiles", pd.Series(dtype=str))],
                    axis=0,
                ).dropna().unique()),
            )
            initial_smiles = pd.concat(
                [self.labelled.get("smiles", pd.Series(dtype=str)), self.pool.get("smiles", pd.Series(dtype=str))],
                axis=0,
            ).dropna().unique()
            for idx, smi in enumerate(initial_smiles, 1):
                fp = self._fingerprint(smi)
                if fp is not None:
                    self._fingerprint_cache[smi] = fp
                    self._fingerprints.append(fp)
                if idx % 5000 == 0:
                    logger.debug("Processed %d/%d fingerprints...", idx, len(initial_smiles))
            logger.info("beendet fingerprint precompute (%d cached).", len(self._fingerprints))

        if len(self.config.target_columns) != len(self.config.maximise):
            raise ValueError("target_columns und maximise length mismatch.")
        self._init_live_dashboard()

    def _build_requested_qc_properties(self) -> List[str]:
        inverse_alias = {v: k for k, v in self.property_aliases.items()}
        requested: List[str] = []
        seen: set[str] = set()
        for column in self.config.target_columns:
            base = str(inverse_alias.get(column, column))
            key = self._norm_property_name(base)
            if key not in seen:
                seen.add(key)
                requested.append(base)
        for prop in self._qc_extra_properties:
            key = self._norm_property_name(prop)
            if key not in seen:
                seen.add(key)
                requested.append(prop)
        return requested

    def _append_live(self, message: str) -> None:
        if not self._live_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        self._live_lines.append(f"[{stamp}] {message}")
        if len(self._live_lines) > 300:
            del self._live_lines[:-300]

    def _init_live_dashboard(self) -> None:
        if not self._live_enabled:
            return
        self._append_live(f"Active-loop dashboard enabled: {self._live_path}")
        self._update_live_dashboard(selected=None, generated=0)
        if bool(self._live_cfg.get("local_view_enabled", False)):
            try:
                self._live_server, self._live_url = start_dashboard_server(
                    self._live_path,
                    host=str(self._live_cfg.get("local_view_host", "127.0.0.1")),
                    port=int(self._live_cfg.get("local_view_port", 0)),
                    open_browser=bool(self._live_cfg.get("open_browser", True)),
                )
                if self._live_url:
                    self._append_live(f"Local view: {self._live_url}")
            except Exception as exc:
                logger.exception("Failed to start active-loop local dashboard server.")
                self._append_live(f"Local view failed: {exc}")

    def _update_live_dashboard(
        self,
        *,
        selected: Optional[pd.DataFrame],
        generated: int,
    ) -> None:
        if not self._live_enabled:
            return
        selected_rows: List[Dict[str, object]] = []
        if selected is not None and not selected.empty:
            prediction_targets: List[str] = list(self.config.target_columns)
            for target in self._optical_target_columns:
                if target not in prediction_targets:
                    prediction_targets.append(target)
            for target in self._oscillator_target_columns:
                if target not in prediction_targets:
                    prediction_targets.append(target)
            for _, row in selected.head(self._live_top_k).iterrows():
                predictions: List[Dict[str, object]] = []
                for target in prediction_targets:
                    predictions.append(
                        {
                            "name": target,
                            "pred": row.get(f"pred_{target}", None),
                            "std": row.get(f"pred_std_{target}", None),
                            "label": row.get(target, None),
                        }
                    )
                status = row.get("qc_status", row.get("assembly_status", "selected"))
                selected_rows.append(
                    {
                        "smiles": row.get("smiles", ""),
                        "acquisition_score": row.get("acquisition_score", None),
                        "status": status,
                        "predictions": predictions,
                    }
                )
        try:
            write_active_loop_dashboard(
                self._live_path,
                iteration=int(self.scheduler.iteration),
                total_iterations=int(self._live_total_iterations),
                labelled_count=int(len(self.labelled)),
                pool_count=int(len(self.pool)),
                generated_last=int(generated),
                selected_last=0 if selected is None else int(len(selected)),
                selected_rows=selected_rows,
                history_rows=list(self._live_history_rows),
                refresh_ms=self._live_refresh_ms,
                started_at=self._live_started_at,
                cli_lines=self._live_lines,
            )
        except Exception:
            logger.exception("Failed updating active-loop live dashboard.")

    def _current_best(self) -> Optional[np.ndarray]:
        if self.labelled.empty:
            return None
        target_cols = list(self.config.target_columns)
        arr = self.labelled[target_cols].to_numpy(dtype=float)
        best = []
        for dim, maximise in enumerate(self.config.maximise):
            column = arr[:, dim]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                best.append(0.0)
            else:
                best.append(finite.max() if maximise else finite.min())
        return np.array(best)

    def _fingerprint(self, smiles: str):
        if not RDKit_AVAILABLE:
            return None
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles]
        with _suppress_rdkit_errors():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = None
        if rdFingerprintGenerator is not None:
            if self._morgan_generator is None:
                self._morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            try:
                fp = self._morgan_generator.GetFingerprint(mol)
            except Exception:
                fp = None
        if fp is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        self._fingerprint_cache[smiles] = fp
        return fp

    def _refresh_target_indices(self) -> None:
        """Align desired target columns with surrogate outputs (supports aliases/unit suffixes)."""
        surrogate_targets = list(getattr(self.surrogate, "target_columns", ()))
        if not surrogate_targets:
            raise ValueError("Surrogate keine target_columns defined.")
        self._target_indices = []
        resolved_pairs: List[str] = []
        for target_name in self.config.target_columns:
            resolved = self._resolve_surrogate_target(target_name, surrogate_targets)
            if resolved is None:
                raise ValueError(
                    f"Target column '{target_name}' nicht gefunden in surrogate outputs: {surrogate_targets}"
                )
            self._target_indices.append(surrogate_targets.index(resolved))
            resolved_pairs.append(f"{target_name}->{resolved}")
        logger.info("Active-loop target mapping: %s", ", ".join(resolved_pairs))

    def _refresh_optical_target_indices(self) -> None:
        if self.optical_surrogate is None:
            self._optical_target_indices = []
            return
        surrogate_targets = list(getattr(self.optical_surrogate, "target_columns", ()))
        if not surrogate_targets:
            raise ValueError("Optical surrogate has no target_columns defined.")
        self._optical_target_indices = []
        resolved_pairs: List[str] = []
        for target_name in self._optical_target_columns:
            resolved = self._resolve_surrogate_target(target_name, surrogate_targets)
            if resolved is None:
                raise ValueError(
                    f"Optical target column '{target_name}' not found in optical surrogate outputs: {surrogate_targets}"
                )
            self._optical_target_indices.append(surrogate_targets.index(resolved))
            resolved_pairs.append(f"{target_name}->{resolved}")
        logger.info("Optical surrogate target mapping: %s", ", ".join(resolved_pairs))

    def _validate_optical_config(self) -> None:
        n_targets = len(self._optical_target_columns)
        if n_targets == 0:
            return
        if self._optical_targets is not None and len(self._optical_targets) != n_targets:
            raise ValueError(
                f"optical_targets length mismatch: expected {n_targets}, got {len(self._optical_targets)}"
            )
        if self._optical_tolerances is not None and len(self._optical_tolerances) != n_targets:
            raise ValueError(
                f"optical_tolerances length mismatch: expected {n_targets}, got {len(self._optical_tolerances)}"
            )
        if self._optical_weights is not None and len(self._optical_weights) != n_targets:
            raise ValueError(
                f"optical_weights length mismatch: expected {n_targets}, got {len(self._optical_weights)}"
            )

    def _refresh_oscillator_target_indices(self) -> None:
        if self.oscillator_surrogate is None:
            self._oscillator_target_indices = []
            return
        surrogate_targets = list(getattr(self.oscillator_surrogate, "target_columns", ()))
        if not surrogate_targets:
            raise ValueError("Oscillator surrogate has no target_columns defined.")
        self._oscillator_target_indices = []
        resolved_pairs: List[str] = []
        for target_name in self._oscillator_target_columns:
            resolved = self._resolve_surrogate_target(target_name, surrogate_targets)
            if resolved is None:
                raise ValueError(
                    f"Oscillator target column '{target_name}' not found in oscillator surrogate outputs: {surrogate_targets}"
                )
            self._oscillator_target_indices.append(surrogate_targets.index(resolved))
            resolved_pairs.append(f"{target_name}->{resolved}")
        logger.info("Oscillator surrogate target mapping: %s", ", ".join(resolved_pairs))

    def _validate_oscillator_config(self) -> None:
        n_targets = len(self._oscillator_target_columns)
        if n_targets == 0:
            return
        if self._oscillator_targets is not None and len(self._oscillator_targets) != n_targets:
            raise ValueError(
                f"oscillator_targets length mismatch: expected {n_targets}, got {len(self._oscillator_targets)}"
            )
        if self._oscillator_tolerances is not None and len(self._oscillator_tolerances) != n_targets:
            raise ValueError(
                f"oscillator_tolerances length mismatch: expected {n_targets}, got {len(self._oscillator_tolerances)}"
            )
        if self._oscillator_weights is not None and len(self._oscillator_weights) != n_targets:
            raise ValueError(
                f"oscillator_weights length mismatch: expected {n_targets}, got {len(self._oscillator_weights)}"
            )

    def _validate_red_score_config(self) -> None:
        n_terms = len(self._red_score_columns)
        if n_terms == 0:
            return
        if self._red_score_targets is None:
            raise ValueError("red_score_columns set but red_score_targets are missing.")
        if len(self._red_score_targets) != n_terms:
            raise ValueError(
                f"red_score_targets length mismatch: expected {n_terms}, got {len(self._red_score_targets)}"
            )
        if self._red_score_tolerances is not None and len(self._red_score_tolerances) != n_terms:
            raise ValueError(
                f"red_score_tolerances length mismatch: expected {n_terms}, got {len(self._red_score_tolerances)}"
            )
        if self._red_score_weights is not None and len(self._red_score_weights) != n_terms:
            raise ValueError(
                f"red_score_weights length mismatch: expected {n_terms}, got {len(self._red_score_weights)}"
            )
        if self._red_score_missing_penalty < 0:
            raise ValueError("red_score_missing_penalty must be >= 0.")

    def _count_optical_labels(self, frame: pd.DataFrame, *, require_success: bool) -> int:
        if frame.empty or not self._optical_target_columns:
            return 0
        target_cols = [col for col in self._optical_target_columns if col in frame.columns]
        if len(target_cols) != len(self._optical_target_columns):
            return 0
        mask = frame[target_cols].notna().all(axis=1)
        if require_success and "qc_status" in frame.columns:
            status = frame["qc_status"].astype(str).str.lower()
            mask = mask & (status == "success")
        return int(mask.sum())

    def _count_oscillator_labels(self, frame: pd.DataFrame, *, require_success: bool) -> int:
        if frame.empty or not self._oscillator_target_columns:
            return 0
        target_cols = [col for col in self._oscillator_target_columns if col in frame.columns]
        if len(target_cols) != len(self._oscillator_target_columns):
            return 0
        mask = frame[target_cols].notna().all(axis=1)
        if require_success and "qc_status" in frame.columns:
            status = frame["qc_status"].astype(str).str.lower()
            mask = mask & (status == "success")
        return int(mask.sum())

    def _effective_optical_weight(self) -> float:
        if self.optical_surrogate is None:
            return 0.0
        base_weight = float(self._optical_base_weight)
        if self._optical_weight_schedule:
            iter_idx = int(self.scheduler.iteration + 1)
            base_weight = float(self._optical_weight_schedule[-1][1])
            for until, stage_weight in self._optical_weight_schedule:
                if iter_idx <= int(until):
                    base_weight = float(stage_weight)
                    break
        if abs(base_weight) < 1e-12:
            return 0.0
        if self._min_labels_for_optical <= 0:
            return base_weight
        n_labels = self._count_optical_labels(
            self.labelled,
            require_success=self._optical_retrain_on_success_only,
        )
        if n_labels < self._min_labels_for_optical:
            if self._optical_weight_unlocked:
                self._optical_weight_unlocked = False
            logger.info(
                "Optical acquisition warmup active: %d/%d labelled rows with optical targets.",
                n_labels,
                self._min_labels_for_optical,
            )
            return 0.0
        if not self._optical_weight_unlocked:
            logger.info(
                "Optical acquisition enabled after warmup: %d labels reached (threshold=%d).",
                n_labels,
                self._min_labels_for_optical,
            )
            self._optical_weight_unlocked = True
        return base_weight

    def _effective_oscillator_weight(self) -> float:
        if self.oscillator_surrogate is None:
            return 0.0
        base_weight = float(self._oscillator_base_weight)
        if abs(base_weight) < 1e-12:
            return 0.0
        if self._min_labels_for_oscillator <= 0:
            return base_weight
        n_labels = self._count_oscillator_labels(
            self.labelled,
            require_success=self._oscillator_retrain_on_success_only,
        )
        if n_labels < self._min_labels_for_oscillator:
            if self._oscillator_weight_unlocked:
                self._oscillator_weight_unlocked = False
            logger.info(
                "Oscillator acquisition warmup active: %d/%d labelled rows with oscillator targets.",
                n_labels,
                self._min_labels_for_oscillator,
            )
            return 0.0
        if not self._oscillator_weight_unlocked:
            logger.info(
                "Oscillator acquisition enabled after warmup: %d labels reached (threshold=%d).",
                n_labels,
                self._min_labels_for_oscillator,
            )
            self._oscillator_weight_unlocked = True
        return base_weight

    def _compute_red_score(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if not self._red_score_columns or self._red_score_targets is None:
            return
        inverse_alias = {v: k for k, v in self.property_aliases.items()}

        def _resolve_value(row_index: int, desired_col: str):
            candidates: List[str] = [desired_col]
            mapped = self.property_aliases.get(desired_col)
            if mapped:
                candidates.append(str(mapped))
            inverse = inverse_alias.get(desired_col)
            if inverse:
                candidates.append(str(inverse))
            if desired_col.endswith("_eV") or desired_col.endswith("_nm"):
                candidates.append(desired_col[:-3])
            else:
                candidates.extend([f"{desired_col}_eV", f"{desired_col}_nm"])
            seen: set[str] = set()
            for cand in candidates:
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                if key in frame.columns:
                    return frame.at[row_index, key]
            return np.nan

        targets = np.asarray(self._red_score_targets, dtype=float)
        if self._red_score_tolerances is None:
            tolerances = np.ones_like(targets, dtype=float)
        else:
            tolerances = np.asarray(self._red_score_tolerances, dtype=float)
        tolerances = np.where(tolerances <= 0.0, 1.0, tolerances)
        if self._red_score_weights is None:
            weights = np.ones_like(targets, dtype=float)
        else:
            weights = np.asarray(self._red_score_weights, dtype=float)

        for row in frame.itertuples(index=True):
            idx = row.Index
            if (
                self._red_score_require_qc_success
                and "qc_status" in frame.columns
                and str(frame.at[idx, "qc_status"]).lower() != "success"
            ):
                frame.at[idx, "red_score"] = np.nan
                frame.at[idx, "red_pass"] = False
                frame.at[idx, "red_reason"] = "qc_not_success"
                continue

            score = 0.0
            reasons: List[str] = []
            for col, target, tol, weight in zip(
                self._red_score_columns,
                targets,
                tolerances,
                weights,
            ):
                val = _resolve_value(idx, col)
                if pd.isna(val):
                    score -= float(self._red_score_missing_penalty) * float(weight)
                    reasons.append(f"missing_{col}")
                    continue
                diff = abs(float(val) - float(target)) / float(tol)
                score -= float(weight) * diff
                if diff > 1.0:
                    reasons.append(f"{col}_off")

            passed = True
            if self._red_score_pass_threshold is not None:
                passed = score >= float(self._red_score_pass_threshold)
                if not passed and "score_below_threshold" not in reasons:
                    reasons.append("score_below_threshold")
            frame.at[idx, "red_score"] = float(score)
            frame.at[idx, "red_pass"] = bool(passed)
            frame.at[idx, "red_reason"] = "ok" if passed else ";".join(reasons) if reasons else "no_reason"

    def _append_optical_incremental_dataset(self, frame: pd.DataFrame) -> None:
        if self._optical_incremental_path is None or frame.empty:
            return
        work = frame.copy()
        if self._optical_retrain_on_success_only and "qc_status" in work.columns:
            status = work["qc_status"].astype(str).str.lower()
            work = work[status == "success"].copy()
        target_cols = [col for col in self._optical_target_columns if col in work.columns]
        if not target_cols:
            return
        if self._optical_incremental_require_all_targets:
            work = work.dropna(subset=target_cols)
        else:
            work = work[work[target_cols].notna().any(axis=1)].copy()
        if work.empty:
            return

        preferred_cols = [
            "smiles",
            "mol",
            "ctag",
            "homo",
            "lumo",
            "gap",
            *self._optical_target_columns,
            "red_score",
            "red_pass",
            "red_reason",
            "iteration",
            "qc_status",
            "qc_wall_time",
            "qc_error",
            "basis",
            "total_energy",
            "qc_metadata",
        ]
        keep_cols = [col for col in preferred_cols if col in work.columns]
        if not keep_cols:
            return
        work = work[keep_cols].copy()

        out_path = self._optical_incremental_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            merged = pd.concat([existing, work], ignore_index=True)
        else:
            merged = work

        dedupe_cols = [col for col in self._optical_incremental_dedupe_on if col in merged.columns]
        if dedupe_cols:
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
        else:
            merged = merged.drop_duplicates(keep="last")

        merged.to_csv(out_path, index=False)
        logger.info(
            "Updated optical incremental dataset at %s (rows=%d).",
            out_path,
            len(merged),
        )

    def _update_red_score_improvement(self, labelled: pd.DataFrame) -> None:
        if self._early_stop_no_improve_iterations <= 0:
            return
        if "red_score" not in labelled.columns:
            return
        current = pd.to_numeric(labelled["red_score"], errors="coerce")
        current = current[np.isfinite(current.to_numpy(dtype=float))]
        if current.empty:
            self._iters_since_red_improve += 1
        else:
            current_best = float(current.max())
            if current_best > self._best_red_score_seen + float(self._early_stop_min_delta):
                self._best_red_score_seen = current_best
                self._iters_since_red_improve = 0
                logger.info("red_score improved: best=%.4f", self._best_red_score_seen)
            else:
                self._iters_since_red_improve += 1
        if self._iters_since_red_improve >= self._early_stop_no_improve_iterations:
            self._manual_stop_requested = True
            logger.info(
                "Early stop requested: no red_score improvement for %d iterations.",
                self._iters_since_red_improve,
            )

    @staticmethod
    def _norm_property_name(name: str) -> str:
        key = str(name).strip().lower()
        if key.endswith("_ev") or key.endswith("_nm"):
            key = key[:-3]
        return key

    def _resolve_surrogate_target(self, name: str, surrogate_targets: Sequence[str]) -> Optional[str]:
        if not surrogate_targets:
            return None
        if name in surrogate_targets:
            return name

        # Direct case-insensitive match.
        lower_map = {str(col).lower(): str(col) for col in surrogate_targets}
        direct_ci = lower_map.get(str(name).lower())
        if direct_ci is not None:
            return direct_ci

        # Normalized base-name match: e.g. homo -> HOMO_eV.
        wanted = self._norm_property_name(name)
        normalized_matches = [
            str(col) for col in surrogate_targets if self._norm_property_name(str(col)) == wanted
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        if normalized_matches:
            return normalized_matches[0]

        # Alias-based candidates (both directions).
        inverse_alias = {v: k for k, v in self.property_aliases.items()}
        alias_candidates: List[str] = []
        for key in (name, str(name).lower(), str(name).upper(), str(name).capitalize()):
            mapped = self.property_aliases.get(key)
            if mapped:
                alias_candidates.append(str(mapped))
            inv = inverse_alias.get(key)
            if inv:
                alias_candidates.append(str(inv))

        for candidate in alias_candidates:
            if candidate in surrogate_targets:
                return candidate
            ci = lower_map.get(candidate.lower())
            if ci is not None:
                return ci
            cand_norm = self._norm_property_name(candidate)
            for col in surrogate_targets:
                if self._norm_property_name(str(col)) == cand_norm:
                    return str(col)
        return None

    def _parse_smiles(self, smiles: str):
        """Parset SMILES waehrend suppressed RDKit stderr spam auf invalid inputs"""
        if not RDKit_AVAILABLE or not smiles:
            return None
        with _suppress_rdkit_errors():
            return Chem.MolFromSmiles(smiles)

    def _init_property_filters(self) -> None:
        """Map property_filters keys zum surrogate output indices"""
        self._filter_indices.clear()
        if not self.config.property_filters:
            return
        surrogate_targets = list(getattr(self.surrogate, "target_columns", ()))
        missing = []
        for prop in self.config.property_filters.keys():
            resolved = self._resolve_surrogate_target(prop, surrogate_targets)
            if resolved is None:
                missing.append(prop)
                continue
            self._filter_indices[prop] = surrogate_targets.index(resolved)
        if missing:
            logger.warning("property_filters entries nicht da in surrogate outputs und wird ignoriert: %s", missing)

    def _load_excluded_smiles(self, paths: Sequence[str]) -> set[str]:
        excluded: set[str] = set()
        for p in paths or ():
            try:
                path = Path(p)
                if not path.exists():
                    logger.warning("Exclude SMILES path not found: %s", path)
                    continue
                before_count = len(excluded)
                if path.suffix.lower() in {".csv", ".tsv"}:
                    import pandas as pd  # lazy import

                    df = pd.read_csv(path)
                    smiles_col = None
                    for candidate in ("smiles", "smile", "SMILES", "Smile"):
                        if candidate in df.columns:
                            smiles_col = candidate
                            break
                    if smiles_col is None:
                        for col in df.columns:
                            if "smile" in str(col).lower():
                                smiles_col = str(col)
                                break
                    if smiles_col is not None:
                        excluded.update(df[smiles_col].dropna().astype(str).tolist())
                    else:
                        first_col = str(df.columns[0]) if len(df.columns) else ""
                        if first_col.startswith("version https://git-lfs.github.com/spec/v1"):
                            logger.warning(
                                "Exclude SMILES CSV appears to be a Git LFS pointer: %s. Run `git lfs pull`.",
                                path,
                            )
                        else:
                            logger.warning(
                                "Exclude SMILES CSV missing smiles column (smiles/smile): %s",
                                path,
                            )
                else:
                    # als plain text (1 SMILES pro linie)
                    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if line:
                            excluded.add(line)
                added = len(excluded) - before_count
                if added > 0:
                    logger.info("Loaded %d exclude SMILES from %s", added, path)
            except Exception as exc:
                logger.warning("fehler beim laden exclude SMILES von %s: %s", p, exc)
        return excluded

    def _canonical_smiles(self, smiles: str) -> Optional[str]:
        if smiles is None:
            return None
        text = str(smiles).strip()
        if not text:
            return None
        if text in self._smiles_cache:
            return self._smiles_cache[text]
        if not RDKit_AVAILABLE:
            self._smiles_cache[text] = text
            return text
        mol = self._parse_smiles(text)
        if mol is None:
            self._smiles_cache[text] = None
            return None
        try:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            canonical = None
        self._smiles_cache[text] = canonical
        if canonical is not None:
            self._smiles_cache.setdefault(canonical, canonical)
        return canonical

    def _canonicalize_smiles_set(self, smiles: Sequence[str], name: str) -> set[str]:
        canonical: set[str] = set()
        invalid = 0
        for smi in smiles:
            canon = self._canonical_smiles(smi)
            if canon:
                canonical.add(canon)
            else:
                invalid += 1
        if invalid:
            logger.warning("Dropped %d invalid SMILES von %s", invalid, name)
        return canonical

    def _canonicalize_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "smiles" not in df.columns:
            return df
        canonical = []
        invalid = 0
        for smi in df["smiles"].tolist():
            canon = self._canonical_smiles(smi)
            if canon is None:
                invalid += 1
            canonical.append(canon)
        if invalid:
            logger.warning("Dropping %d rows mit invaliden SMILES von %s dataset", invalid, name)
        cleaned = df.copy()
        cleaned["smiles"] = canonical
        cleaned = cleaned.dropna(subset=["smiles"])
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        dropped = before - len(cleaned)
        if dropped:
            logger.info("entfernt %d duplicate SMILES von %s dataset", dropped, name)
        return cleaned

    def _filter_pool_overlaps(self) -> None:
        if "smiles" not in self.pool.columns or self.pool.empty:
            return
        if "smiles" not in self.labelled.columns:
            return
        known = set(self.labelled["smiles"]).union(self._excluded_smiles)
        if not known:
            return
        before = len(self.pool)
        self.pool = self.pool[~self.pool["smiles"].isin(known)].reset_index(drop=True)
        removed = before - len(self.pool)
        if removed:
            logger.info(
                "entfernt %d pool entries schon enthalten in labelled/excluded datasets",
                removed,
            )

    def _passes_diversity(self, smiles: str) -> bool:
        if self.diversity_threshold <= 0 or not RDKit_AVAILABLE:
            return True
        fp = self._fingerprint(smiles)
        if fp is None:
            return False
        if not self._fingerprints:
            self._fingerprints.append(fp)
            return True
        sims = [DataStructs.TanimotoSimilarity(fp, existing) for existing in self._fingerprints]
        if sims and max(sims) >= self.diversity_threshold:
            return False
        self._fingerprints.append(fp)
        return True

    def _has_conjugation(self, mol: "Chem.Mol") -> bool:
        """Basic OSC filter, checkt fuer conjugated path length"""
        if mol is None:
            return False
        min_len = int(getattr(self.config, "min_conjugated_bonds", 0) or 0)
        if min_len <= 0:
            return True
        return self._longest_conjugated_path(mol) >= min_len

    def _count_aromatic_rings(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        rings = Chem.GetSymmSSSR(mol)
        aromatic = 0
        for ring in rings:
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic += 1
        return aromatic

    def _pi_conjugated_fraction(self, mol: "Chem.Mol") -> float:
        if mol is None:
            return 0.0
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not atoms:
            return 0.0
        conj_atoms = 0
        for atom in atoms:
            if atom.GetIsAromatic():
                conj_atoms += 1
                continue
            for bond in atom.GetBonds():
                if bond.GetIsConjugated() or bond.GetIsAromatic():
                    conj_atoms += 1
                    break
        return conj_atoms / len(atoms)

    def _conjugated_atom_indices(self, mol: "Chem.Mol") -> set[int]:
        if mol is None:
            return set()
        conj_atoms: set[int] = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1 and atom.GetIsAromatic():
                conj_atoms.add(atom.GetIdx())
        for bond in mol.GetBonds():
            if bond.GetIsConjugated() or bond.GetIsAromatic():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetAtomicNum() > 1:
                    conj_atoms.add(a1.GetIdx())
                if a2.GetAtomicNum() > 1:
                    conj_atoms.add(a2.GetIdx())
        return conj_atoms

    def _is_rotatable_bond(self, bond: "Chem.Bond") -> bool:
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        if bond.IsInRing() or bond.GetIsAromatic():
            return False
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
            return False
        # exclude terminal heavy-atom bonds
        if sum(1 for n in a1.GetNeighbors() if n.GetAtomicNum() > 1) <= 1:
            return False
        if sum(1 for n in a2.GetNeighbors() if n.GetAtomicNum() > 1) <= 1:
            return False
        return True

    def _rotatable_bonds_in_conjugated(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        conj_atoms = self._conjugated_atom_indices(mol)
        if not conj_atoms:
            return 0
        count = 0
        for bond in mol.GetBonds():
            if not (bond.GetIsConjugated() or bond.GetIsAromatic()):
                continue
            if not self._is_rotatable_bond(bond):
                continue
            if bond.GetBeginAtomIdx() in conj_atoms and bond.GetEndAtomIdx() in conj_atoms:
                count += 1
        return count

    def _charged_atom_count(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)

    def _branch_points(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
            if heavy_neighbors >= 3:
                count += 1
        return count

    def _max_heavy_degree(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        max_degree = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
            if heavy_neighbors > max_degree:
                max_degree = heavy_neighbors
        return max_degree

    def _longest_conjugated_path(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        if not heavy:
            return 0
        adjacency: Dict[int, List[int]] = {idx: [] for idx in heavy}
        for bond in mol.GetBonds():
            if not (bond.GetIsConjugated() or bond.GetIsAromatic()):
                continue
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 not in heavy or a2 not in heavy:
                continue
            adjacency[a1].append(a2)
            adjacency[a2].append(a1)
        if not any(adjacency.values()):
            return 0
        max_len = 0
        for start in adjacency:
            if not adjacency[start]:
                continue
            distances = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adjacency.get(node, []):
                    if nb in distances:
                        continue
                    distances[nb] = distances[node] + 1
                    queue.append(nb)
            if distances:
                max_len = max(max_len, max(distances.values()))
        return max_len

    def _murcko_scaffold(self, mol: "Chem.Mol") -> Optional[str]:
        if mol is None:
            return None
        try:
            return rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
        except Exception:
            return None

    def _physchem_ok(self, mol: "Chem.Mol") -> bool:
        """Checkt lightweight physchem/processability windows"""

        if mol is None or not RDKit_AVAILABLE:
            return False
        cfg = getattr(self.config, "physchem_filters", {}) or {}

        def _in_range(val: float, bounds: Sequence[float]) -> bool:
            if not bounds or len(bounds) != 2:
                return True
            lo, hi = bounds
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
            return True

        # TPSA, logP, HBA/HBD, fractionCSP3
        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            clogp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        except Exception as exc:
            logger.debug("Physchem calc failed: %s", exc)
            return False

        if not _in_range(tpsa, cfg.get("tpsa", ())):
            return False
        if not _in_range(clogp, cfg.get("clogp", ())):
            return False
        if not _in_range(hbd, cfg.get("hbd", ())):
            return False
        if not _in_range(hba, cfg.get("hba", ())):
            return False
        if not _in_range(frac_csp3, cfg.get("frac_csp3", ())):
            return False
        return True

    def _has_alternating_conjugation(self, mol: "Chem.Mol", min_bonds: int) -> bool:
        """Checkt für einen conjugated path mit alternating single/double bonds"""
        if mol is None or min_bonds <= 0:
            return False

        aromatic_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetIsAromatic()}
        # Kekulize a copy so aromatic systems become an explicit single/double pattern.
        try:
            work = Chem.Mol(mol)
            Chem.Kekulize(work, clearAromaticFlags=True)
        except Exception:
            work = mol

        conj_bonds = []
        for bond in work.GetBonds():
            if bond.GetIdx() in aromatic_bonds:
                continue
            if not bond.GetIsConjugated():
                continue
            btype = bond.GetBondType()
            if btype not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
                continue
            conj_bonds.append((bond.GetIdx(), btype, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if not conj_bonds:
            return False

        bond_adj: Dict[int, List[int]] = {idx: [] for idx, *_ in conj_bonds}
        for idx, _, a1, a2 in conj_bonds:
            for jdx, _, b1, b2 in conj_bonds:
                if idx == jdx:
                    continue
                if a1 in (b1, b2) or a2 in (b1, b2):
                    bond_adj[idx].append(jdx)

        classes = {idx: ("S" if btype == Chem.BondType.SINGLE else "D") for idx, btype, _, _ in conj_bonds}

        for start_idx, start_class in classes.items():
            stack = [(start_idx, start_class, 1)]
            visited = set()
            while stack:
                bond_idx, cls, length = stack.pop()
                if length >= min_bonds:
                    return True
                visited.add(bond_idx)
                for nb in bond_adj.get(bond_idx, []):
                    ncls = classes.get(nb)
                    if ncls is None or ncls == cls or nb in visited:
                        continue
                    stack.append((nb, ncls, length + 1))
        return False

    def _compute_median_smiles_len(self) -> Optional[float]:
        lengths = []
        if "smiles" in self.labelled:
            lengths.extend(self.labelled["smiles"].dropna().astype(str).str.len().tolist())
        if "smiles" in self.pool:
            lengths.extend(self.pool["smiles"].dropna().astype(str).str.len().tolist())
        if not lengths:
            return None
        lengths.sort()
        mid = len(lengths) // 2
        if len(lengths) % 2 == 0:
            return (lengths[mid - 1] + lengths[mid]) / 2.0
        return float(lengths[mid])

    def _build_graph(self, smiles: str, row: Optional[pd.Series] = None):
        """Featurize generated structures in SchNet 3D format."""
        mol_block = None
        if row is not None:
            try:
                mol_block = row.get("mol", None)
            except Exception:
                mol_block = getattr(row, "mol", None)
        cache_key = f"schnet::{smiles}::{hash(mol_block) if mol_block is not None else ''}"
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            return cached.clone() if hasattr(cached, "clone") else cached
        try:
            data = molblock_to_data(mol_block or "", smiles=smiles)
            self._store_graph_cache(cache_key, data)
            return data
        except Exception as exc:
            logger.debug("SchNet featurization failed fuer %s: %s", smiles, exc)
            return None

    def _store_graph_cache(self, key: str, data) -> None:
        try:
            self._graph_cache[key] = data.clone() if hasattr(data, "clone") else data
            self._graph_cache_keys.append(key)
            if len(self._graph_cache_keys) > self._graph_cache_cap:
                # evict oldest
                old_key = self._graph_cache_keys.popleft()
                self._graph_cache.pop(old_key, None)
        except Exception:
            logger.debug("Graph cache store failed for key %s", key, exc_info=True)

    def _passes_property_filters(self, smiles: str) -> bool:
        """benutzt surrogate predictions um property ranges auf generated SMILES zu enforcen"""
        if not self._filter_indices:
            return True
        graph = self._build_graph(smiles)
        if graph is None:
            logger.debug("Property filter: failed to featurize %s", smiles)
            return False
        mean, std, _ = self.surrogate.predict([graph], batch_size=1)
        for prop, idx in self._filter_indices.items():
            vmin, vmax = self.config.property_filters.get(prop, (None, None))
            if vmin is None or vmax is None:
                continue
            val = float(mean[0, idx])
            if val < vmin or val > vmax:
                return False
        return True

    def _normalise_predictions(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_targets = mean.shape[1]
        directions = np.array([1.0 if maximise else -1.0 for maximise in self.config.maximise])
        mus = np.zeros(n_targets)
        sigmas = np.ones(n_targets)
        for i, target in enumerate(self.config.target_columns):
            values = self.labelled[target].dropna().to_numpy(dtype=float)
            if values.size >= 2:
                oriented = values * directions[i]
                mus[i] = oriented.mean()
                sigma = oriented.std()
                sigmas[i] = sigma if sigma > 1e-6 else 1.0
            elif values.size == 1:
                mus[i] = values[0] * directions[i]
                sigmas[i] = 1.0
            else:
                mus[i] = 0.0
                sigmas[i] = 1.0
        mean_norm = ((mean * directions) - mus) / sigmas
        std_norm = std / sigmas
        return mean_norm, std_norm, mus, sigmas, directions

    def _save_diagnostics(self, pool_slice: pd.DataFrame, iteration: int) -> None:
        if not bool(getattr(self.config, "save_diagnostics", False)):
            return
        every = int(getattr(self.config, "diagnostics_every", 0) or 0)
        if every <= 0 or (iteration % every) != 0:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("Matplotlib nicht available; kein diagnostics plot fuer iteration %d.", iteration)
            return
        max_points = int(getattr(self.config, "diagnostics_max_points", 0) or 0)
        if max_points > 0 and len(pool_slice) > max_points:
            pool_slice = pool_slice.sample(n=max_points, random_state=int(iteration), replace=False)
        diag_dir = self.results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        n_targets = len(self.config.target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(6, 3 * n_targets), squeeze=False)
        for idx, target in enumerate(self.config.target_columns):
            ax = axes[idx, 0]
            ax.scatter(
                pool_slice[f"pred_{target}"],
                pool_slice["acquisition_score"],
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel(f"Predicted {target}")
            ax.set_ylabel("Acquisition")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Acquisition diagnostics (iteration {iteration})")
        fig.tight_layout()
        fig.savefig(diag_dir / f"diag_iter_{iteration:03d}.png", dpi=150)
        plt.close(fig)

    def _refresh_generator(self) -> None:
        if self.generator is None or not self.fragment_vocab:
            return
        if len(self.labelled) < 5:
            return
        try:
            from src.data.jt_preprocess import JTPreprocessConfig, prepare_jtvae_examples
            from src.models.jtvae_extended import JTVDataset, train_jtvae
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Skippen von generator refresh: preprocessing utilities nicht da (%s).", exc)
            return
        df = self.labelled.dropna(subset=["smiles", *self.config.target_columns])
        if df.empty:
            return
        df = df[["smiles", *self.config.target_columns]].drop_duplicates(subset="smiles")
        config = JTPreprocessConfig(
            max_fragments=getattr(self.generator, "max_tree_nodes", 12),
            condition_columns=self.config.target_columns,
        )
        max_heavy_atoms = self.generator_refresh_kwargs.get("max_heavy_atoms")
        if max_heavy_atoms is None:
            max_heavy_atoms = getattr(self.config, "max_generated_heavy_atoms", None)
        try:
            examples = prepare_jtvae_examples(
                df,
                self.fragment_vocab,
                config=config,
                max_heavy_atoms=max_heavy_atoms,
            )
        except Exception as exc:
            logger.warning("Failed prepare JT-VAE examples für refresh: %s", exc)
            return
        dataset = JTVDataset(examples)
        if len(dataset) == 0:
            logger.debug("Generator refresh skipped: no valid examples.")
            return
        refresh_cfg = {
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-4,
            "kl_weight": 0.5,
            "property_weight": 0.0,
            "adj_weight": 1.0,
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
            "save_dir": self.results_dir / "generator_refresh",
            "cond_stats": config.condition_stats,
        }
        refresh_cfg.update(self.generator_refresh_kwargs)
        refresh_cfg.pop("max_heavy_atoms", None)
        refresh_cfg["epochs"] = int(refresh_cfg.get("epochs", 1))
        refresh_cfg["batch_size"] = int(refresh_cfg.get("batch_size", 16))
        refresh_cfg["lr"] = float(refresh_cfg.get("lr", 1e-4))
        refresh_cfg["kl_weight"] = float(refresh_cfg.get("kl_weight", 0.5))
        refresh_cfg["property_weight"] = float(refresh_cfg.get("property_weight", 0.0))
        refresh_cfg["adj_weight"] = float(refresh_cfg.get("adj_weight", 1.0))
        device = next(self.generator.parameters()).device
        save_dir = Path(refresh_cfg.pop("save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Refreshing generator on %d molecules fuer %d epochs (lr=%s)",
            len(dataset),
            refresh_cfg.get("epochs", 1),
            refresh_cfg.get("lr", 1e-4),
        )
        train_jtvae(
            self.generator,
            dataset,
            self.fragment_vocab,
            device=str(device),
            save_dir=str(save_dir),
            start_epoch=1,
            **refresh_cfg,
        )

    def _ensure_pool(self, min_size: int, cond: Optional[np.ndarray], assemble_kwargs: Optional[Dict]) -> int:
        if self.generator is None:
            return 0
        generated = 0
        rejected = {"invalid": 0, "duplicate": 0, "filtered": 0}
        max_attempts = max(1, int(getattr(self.config, "generator_attempts", 5)))
        existing = set(self.pool["smiles"]).union(set(self.labelled["smiles"])).union(self._excluded_smiles)
        # dynamic length cap based on dataset median if explicit cap not set
        effective_len_cap = self.config.max_generated_smiles_len
        if self._median_smiles_len is not None:
            factor = getattr(self.config, "generated_smiles_len_factor", None)
            if factor is not None and factor > 0:
                dynamic_cap = int(self._median_smiles_len * factor)
                if effective_len_cap is None:
                    effective_len_cap = dynamic_cap
                    logger.info(
                        "Using dynamic SMILES length cap: median %.1f * %.2f -> %d",
                        self._median_smiles_len,
                        factor,
                        effective_len_cap,
                    )
                else:
                    capped = min(effective_len_cap, dynamic_cap)
                    if capped != effective_len_cap:
                        logger.info(
                            "Tightening SMILES length cap: min(config=%d, dynamic=%d) -> %d",
                            effective_len_cap,
                            dynamic_cap,
                            capped,
                        )
                        effective_len_cap = capped
        attempts = 0
        relaxed_used = False
        use_relaxed = False
        skip_property_filters = False
        skip_diversity = False
        relax_structural = False
        while len(self.pool) < min_size and attempts < max_attempts:
            attempts += 1
            assemble_kwargs_current = assemble_kwargs or self.assemble_kwargs
            if use_relaxed and self.generator is not None:
                assemble_kwargs_current = dict(assemble_kwargs_current)
                assemble_kwargs_current["adjacency_threshold"] = min(
                    assemble_kwargs_current.get("adjacency_threshold", 0.7), 0.4
                )
                assemble_kwargs_current["max_tree_nodes"] = min(
                    assemble_kwargs_current.get("max_tree_nodes", 8) or 8, 6
                )
                assemble_kwargs_current["beam_width"] = max(assemble_kwargs_current.get("beam_width", 3), 5)
            if self.pool.empty and self.generator is not None:
                assemble_kwargs_current = dict(assemble_kwargs_current)
                base_nodes = int(assemble_kwargs_current.get("max_tree_nodes", 12) or 12)
                base_beam = int(assemble_kwargs_current.get("beam_width", 5) or 5)
                base_topk = int(assemble_kwargs_current.get("topk_per_node", 5) or 5)
                capped_nodes = min(base_nodes, 10)
                capped_beam = min(base_beam, 4)
                capped_topk = min(base_topk, 4)
                if (capped_nodes, capped_beam, capped_topk) != (base_nodes, base_beam, base_topk):
                    logger.info(
                        "Pool leer -> fast refill caps: max_tree_nodes=%d, beam_width=%d, topk_per_node=%d",
                        capped_nodes,
                        capped_beam,
                        capped_topk,
                    )
                assemble_kwargs_current["max_tree_nodes"] = capped_nodes
                assemble_kwargs_current["beam_width"] = capped_beam
                assemble_kwargs_current["topk_per_node"] = capped_topk
                # Avoid expensive partial assembly scoring in beam search during pool refill.
                assemble_kwargs_current.setdefault("score_partial_assembly", False)
            logger.info(
                "Generator refill attempt %d/%d (pool=%d/%d, n_samples=%d)",
                attempts,
                max_attempts,
                len(self.pool),
                min_size,
                int(self.config.generator_samples),
            )
            samples = []
            if self.generator is not None and self.fragment_vocab:
                samples = sample_conditional(
                    self.generator,
                    self.fragment_vocab,
                    cond=cond,
                    n_samples=self.config.generator_samples,
                    temperature=getattr(self.config, "generator_temperature", 1.0),
                    assembler="beam",
                    assemble_kwargs=assemble_kwargs_current,
                    device=self.generator_device,
                )
                logger.info("Generator returned %d raw candidates in attempt %d.", len(samples), attempts)
            new_rows = []
            for sample in samples:
                smiles = sample.get("smiles")
                status = sample.get("status")
                if not smiles:
                    rejected["invalid"] += 1
                    continue
                smiles = self._canonical_smiles(smiles)
                if not smiles:
                    rejected["invalid"] += 1
                    continue
                if smiles in existing:
                    rejected["duplicate"] += 1
                    continue
                if effective_len_cap and len(smiles) > effective_len_cap:
                    logger.debug(
                        "Skipping generated SMILES (len %d > cap %d): %s",
                        len(smiles),
                        effective_len_cap,
                        smiles,
                    )
                    rejected["filtered"] += 1
                    continue
                mol = self._parse_smiles(smiles) if RDKit_AVAILABLE else None
                if mol is not None and RDKit_AVAILABLE:
                    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    if len(frags) > 1:
                        # keept den largest connected component um disconnected assemblies zu vermeiden
                        frags_sorted = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
                        mol = frags_sorted[0]
                        smiles_new = Chem.MolToSmiles(mol, isomericSmiles=True)
                        if smiles_new != smiles:
                            smiles = smiles_new
                            if smiles in existing:
                                rejected["duplicate"] += 1
                                continue
                            if effective_len_cap and len(smiles) > effective_len_cap:
                                rejected["filtered"] += 1
                                continue
                heavy_atoms = None
                if RDKit_AVAILABLE:
                    if mol is None:
                        logger.debug("Skipping invalid generated SMILES: %s", smiles)
                        rejected["invalid"] += 1
                        continue
                    heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                    if self.config.require_neutral and Chem.GetFormalCharge(mol) != 0:
                        logger.debug(
                            "Skipping charged generated SMILES (charge %d): %s",
                            Chem.GetFormalCharge(mol),
                            smiles,
                        )
                        rejected["filtered"] += 1
                        continue
                    if getattr(self.config, "max_charged_atoms", None) is not None:
                        charged_atoms = self._charged_atom_count(mol)
                        if charged_atoms > int(self.config.max_charged_atoms):
                            logger.debug(
                                "Skipping generated SMILES (charged atoms %d > %d): %s",
                                charged_atoms,
                                self.config.max_charged_atoms,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if (
                        self.config.max_generated_heavy_atoms is not None
                        and heavy_atoms is not None
                        and heavy_atoms > self.config.max_generated_heavy_atoms
                    ):
                        logger.debug(
                            "Skipping generated SMILES (heavy atoms %d > %d): %s",
                            heavy_atoms,
                            self.config.max_generated_heavy_atoms,
                            smiles,
                        )
                        rejected["filtered"] += 1
                        continue
                    if self.config.require_conjugation and not self._has_conjugation(mol):
                        logger.debug("Skipping generated SMILES (failt conjugation filter): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if (
                        not relax_structural
                        and getattr(self.config, "min_pi_conjugated_fraction", None) is not None
                        and self._pi_conjugated_fraction(mol) < float(self.config.min_pi_conjugated_fraction)
                    ):
                        logger.debug("Skipping generated SMILES (pi-conjugated fraction unter threshold): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if (
                        (self.config.min_alternating_conjugated_bonds and not relax_structural)
                        and not self._has_alternating_conjugation(
                            mol, min_bonds=self.config.min_alternating_conjugated_bonds
                        )
                    ):
                        logger.debug(
                            "Skipping generated SMILES (fails alternating conjugation filter): %s", smiles
                        )
                        rejected["filtered"] += 1
                        continue
                    if self.config.min_aromatic_rings:
                        aromatic_rings = self._count_aromatic_rings(mol)
                        if aromatic_rings < self.config.min_aromatic_rings:
                            logger.debug(
                                "Skipping generated SMILES (aromatic rings %d < %d): %s",
                                aromatic_rings,
                                self.config.min_aromatic_rings,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_branch_points", None) is not None and not relax_structural:
                        branches = self._branch_points(mol)
                        if branches > int(self.config.max_branch_points):
                            logger.debug(
                                "Skipping generated SMILES (branch points %d > %d): %s",
                                branches,
                                self.config.max_branch_points,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_branch_degree", None) is not None and not relax_structural:
                        max_deg = self._max_heavy_degree(mol)
                        if max_deg > int(self.config.max_branch_degree):
                            logger.debug(
                                "Skipping generated SMILES (max heavy degree %d > %d): %s",
                                max_deg,
                                self.config.max_branch_degree,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if self.config.max_rotatable_bonds is not None:
                        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                        if rotb > self.config.max_rotatable_bonds:
                            logger.debug(
                                "Skipping generated SMILES (rotatable bonds %d > %d): %s",
                                rotb,
                                self.config.max_rotatable_bonds,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_rotatable_bonds_conjugated", None) is not None:
                        rotb_conj = self._rotatable_bonds_in_conjugated(mol)
                        if rotb_conj > int(self.config.max_rotatable_bonds_conjugated):
                            logger.debug(
                                "Skipping generated SMILES (rotatable bonds in conjugated core %d > %d): %s",
                                rotb_conj,
                                self.config.max_rotatable_bonds_conjugated,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                if mol is not None:
                    if self.config.physchem_filters and not self._physchem_ok(mol):
                        logger.debug("Skipping generated SMILES (fails physchem filters): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if getattr(self.config, "sa_score_max", None) is not None and _sa_score is not None:
                        sa = _sa_score(mol)
                        if sa > float(self.config.sa_score_max):
                            logger.debug("Skipping generated SMILES (SA %.2f > %.2f): %s", sa, self.config.sa_score_max, smiles)
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "scaffold_unique", False):
                        scaffold = self._murcko_scaffold(mol)
                        if scaffold and scaffold in self._scaffolds_seen:
                            logger.debug("Skipping generated SMILES (duplicate scaffold): %s", smiles)
                            rejected["duplicate"] += 1
                            continue
                        if scaffold:
                            self._scaffolds_seen.add(scaffold)
                if mol is None and RDKit_AVAILABLE:
                    rejected["invalid"] += 1
                    continue
                if not skip_diversity and not self._passes_diversity(smiles):
                    logger.debug("Filtered out %s wegen diversity threshold.", smiles)
                    rejected["filtered"] += 1
                    continue
                if self._filter_indices and not skip_property_filters and not self._passes_property_filters(smiles):
                    logger.debug("Skipping generated SMILES (failt property filters): %s", smiles)
                    rejected["filtered"] += 1
                    continue
                existing.add(smiles)
                new_rows.append({"smiles": smiles, "assembly_status": status})
            if not new_rows:
                #  relaxed assembly/sample wenn nichts dazugekommen ist
                if not relaxed_used and attempts >= max(1, max_attempts // 2):
                    relaxed_used = True
                    use_relaxed = True
                    attempts -= 1  #zählt nicht als vollwertiger versuch
                    relaxed_adj = min((assemble_kwargs or self.assemble_kwargs).get("adjacency_threshold", 0.7), 0.4)
                    relaxed_nodes = min((assemble_kwargs or self.assemble_kwargs).get("max_tree_nodes", 8) or 8, 6)
                    relaxed_beam = max((assemble_kwargs or self.assemble_kwargs).get("beam_width", 3), 5)
                    logger.info(
                        "keine accepted samples bis jetzt; nochmal mit relaxed assembly (adjacency_threshold=%s, max_tree_nodes=%s, beam_width=%s).",
                        relaxed_adj,
                        relaxed_nodes,
                        relaxed_beam,
                    )
                    continue
                # wenn immernoch nichts accepted, dann property filter ausschalten
                if self._filter_indices and not skip_property_filters and attempts >= max_attempts:
                    logger.info("No candidates accepted with property filters; retrying once with property filters disabled.")
                    skip_property_filters = True
                    attempts = 0
                    continue
                # finaler structural relaxation fallback
                if (
                    getattr(self.config, "auto_relax_filters", False)
                    and not relax_structural
                    and attempts >= max_attempts
                    and generated == 0
                ):
                    relax_structural = True
                    skip_diversity = True
                    logger.info(
                        "Generation stalled;  structural relaxation (disable pi_fraction/branch caps, lower alternation requirement, skip diversity) und nochaml"
                    )
                    attempts = 0
                    continue
                continue
            use_relaxed = False
            self.pool = pd.concat([self.pool, pd.DataFrame(new_rows)], ignore_index=True)
            generated += len(new_rows)
            logger.info(
                "Accepted %d candidates in attempt %d (generated total=%d, pool now=%d).",
                len(new_rows),
                attempts,
                generated,
                len(self.pool),
            )
        if len(self.pool) < min_size:
            logger.warning(
                "Generator sampling exhausted after %d attempt(s): pool size %d (target %d). "
                "Generated %d new entries; rejected invalid=%d duplicate=%d filtered=%d. "
                "filter sind kacke oder unleeren seed pool geben",
                attempts,
                len(self.pool),
                min_size,
                generated,
                rejected["invalid"],
                rejected["duplicate"],
                rejected["filtered"],
            )
        return generated

    def _featurize_pool(self) -> List:
        graphs = []
        valid_indices = []
        invalid_indices = []
        for idx, row in self.pool.iterrows():
            smiles = row["smiles"]
            if RDKit_AVAILABLE:
                mol = self._parse_smiles(smiles)
                if mol is None:
                    logger.warning("Skipping invalid SMILES %s during featurization", smiles)
                    invalid_indices.append(idx)
                    continue
            data = self._build_graph(smiles, row=row)
            if data is None:
                logger.warning("Skipping invalid SMILES %s: featurization failed", smiles)
                invalid_indices.append(idx)
                continue
            graphs.append(data)
            valid_indices.append(idx)
        if invalid_indices:
            self.pool = self.pool.drop(index=invalid_indices).reset_index(drop=True)
            valid_indices = list(range(len(graphs)))
            logger.info(
                "Dropped %d invalid pool entries nach featurization; %d candidates remain.",
                len(invalid_indices),
                len(graphs),
            )
        return graphs, valid_indices

    def _predict_with_surrogate(
        self,
        surrogate_obj,
        graphs: List,
        *,
        batch_size: int,
        mc_samples: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, std, _ = surrogate_obj.predict(
            graphs,
            batch_size=batch_size,
            mc_samples=mc_samples,
        )
        return mean, std

    def _predict_pool(
        self,
        graphs: List,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        predict_batch_size = getattr(self.config, "predict_batch_size", None)
        if predict_batch_size is None:
            surrogate_cfg = getattr(self.surrogate, "config", None)
            predict_batch_size = getattr(surrogate_cfg, "batch_size", None)
        if predict_batch_size is None:
            predict_batch_size = self.config.batch_size
        predict_batch_size = max(1, int(predict_batch_size))

        predict_mc_samples = getattr(self.config, "predict_mc_samples", None)
        if predict_mc_samples is None:
            acq_kind = str(getattr(self.config.acquisition, "kind", "")).lower()
            acq_beta = float(getattr(self.config.acquisition, "beta", 0.0) or 0.0)
            # Fast-path: target acquisition with beta=0 ignores uncertainty.
            if acq_kind == "target" and abs(acq_beta) < 1e-12:
                predict_mc_samples = 1
        if predict_mc_samples is not None:
            predict_mc_samples = max(1, int(predict_mc_samples))

        logger.info(
            "Surrogate inference: evaluating %d graphs (batch_size=%d, mc_samples=%s).",
            len(graphs),
            predict_batch_size,
            predict_mc_samples if predict_mc_samples is not None else "default",
        )
        mean, std = self._predict_with_surrogate(
            self.surrogate,
            graphs,
            batch_size=predict_batch_size,
            mc_samples=predict_mc_samples,
        )
        logger.info("Surrogate inference completed: mean shape=%s std shape=%s", mean.shape, std.shape)
        if self._target_indices:
            mean = mean[:, self._target_indices]
            std = std[:, self._target_indices]
        logger.debug("Mapped surrogate outputs to %d targets mit indices %s.", mean.shape[1], self._target_indices)
        optical_mean: Optional[np.ndarray] = None
        optical_std: Optional[np.ndarray] = None
        if self.optical_surrogate is not None and self._optical_target_indices:
            opt_batch_size = getattr(self.config, "optical_predict_batch_size", None)
            if opt_batch_size is None:
                opt_batch_size = predict_batch_size
            opt_batch_size = max(1, int(opt_batch_size))

            opt_mc_samples = getattr(self.config, "optical_predict_mc_samples", None)
            if opt_mc_samples is None:
                opt_mc_samples = predict_mc_samples
            if opt_mc_samples is not None:
                opt_mc_samples = max(1, int(opt_mc_samples))

            logger.info(
                "Optical surrogate inference: evaluating %d graphs (batch_size=%d, mc_samples=%s).",
                len(graphs),
                opt_batch_size,
                opt_mc_samples if opt_mc_samples is not None else "default",
            )
            optical_mean, optical_std = self._predict_with_surrogate(
                self.optical_surrogate,
                graphs,
                batch_size=opt_batch_size,
                mc_samples=opt_mc_samples,
            )
            if self._optical_target_indices:
                optical_mean = optical_mean[:, self._optical_target_indices]
                optical_std = optical_std[:, self._optical_target_indices]
            logger.info(
                "Optical surrogate inference completed: mean shape=%s std shape=%s",
                optical_mean.shape,
                optical_std.shape,
            )
        oscillator_mean: Optional[np.ndarray] = None
        oscillator_std: Optional[np.ndarray] = None
        if self.oscillator_surrogate is not None and self._oscillator_target_indices:
            osc_batch_size = getattr(self.config, "oscillator_predict_batch_size", None)
            if osc_batch_size is None:
                osc_batch_size = predict_batch_size
            osc_batch_size = max(1, int(osc_batch_size))

            osc_mc_samples = getattr(self.config, "oscillator_predict_mc_samples", None)
            if osc_mc_samples is None:
                osc_mc_samples = predict_mc_samples
            if osc_mc_samples is not None:
                osc_mc_samples = max(1, int(osc_mc_samples))

            logger.info(
                "Oscillator surrogate inference: evaluating %d graphs (batch_size=%d, mc_samples=%s).",
                len(graphs),
                osc_batch_size,
                osc_mc_samples if osc_mc_samples is not None else "default",
            )
            oscillator_mean, oscillator_std = self._predict_with_surrogate(
                self.oscillator_surrogate,
                graphs,
                batch_size=osc_batch_size,
                mc_samples=osc_mc_samples,
            )
            if self._oscillator_target_indices:
                oscillator_mean = oscillator_mean[:, self._oscillator_target_indices]
                oscillator_std = oscillator_std[:, self._oscillator_target_indices]
            logger.info(
                "Oscillator surrogate inference completed: mean shape=%s std shape=%s",
                oscillator_mean.shape,
                oscillator_std.shape,
            )
        return mean, std, optical_mean, optical_std, oscillator_mean, oscillator_std

    def _score_candidates(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        norm_mean, norm_std, mus, sigmas, directions = self._normalise_predictions(mean, std)
        best = self._current_best()
        norm_best = None
        if best is not None:
            norm_best = ((best * directions) - mus) / sigmas
        acq_cfg = self.config.acquisition
        cfg = AcquisitionConfig(
            kind=acq_cfg.kind,
            beta=acq_cfg.beta,
            xi=acq_cfg.xi,
            maximise=acq_cfg.maximise,
            weights=acq_cfg.weights,
            targets=acq_cfg.targets,
            tolerances=acq_cfg.tolerances,
        )
        if cfg.kind in {"pareto", "pareto_ucb"}:
            cfg.maximise = [True] * norm_mean.shape[1]
        scores = acquisition_score(norm_mean, norm_std, cfg, best_so_far=norm_best)
        return scores

    def _score_optical_candidates(
        self,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        n_candidates: int,
    ) -> np.ndarray:
        if (
            mean is None
            or std is None
            or self.optical_surrogate is None
            or n_candidates <= 0
        ):
            return np.zeros(max(0, int(n_candidates)), dtype=float)
        if mean.shape[0] != n_candidates:
            raise ValueError(
                f"optical score candidate-count mismatch: expected {n_candidates}, got {mean.shape[0]}"
            )
        if self._optical_score_weight == 0.0:
            return np.zeros(n_candidates, dtype=float)
        if self._optical_targets is None:
            if not self._optical_warned_no_targets:
                logger.warning(
                    "optical_score_weight is set (%.3f) but optical_targets are missing; optical score disabled.",
                    self._optical_score_weight,
                )
                self._optical_warned_no_targets = True
            return np.zeros(n_candidates, dtype=float)

        targets = np.asarray(self._optical_targets, dtype=float)
        if targets.shape[0] != mean.shape[1]:
            raise ValueError(
                f"optical_targets length mismatch with optical surrogate outputs: {targets.shape[0]} vs {mean.shape[1]}"
            )
        diff = np.abs(mean - targets)

        if self._optical_tolerances is not None:
            tol = np.asarray(self._optical_tolerances, dtype=float)
            if tol.shape[0] != mean.shape[1]:
                raise ValueError(
                    f"optical_tolerances length mismatch with optical outputs: {tol.shape[0]} vs {mean.shape[1]}"
                )
            tol = np.where(tol <= 0.0, 1.0, tol)
            diff = diff / tol

        if self._optical_weights is not None:
            weights = np.asarray(self._optical_weights, dtype=float)
            if weights.shape[0] != mean.shape[1]:
                raise ValueError(
                    f"optical_weights length mismatch with optical outputs: {weights.shape[0]} vs {mean.shape[1]}"
                )
            diff = diff * weights

        score = -diff.sum(axis=1)
        if abs(self._optical_beta) > 1e-12:
            score = score + float(self._optical_beta) * std.mean(axis=1)
        return score

    def _score_oscillator_candidates(
        self,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        n_candidates: int,
    ) -> np.ndarray:
        if (
            mean is None
            or std is None
            or self.oscillator_surrogate is None
            or n_candidates <= 0
        ):
            return np.zeros(max(0, int(n_candidates)), dtype=float)
        if mean.shape[0] != n_candidates:
            raise ValueError(
                f"oscillator score candidate-count mismatch: expected {n_candidates}, got {mean.shape[0]}"
            )
        if self._oscillator_score_weight == 0.0:
            return np.zeros(n_candidates, dtype=float)
        if self._oscillator_targets is None:
            if not self._oscillator_warned_no_targets:
                logger.warning(
                    "oscillator_score_weight is set (%.3f) but oscillator_targets are missing; oscillator score disabled.",
                    self._oscillator_score_weight,
                )
                self._oscillator_warned_no_targets = True
            return np.zeros(n_candidates, dtype=float)

        targets = np.asarray(self._oscillator_targets, dtype=float)
        if targets.shape[0] != mean.shape[1]:
            raise ValueError(
                f"oscillator_targets length mismatch with oscillator surrogate outputs: {targets.shape[0]} vs {mean.shape[1]}"
            )
        diff = np.abs(mean - targets)

        if self._oscillator_tolerances is not None:
            tol = np.asarray(self._oscillator_tolerances, dtype=float)
            if tol.shape[0] != mean.shape[1]:
                raise ValueError(
                    f"oscillator_tolerances length mismatch with oscillator outputs: {tol.shape[0]} vs {mean.shape[1]}"
                )
            tol = np.where(tol <= 0.0, 1.0, tol)
            diff = diff / tol

        if self._oscillator_weights is not None:
            weights = np.asarray(self._oscillator_weights, dtype=float)
            if weights.shape[0] != mean.shape[1]:
                raise ValueError(
                    f"oscillator_weights length mismatch with oscillator outputs: {weights.shape[0]} vs {mean.shape[1]}"
                )
            diff = diff * weights

        score = -diff.sum(axis=1)
        if abs(self._oscillator_beta) > 1e-12:
            score = score + float(self._oscillator_beta) * std.mean(axis=1)
        return score

    def _label_with_dft(self, selected: pd.DataFrame) -> pd.DataFrame:
        if self.dft is None:
            return selected
        jobs = [
            DFTJobSpec(smiles=row["smiles"], properties=self._qc_requested_properties, **self._dft_job_defaults)
            for _, row in selected.iterrows()
        ]
        ids = self.dft.submit_batch(jobs)
        results = []
        for job_id in ids:
            res = self.dft.fetch(job_id, block=True, poll_interval=1.0)
            results.append(res)
        for df_row, res in zip(selected.itertuples(index=True), results):
            if res is None:
                continue
            if res.status != "success":
                if res.error_message:
                    logger.warning(
                        "QC job %s returned status %s: %s",
                        res.job.job_id,
                        res.status,
                        res.error_message,
                    )
                else:
                    logger.warning("QC job %s returned status %s", res.job.job_id, res.status)
            self._apply_result(df_row.Index, selected, res)
            selected.at[df_row.Index, "qc_status"] = res.status
            selected.at[df_row.Index, "qc_wall_time"] = res.wall_time
            selected.at[df_row.Index, "qc_error"] = res.error_message
            if res.metadata:
                for meta_key in ("total_energy", "basis"):
                    if meta_key in res.metadata:
                        selected.at[df_row.Index, meta_key] = res.metadata[meta_key]
                selected.at[df_row.Index, "qc_metadata"] = json.dumps(res.metadata, ensure_ascii=False)
            self._annotate_optical_gate(df_row.Index, selected)
        if "red_absorber_gate" in selected.columns:
            passed = int(selected["red_absorber_gate"].fillna(False).sum())
            logger.info(
                "Red-absorber gate: %d/%d passed (lambda_max_nm >= %s, oscillator_strength >= %s).",
                passed,
                len(selected),
                self._min_lambda_max_nm if self._min_lambda_max_nm is not None else "n/a",
                self._min_oscillator_strength if self._min_oscillator_strength is not None else "n/a",
            )
        return selected

    def _apply_result(self, row_index: int, frame: pd.DataFrame, result: DFTResult) -> Dict[str, float]:
        mapped = {}
        for prop, value in result.properties.items():
            column = self.property_aliases.get(prop)
            if column is None and prop.endswith("_eV"):
                column = self.property_aliases.get(prop[:-3])
            if column is None and prop.endswith("_nm"):
                column = self.property_aliases.get(prop[:-3])
            if column is None:
                column = prop
            frame.at[row_index, column] = value
            mapped[column] = value
        return mapped

    def _annotate_optical_gate(self, row_index: int, frame: pd.DataFrame) -> None:
        if self._min_lambda_max_nm is None and self._min_oscillator_strength is None:
            return
        lambda_val = frame.at[row_index, "lambda_max_nm"] if "lambda_max_nm" in frame.columns else np.nan
        if pd.isna(lambda_val) and "lambda_max" in frame.columns:
            lambda_val = frame.at[row_index, "lambda_max"]
        osc_val = (
            frame.at[row_index, "oscillator_strength"] if "oscillator_strength" in frame.columns else np.nan
        )
        passed = True
        reason: List[str] = []
        if self._min_lambda_max_nm is not None:
            if pd.isna(lambda_val):
                passed = False
                reason.append("missing_lambda_max_nm")
            elif float(lambda_val) < self._min_lambda_max_nm:
                passed = False
                reason.append(f"lambda<{self._min_lambda_max_nm}")
        if self._min_oscillator_strength is not None:
            if pd.isna(osc_val):
                passed = False
                reason.append("missing_oscillator_strength")
            elif float(osc_val) < self._min_oscillator_strength:
                passed = False
                reason.append(f"f<{self._min_oscillator_strength}")
        frame.at[row_index, "red_absorber_gate"] = bool(passed)
        frame.at[row_index, "red_absorber_reason"] = "ok" if passed else ";".join(reason)

    def _retrain_surrogate(self) -> None:
        if len(self.labelled) < len(self.config.target_columns) + 5:
            return
        cols = ["smiles", *self.config.target_columns]
        if "mol" in self.labelled.columns:
            cols = ["mol"] + cols
        train_df = self.labelled[cols].dropna()
        if train_df.empty:
            return
        split = split_dataframe(train_df, val_fraction=0.1, test_fraction=0.0, seed=self.scheduler.iteration + 42)
        logger.info("Retraining surrogate on %d molecules", len(split.train))
        self.surrogate.fit(split.train, split.val)
        self._refresh_target_indices()

    def _retrain_optical_surrogate(self) -> None:
        if self.optical_surrogate is None:
            return
        if not hasattr(self.optical_surrogate, "fit"):
            logger.warning("Optical surrogate has no fit() method; skipping optical retrain.")
            return
        if not self._optical_target_columns:
            return
        if self._optical_retrain_every <= 0:
            return
        if self.scheduler.iteration % self._optical_retrain_every != 0:
            return

        target_cols = [col for col in self._optical_target_columns if col in self.labelled.columns]
        if len(target_cols) != len(self._optical_target_columns):
            logger.info(
                "Optical retrain skipped: missing target columns in labelled set (%s).",
                self._optical_target_columns,
            )
            return
        cols = ["smiles", *target_cols]
        if "mol" in self.labelled.columns:
            cols = ["mol"] + cols
        train_df = self.labelled[cols].copy()
        if self._optical_retrain_on_success_only and "qc_status" in self.labelled.columns:
            status = self.labelled["qc_status"].astype(str).str.lower()
            train_df = train_df.loc[status == "success"].copy()
        train_df = train_df.dropna(subset=target_cols)
        if len(train_df) < self._optical_retrain_min_labels:
            logger.info(
                "Optical retrain skipped: %d labels available (min=%d).",
                len(train_df),
                self._optical_retrain_min_labels,
            )
            return

        val_fraction = float(self._optical_retrain_val_fraction)
        val_fraction = min(max(val_fraction, 0.0), 0.4)
        split = split_dataframe(
            train_df,
            val_fraction=val_fraction,
            test_fraction=0.0,
            seed=self.scheduler.iteration + 4242,
        )
        logger.info(
            "Retraining optical surrogate on %d molecules (val=%d).",
            len(split.train),
            len(split.val),
        )
        self.optical_surrogate.fit(split.train, split.val if len(split.val) > 0 else None)
        self._refresh_optical_target_indices()

    def _retrain_oscillator_surrogate(self) -> None:
        if self.oscillator_surrogate is None:
            return
        if not hasattr(self.oscillator_surrogate, "fit"):
            logger.warning("Oscillator surrogate has no fit() method; skipping oscillator retrain.")
            return
        if not self._oscillator_target_columns:
            return
        if self._oscillator_retrain_every <= 0:
            return
        if self.scheduler.iteration % self._oscillator_retrain_every != 0:
            return

        target_cols = [col for col in self._oscillator_target_columns if col in self.labelled.columns]
        if len(target_cols) != len(self._oscillator_target_columns):
            logger.info(
                "Oscillator retrain skipped: missing target columns in labelled set (%s).",
                self._oscillator_target_columns,
            )
            return
        cols = ["smiles", *target_cols]
        if "mol" in self.labelled.columns:
            cols = ["mol"] + cols
        train_df = self.labelled[cols].copy()
        if self._oscillator_retrain_on_success_only and "qc_status" in self.labelled.columns:
            status = self.labelled["qc_status"].astype(str).str.lower()
            train_df = train_df.loc[status == "success"].copy()
        train_df = train_df.dropna(subset=target_cols)
        if len(train_df) < self._oscillator_retrain_min_labels:
            logger.info(
                "Oscillator retrain skipped: %d labels available (min=%d).",
                len(train_df),
                self._oscillator_retrain_min_labels,
            )
            return

        val_fraction = float(self._oscillator_retrain_val_fraction)
        val_fraction = min(max(val_fraction, 0.0), 0.4)
        split = split_dataframe(
            train_df,
            val_fraction=val_fraction,
            test_fraction=0.0,
            seed=self.scheduler.iteration + 8424,
        )
        logger.info(
            "Retraining oscillator surrogate on %d molecules (val=%d).",
            len(split.train),
            len(split.val),
        )
        self.oscillator_surrogate.fit(split.train, split.val if len(split.val) > 0 else None)
        self._refresh_oscillator_target_indices()

    @staticmethod
    def _merge_scalar(target: Dict[str, float], key: str, value: float) -> None:
        if not np.isfinite(value):
            return
        if key in target and np.isfinite(float(target[key])):
            target[key] = float((float(target[key]) + float(value)) * 0.5)
        else:
            target[key] = float(value)

    def _prediction_maps_for_samples(
        self,
        sample_count: int,
        sample_indices: Sequence[int],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        optical_mean: Optional[np.ndarray],
        optical_std: Optional[np.ndarray],
        oscillator_mean: Optional[np.ndarray],
        oscillator_std: Optional[np.ndarray],
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        preds: List[Dict[str, float]] = [dict() for _ in range(sample_count)]
        uncs: List[Dict[str, float]] = [dict() for _ in range(sample_count)]

        def _assign(
            arr_mean: Optional[np.ndarray],
            arr_std: Optional[np.ndarray],
            names: Sequence[str],
        ) -> None:
            if arr_mean is None or arr_std is None:
                return
            for local_idx, sample_idx in enumerate(sample_indices):
                for j, name in enumerate(names):
                    val = float(arr_mean[local_idx, j])
                    stdv = float(arr_std[local_idx, j])
                    self._merge_scalar(preds[sample_idx], str(name), val)
                    self._merge_scalar(uncs[sample_idx], str(name), stdv)

        _assign(mean, std, self.config.target_columns)
        _assign(optical_mean, optical_std, self._optical_target_columns)
        _assign(oscillator_mean, oscillator_std, self._oscillator_target_columns)
        return preds, uncs

    def _rl_current_entropy_weight(self) -> float:
        start = self._rl_entropy_weight_start
        end = self._rl_entropy_weight_end
        if start is None and end is None:
            return float(self._rl_entropy_weight)
        if start is None:
            start = float(self._rl_entropy_weight)
        if end is None:
            end = float(self._rl_entropy_weight)
        updates = max(1, int(self._rl_entropy_decay_updates))
        progress = float(np.clip(float(self._rl_updates) / float(updates), 0.0, 1.0))
        return float(start + (end - start) * progress)

    def _rl_normalize_rewards_running(self, rewards: np.ndarray) -> np.ndarray:
        arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        if not self._rl_reward_running_norm:
            return arr

        count = float(self._rl_reward_stats.get("count", 0.0))
        mean = float(self._rl_reward_stats.get("mean", 0.0))
        var = float(self._rl_reward_stats.get("var", 1.0))
        std = float(np.sqrt(max(var, self._rl_reward_norm_eps)))
        if count < 2.0:
            normed = arr.copy()
        else:
            normed = (arr - mean) / max(std, self._rl_reward_norm_eps)

        if self._rl_reward_norm_clip is not None:
            clip_val = abs(float(self._rl_reward_norm_clip))
            normed = np.clip(normed, -clip_val, clip_val)

        for value in arr.tolist():
            count += 1.0
            delta = float(value) - mean
            mean += delta / count
            delta2 = float(value) - mean
            var = ((count - 1.0) * var + delta * delta2) / max(count, 1.0)

        self._rl_reward_stats["count"] = float(count)
        self._rl_reward_stats["mean"] = float(mean)
        self._rl_reward_stats["var"] = float(max(var, self._rl_reward_norm_eps))
        return normed.astype(np.float32)

    def _estimate_diversity_reward(
        self,
        smiles: str,
        *,
        seen_fingerprints: Sequence[object],
        max_refs: int = 256,
    ) -> float:
        if not smiles or not RDKit_AVAILABLE:
            return 0.0
        fp = self._fingerprint(smiles)
        if fp is None:
            return 0.0
        refs: List[object] = []
        if self._fingerprints:
            if len(self._fingerprints) <= max_refs:
                refs.extend(self._fingerprints)
            else:
                idx = self._rng.choice(len(self._fingerprints), size=max_refs, replace=False)
                refs.extend([self._fingerprints[int(i)] for i in idx.tolist()])
        refs.extend(list(seen_fingerprints))
        if not refs:
            return 1.0
        sims: List[float] = []
        for other in refs:
            try:
                sims.append(float(DataStructs.TanimotoSimilarity(fp, other)))
            except Exception:
                continue
        if not sims:
            return 0.0
        max_sim = float(np.clip(max(sims), 0.0, 1.0))
        return float(1.0 - max_sim)

    def _rl_should_run(self) -> bool:
        if not self._rl_enabled:
            return False
        if self._rl_algorithm not in {"reinforce", "pg_reinforce", "policy_gradient", "pg", "actor_critic", "ppo"}:
            logger.warning("Unknown rl_algorithm=%s; skipping RL update.", self._rl_algorithm)
            return False
        if self.generator is None:
            return False
        if not hasattr(self.generator, "sample_with_trace"):
            logger.warning("RL enabled but generator lacks sample_with_trace(); skipping RL update.")
            return False
        if not self.fragment_vocab:
            logger.warning("RL enabled but fragment_vocab is empty; skipping RL update.")
            return False
        if self.scheduler.iteration < self._rl_warmup_iterations:
            return False
        if (self.scheduler.iteration % self._rl_every_n_iterations) != 0:
            return False
        return True

    def _apply_qc_reward_override(
        self,
        sample_rows: List[Dict[str, object]],
        rewards: np.ndarray,
        pred_maps: List[Dict[str, float]],
        unc_maps: List[Dict[str, float]],
    ) -> int:
        if not self._rl_use_qc_top_k or self.dft is None or self._rl_qc_top_k <= 0:
            return 0
        if rewards.size == 0:
            return 0
        ranked = np.argsort(rewards)[::-1]
        pick: List[int] = []
        for idx in ranked.tolist():
            smi = str(sample_rows[idx].get("smiles", "") or "")
            if not smi:
                continue
            pick.append(int(idx))
            if len(pick) >= int(self._rl_qc_top_k):
                break
        if not pick:
            return 0

        qc_rows: List[Dict[str, object]] = []
        for idx in pick:
            row = {"smiles": sample_rows[idx].get("smiles", ""), "sample_index": int(idx)}
            row.update({f"pred_{k}": v for k, v in pred_maps[idx].items()})
            qc_rows.append(row)
        qc_df = pd.DataFrame(qc_rows)
        qc_df = self._label_with_dft(qc_df)

        override_count = 0
        for _, qrow in qc_df.iterrows():
            try:
                sample_idx = int(qrow.get("sample_index"))
            except Exception:
                continue
            if sample_idx < 0 or sample_idx >= len(sample_rows):
                continue
            qc_status = str(qrow.get("qc_status", "")).lower()
            for col, value in qrow.items():
                if col in {"sample_index", "smiles", "qc_status", "qc_wall_time", "qc_error", "qc_metadata"}:
                    continue
                if pd.isna(value):
                    continue
                with contextlib.suppress(Exception):
                    v = float(value)
                    self._merge_scalar(pred_maps[sample_idx], str(col), v)
                    if str(col).endswith("_eV"):
                        self._merge_scalar(pred_maps[sample_idx], str(col)[:-3], v)
                    if str(col).endswith("_nm"):
                        self._merge_scalar(pred_maps[sample_idx], str(col), v)
            invalid = (qc_status != "success")
            reward, _ = compute_reward_from_objective_profile(
                pred_maps[sample_idx],
                objective_mode=self._objective_mode,
                objective_profile=self._objective_profile,
                uncertainties=unc_maps[sample_idx],
                invalid=invalid,
                duplicate=False,
                diversity=0.0,
                novel=not invalid,
                clip=self._rl_reward_clip,
            )
            rewards[sample_idx] = float(reward)
            override_count += 1
        return override_count

    def _save_rl_checkpoint(self, metrics: Dict[str, float]) -> None:
        if self.generator is None or not self._rl_enabled:
            return
        self._rl_dir.mkdir(parents=True, exist_ok=True)
        reward_mean = float(metrics.get("reward_mean", float("-inf")))
        improved = reward_mean > self._rl_best_reward_mean
        should_write_latest = (self._rl_updates % self._rl_checkpoint_every) == 0
        if not improved and not should_write_latest:
            return
        try:
            try:
                param_device = next(self.generator.parameters()).device
                device_spec = get_device(param_device)
            except Exception:
                device_spec = get_device("cpu")
            state_dict = ensure_state_dict_on_cpu(self.generator, device_spec)
            payload = {
                "state_dict": state_dict,
                "rl_metrics": dict(metrics),
                "rl_updates": int(self._rl_updates),
                "objective_mode": self._objective_mode,
                "baseline": dict(self._rl_baseline_state),
                "reward_stats": dict(self._rl_reward_stats),
                "ppo_clip_ratio": float(self._rl_ppo_clip_ratio),
                "actor_lr": self._rl_actor_lr,
                "critic_lr": self._rl_critic_lr,
            }
            latest_path = self._rl_dir / "jtvae_rl_latest.pt"
            if should_write_latest:
                torch.save(payload, latest_path)
            if improved:
                self._rl_best_reward_mean = reward_mean
                best_path = self._rl_dir / "jtvae_rl_best_reward.pt"
                torch.save(payload, best_path)
                (self._rl_dir / "jtvae_rl_best_reward.json").write_text(
                    json.dumps(
                        {
                            "reward_mean": reward_mean,
                            "rl_updates": int(self._rl_updates),
                            "objective_mode": self._objective_mode,
                            "metrics": metrics,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except Exception:
            logger.exception("Failed to save RL checkpoint.")

    def _rl_update_generator(
        self,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict[str, float]]:
        if not self._rl_should_run():
            return None
        if self.generator is None:
            return None

        idx_to_frag = {int(idx): frag for frag, idx in self.fragment_vocab.items()}
        existing = set(self.labelled.get("smiles", pd.Series(dtype=str)).dropna().astype(str).tolist())
        existing.update(self.pool.get("smiles", pd.Series(dtype=str)).dropna().astype(str).tolist())
        agg_metrics: List[Dict[str, float]] = []
        cfg_assemble = dict(assemble_kwargs or self.assemble_kwargs)
        qc_override_total = 0

        for step_idx in range(self._rl_steps_per_update):
            samples, trace = self.generator.sample_with_trace(
                n_samples=self._rl_batch_size,
                cond=cond,
                max_tree_nodes=int(cfg_assemble.get("max_tree_nodes", getattr(self.generator, "max_tree_nodes", 12))),
                fragment_idx_to_smiles=idx_to_frag,
                device=self.generator_device,
                assemble_kwargs=cfg_assemble,
                temperature=float(getattr(self.config, "generator_temperature", 1.0)),
            )
            if not samples or "log_prob" not in trace:
                continue
            sample_count = len(samples)
            graphs: List[Any] = []
            valid_idx: List[int] = []
            for idx, sample in enumerate(samples):
                smiles = str(sample.get("smiles", "") or "")
                if not smiles:
                    continue
                graph = self._build_graph(smiles)
                if graph is None:
                    continue
                graphs.append(graph)
                valid_idx.append(idx)

            mean = std = optical_mean = optical_std = oscillator_mean = oscillator_std = None
            if graphs:
                mean, std, optical_mean, optical_std, oscillator_mean, oscillator_std = self._predict_pool(graphs)
            pred_maps, unc_maps = self._prediction_maps_for_samples(
                sample_count,
                valid_idx,
                mean,
                std,
                optical_mean,
                optical_std,
                oscillator_mean,
                oscillator_std,
            )
            rewards_raw = np.zeros(sample_count, dtype=np.float32)
            anchor_weights = np.zeros(sample_count, dtype=np.float32)
            seen_batch: set[str] = set()
            seen_batch_fps: List[object] = []
            valid_count = 0
            for idx, sample in enumerate(samples):
                smiles = str(sample.get("smiles", "") or "")
                status = str(sample.get("status", "") or "")
                invalid = (not smiles) or status in {
                    "invalid_smiles",
                    "beam_empty",
                    "failed",
                    "no_vocab_mapping",
                }
                if not invalid and self._parse_smiles(smiles) is None:
                    invalid = True
                duplicate = False
                if smiles:
                    if smiles in seen_batch or smiles in existing:
                        duplicate = True
                    seen_batch.add(smiles)
                diversity = self._estimate_diversity_reward(
                    smiles,
                    seen_fingerprints=seen_batch_fps,
                )
                reward, _ = compute_reward_from_objective_profile(
                    pred_maps[idx],
                    objective_mode=self._objective_mode,
                    objective_profile=self._objective_profile,
                    uncertainties=unc_maps[idx],
                    invalid=invalid,
                    duplicate=duplicate,
                    diversity=diversity,
                    novel=(not invalid and not duplicate),
                    clip=self._rl_reward_clip,
                )
                rewards_raw[idx] = float(reward)
                if not invalid:
                    valid_count += 1
                if not invalid and not duplicate:
                    anchor_weights[idx] = 1.0
                    if float(reward) > 0.0:
                        anchor_weights[idx] += float(min(2.0, reward))
                if smiles:
                    fp = self._fingerprint(smiles)
                    if fp is not None:
                        seen_batch_fps.append(fp)

            qc_override_total += self._apply_qc_reward_override(samples, rewards_raw, pred_maps, unc_maps)
            rewards_norm = self._rl_normalize_rewards_running(rewards_raw)
            rewards_t = torch.tensor(rewards_norm, dtype=torch.float32, device=trace["log_prob"].device)
            anchor_t = torch.tensor(anchor_weights, dtype=torch.float32, device=trace["log_prob"].device)
            entropy_weight_now = self._rl_current_entropy_weight()
            metrics, self._rl_optimizer = train_jtvae_rl_step(
                self.generator,
                trace,
                rewards_t,
                optimizer=self._rl_optimizer,
                lr=self._rl_lr,
                algorithm=self._rl_algorithm,
                entropy_weight=entropy_weight_now,
                baseline_state=self._rl_baseline_state,
                baseline_momentum=self._rl_baseline_momentum,
                reward_clip=self._rl_reward_clip,
                normalize_advantage=self._rl_normalize_advantage,
                value_loss_weight=self._rl_value_loss_weight,
                actor_lr=self._rl_actor_lr,
                critic_lr=self._rl_critic_lr,
                ppo_clip_ratio=self._rl_ppo_clip_ratio,
                ppo_epochs=self._rl_ppo_epochs,
                ppo_minibatch_size=self._rl_ppo_minibatch_size,
                ppo_target_kl=self._rl_ppo_target_kl,
                ppo_value_clip_range=self._rl_ppo_value_clip_range,
                ppo_adaptive_kl=self._rl_ppo_adaptive_kl,
                ppo_adaptive_kl_high_mult=self._rl_ppo_adaptive_kl_high_mult,
                ppo_adaptive_kl_low_mult=self._rl_ppo_adaptive_kl_low_mult,
                ppo_lr_down_factor=self._rl_ppo_lr_down_factor,
                ppo_lr_up_factor=self._rl_ppo_lr_up_factor,
                ppo_clip_down_factor=self._rl_ppo_clip_down_factor,
                ppo_clip_up_factor=self._rl_ppo_clip_up_factor,
                ppo_actor_lr_min=self._rl_ppo_actor_lr_min,
                ppo_actor_lr_max=self._rl_ppo_actor_lr_max,
                ppo_clip_ratio_min=self._rl_ppo_clip_ratio_min,
                ppo_clip_ratio_max=self._rl_ppo_clip_ratio_max,
                anchor_weights=anchor_t,
                anchor_weight=self._rl_anchor_weight,
                max_grad_norm=self._rl_max_grad_norm,
            )
            if "next_ppo_clip_ratio" in metrics:
                with contextlib.suppress(Exception):
                    self._rl_ppo_clip_ratio = float(metrics["next_ppo_clip_ratio"])
            if "next_actor_lr" in metrics:
                with contextlib.suppress(Exception):
                    self._rl_actor_lr = float(metrics["next_actor_lr"])
                    self._rl_lr = float(metrics["next_actor_lr"])
            if "critic_lr" in metrics:
                with contextlib.suppress(Exception):
                    self._rl_critic_lr = float(metrics["critic_lr"])
            valid_smiles = [str(s.get("smiles", "") or "") for s in samples if str(s.get("smiles", "") or "")]
            unique_ratio = (len(set(valid_smiles)) / len(valid_smiles)) if valid_smiles else 0.0
            model_reward_mean = float(metrics.get("reward_mean", np.nan))
            model_reward_std = float(metrics.get("reward_std", np.nan))
            metrics["reward_mean_model"] = model_reward_mean
            metrics["reward_std_model"] = model_reward_std
            metrics["reward_mean"] = float(np.mean(rewards_raw)) if rewards_raw.size else 0.0
            metrics["reward_std"] = float(np.std(rewards_raw)) if rewards_raw.size else 0.0
            metrics["reward_mean_norm"] = float(np.mean(rewards_norm)) if rewards_norm.size else 0.0
            metrics["reward_std_norm"] = float(np.std(rewards_norm)) if rewards_norm.size else 0.0
            metrics["reward_running_mean"] = float(self._rl_reward_stats.get("mean", 0.0))
            metrics["reward_running_std"] = float(
                np.sqrt(max(float(self._rl_reward_stats.get("var", 1.0)), self._rl_reward_norm_eps))
            )
            metrics["entropy_weight_used"] = float(entropy_weight_now)
            metrics["validity"] = float(valid_count / max(1, sample_count))
            metrics["uniqueness"] = float(unique_ratio)
            metrics["samples"] = float(sample_count)
            metrics["qc_override_count"] = float(qc_override_total)
            agg_metrics.append(metrics)
            self.generator.eval()

        if not agg_metrics:
            return None
        merged: Dict[str, float] = {}
        keys = {k for m in agg_metrics for k in m.keys()}
        for key in keys:
            vals: List[float] = []
            for m in agg_metrics:
                if key not in m:
                    continue
                try:
                    fv = float(m[key])  # type: ignore[arg-type]
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            if vals:
                merged[key] = float(np.mean(vals))
        self._rl_last_metrics = merged
        self._rl_updates += 1
        merged["rl_updates"] = float(self._rl_updates)
        self._save_rl_checkpoint(merged)
        self._append_live(
            "iter=%d rl_update reward_mean=%s loss=%s validity=%s kl=%s clipfrac=%s"
            % (
                int(self.scheduler.iteration),
                "n/a" if "reward_mean" not in merged else f"{merged['reward_mean']:.4f}",
                "n/a" if "total_loss" not in merged else f"{merged['total_loss']:.4f}",
                "n/a" if "validity" not in merged else f"{merged['validity']:.3f}",
                "n/a" if "approx_kl" not in merged else f"{merged['approx_kl']:.4f}",
                "n/a" if "clipfrac" not in merged else f"{merged['clipfrac']:.3f}",
            )
        )
        logger.info(
            "RL update done (iter=%d, updates=%d): reward_mean=%.4f total_loss=%.4f validity=%.3f kl=%.4f clipfrac=%.3f ev=%.3f",
            self.scheduler.iteration,
            self._rl_updates,
            float(merged.get("reward_mean", 0.0)),
            float(merged.get("total_loss", 0.0)),
            float(merged.get("validity", 0.0)),
            float(merged.get("approx_kl", 0.0)),
            float(merged.get("clipfrac", 0.0)),
            float(merged.get("explained_variance", 0.0)),
        )
        return merged

    def run_iteration(
        self,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if self._manual_stop_requested:
            raise RuntimeError("Early stop triggered (red_score stagnation).")
        if self.scheduler.should_stop():
            raise RuntimeError("Maximum number of iterations erreicht")

        if assemble_kwargs is None:
            assemble_kwargs = self.assemble_kwargs
        logger.info(
            "Starting iteration %d | labelled=%d pool=%d | assemble_kwargs=%s",
            self.scheduler.iteration + 1,
            len(self.labelled),
            len(self.pool),
            assemble_kwargs,
        )
        generated = self._ensure_pool(self.config.batch_size, cond, assemble_kwargs)
        logger.debug("Pool replenished mit %d new samples (wenn ueberhaupt). Current pool=%d", generated, len(self.pool))
        graphs, valid_idx = self._featurize_pool()
        if not graphs:
            raise RuntimeError(
                "Keine valid candidates im pool nach filtering von invalid SMILES. "
                "Generation/seed pool hat 0 usable molecules; relax filters oder fuege mehr seed molecules hinzu"
            )
        if self.config.max_pool_eval is not None and len(graphs) > self.config.max_pool_eval:
            logger.info(
                "Capping pool evaluation zu first %d of %d candidates.",
                self.config.max_pool_eval,
                len(graphs),
            )
            graphs = graphs[: self.config.max_pool_eval]
            valid_idx = valid_idx[: self.config.max_pool_eval]
        logger.debug("Featurized %d pool candidates (valid_idx=%d).", len(graphs), len(valid_idx))
        self._append_live(
            "iter=%d scoring %d pool candidates"
            % (int(self.scheduler.iteration + 1), int(len(graphs)))
        )

        mean, std, optical_mean, optical_std, oscillator_mean, oscillator_std = self._predict_pool(graphs)
        self._append_live(
            "iter=%d scoring completed (candidates=%d)"
            % (int(self.scheduler.iteration + 1), int(len(graphs)))
        )
        logger.debug("Predictions ready: mean shape %s, std shape %s", mean.shape, std.shape)
        effective_optical_weight = self._effective_optical_weight()
        effective_oscillator_weight = self._effective_oscillator_weight()
        base_scores = self._score_candidates(mean, std)
        optical_scores = self._score_optical_candidates(optical_mean, optical_std, len(base_scores))
        oscillator_scores = self._score_oscillator_candidates(oscillator_mean, oscillator_std, len(base_scores))
        scores = (
            base_scores
            + effective_optical_weight * optical_scores
            + effective_oscillator_weight * oscillator_scores
        )
        logger.debug(
            "Acquisition scores computed (base_weight=1.0 optical_weight=%.3f oscillator_weight=%.3f)",
            effective_optical_weight,
            effective_oscillator_weight,
        )

        pool_slice = self.pool.iloc[valid_idx].copy()
        for i, name in enumerate(self.config.target_columns):
            pool_slice[f"pred_{name}"] = mean[:, i]
            pool_slice[f"pred_std_{name}"] = std[:, i]
        if optical_mean is not None and optical_std is not None:
            for i, name in enumerate(self._optical_target_columns):
                pool_slice[f"pred_{name}"] = optical_mean[:, i]
                pool_slice[f"pred_std_{name}"] = optical_std[:, i]
        if oscillator_mean is not None and oscillator_std is not None:
            for i, name in enumerate(self._oscillator_target_columns):
                pool_slice[f"pred_{name}"] = oscillator_mean[:, i]
                pool_slice[f"pred_std_{name}"] = oscillator_std[:, i]
        pool_slice["acquisition_score_base"] = base_scores
        pool_slice["acquisition_score_optical"] = optical_scores
        pool_slice["acquisition_score_oscillator"] = oscillator_scores
        pool_slice["acquisition_score"] = scores
        iteration_idx = self.scheduler.iteration + 1
        self._save_diagnostics(pool_slice, iteration_idx)

        selected = (
            pool_slice.sort_values("acquisition_score", ascending=False)
            .head(self.config.batch_size)
            .copy()
        )
        self.pool = self.pool.drop(selected.index).reset_index(drop=True)

        labelled = self._label_with_dft(selected)
        labelled["iteration"] = self.scheduler.iteration + 1
        self._compute_red_score(labelled)
        if "red_pass" in labelled.columns:
            red_passed = int(pd.to_numeric(labelled["red_pass"], errors="coerce").fillna(0).astype(int).sum())
            logger.info(
                "Red-score pass rate this iteration: %d/%d",
                red_passed,
                len(labelled),
            )
        self._append_optical_incremental_dataset(labelled)
        self.labelled = pd.concat([self.labelled, labelled], ignore_index=True)
        self.history.append(labelled)
        self._update_red_score_improvement(labelled)
        best_acq = None
        mean_acq = None
        if "acquisition_score" in selected.columns and not selected.empty:
            with contextlib.suppress(Exception):
                best_acq = float(selected["acquisition_score"].max())
            with contextlib.suppress(Exception):
                mean_acq = float(selected["acquisition_score"].mean())
        best_red = None
        if "red_score" in labelled.columns and not labelled.empty:
            with contextlib.suppress(Exception):
                best_red = float(pd.to_numeric(labelled["red_score"], errors="coerce").max())
        history_row: Dict[str, object] = {
            "iteration": int(self.scheduler.iteration + 1),
            "selected": int(len(labelled)),
            "generated": int(generated),
            "best_acq": best_acq,
            "mean_acq": mean_acq,
            "best_red": best_red,
            "rl_reward_mean": None,
            "rl_total_loss": None,
            "rl_policy_loss": None,
            "rl_value_loss": None,
            "rl_entropy": None,
            "rl_validity": None,
            "rl_uniqueness": None,
            "rl_approx_kl": None,
            "rl_clipfrac": None,
            "rl_explained_variance": None,
            "rl_entropy_weight": None,
            "rl_actor_lr": None,
            "rl_critic_lr": None,
            "rl_ppo_clip_ratio_used": None,
        }
        self._live_history_rows.append(history_row)

        self.scheduler.step(num_labelled=len(labelled), num_generated=generated)

        if self.scheduler.should_retrain_surrogate():
            self._retrain_surrogate()
        self._retrain_optical_surrogate()
        self._retrain_oscillator_surrogate()
        rl_metrics = self._rl_update_generator(cond=cond, assemble_kwargs=assemble_kwargs)
        if rl_metrics:
            history_row["rl_reward_mean"] = float(rl_metrics.get("reward_mean", np.nan))
            history_row["rl_total_loss"] = float(rl_metrics.get("total_loss", np.nan))
            history_row["rl_policy_loss"] = float(rl_metrics.get("policy_loss", np.nan))
            history_row["rl_value_loss"] = float(rl_metrics.get("value_loss", np.nan))
            history_row["rl_entropy"] = float(rl_metrics.get("entropy", np.nan))
            history_row["rl_validity"] = float(rl_metrics.get("validity", np.nan))
            history_row["rl_uniqueness"] = float(rl_metrics.get("uniqueness", np.nan))
            history_row["rl_approx_kl"] = float(rl_metrics.get("approx_kl", np.nan))
            history_row["rl_clipfrac"] = float(rl_metrics.get("clipfrac", np.nan))
            history_row["rl_explained_variance"] = float(rl_metrics.get("explained_variance", np.nan))
            history_row["rl_entropy_weight"] = float(rl_metrics.get("entropy_weight_used", np.nan))
            history_row["rl_actor_lr"] = float(rl_metrics.get("actor_lr", np.nan))
            history_row["rl_critic_lr"] = float(rl_metrics.get("critic_lr", np.nan))
            history_row["rl_ppo_clip_ratio_used"] = float(rl_metrics.get("ppo_clip_ratio_used", np.nan))
            history_row["rl_qc_override_count"] = float(rl_metrics.get("qc_override_count", 0.0))
            history_row["rl_updates"] = float(rl_metrics.get("rl_updates", self._rl_updates))

        if self.scheduler.should_refresh_generator():
            self._refresh_generator()
        self._append_live(
            "iter=%d selected=%d generated=%d best_acq=%s mean_acq=%s rl_reward=%s"
            % (
                int(self.scheduler.iteration),
                int(len(labelled)),
                int(generated),
                "n/a" if best_acq is None else f"{best_acq:.4f}",
                "n/a" if mean_acq is None else f"{mean_acq:.4f}",
                "n/a"
                if not rl_metrics or ("reward_mean" not in rl_metrics)
                else f"{float(rl_metrics['reward_mean']):.4f}",
            )
        )
        self._update_live_dashboard(selected=labelled, generated=generated)

        return labelled

    def run(
        self,
        n_iterations: int,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> List[pd.DataFrame]:
        if assemble_kwargs is None:
            assemble_kwargs = {}
        merged_kwargs = {**self.assemble_kwargs, **assemble_kwargs}
        self._live_total_iterations = min(
            int(getattr(self.scheduler.config, "max_iterations", n_iterations)),
            int(self.scheduler.iteration + n_iterations),
        )
        self._append_live(
            f"Run started: n_iterations={n_iterations}, goal_iteration={self._live_total_iterations}"
        )
        self._update_live_dashboard(selected=None, generated=0)
        for i in range(n_iterations):
            if self.scheduler.should_stop() or self._manual_stop_requested:
                if self._manual_stop_requested:
                    self._append_live("Run stopped early: red_score stagnation criterion reached.")
                break
            logger.info(
                "Loop progress: iteration %d/%d (labelled=%d, pool=%d)",
                i + 1,
                n_iterations,
                len(self.labelled),
                len(self.pool),
            )
            self.run_iteration(cond=cond, assemble_kwargs=merged_kwargs)
        self._append_live("Run finished.")
        self._update_live_dashboard(selected=None, generated=0)
        return self.history

    def _export_top_candidates(self, history_df: pd.DataFrame) -> Optional[Path]:
        if history_df.empty or self._export_top_k <= 0:
            return None
        work = history_df.copy()
        if self._export_require_qc_success and "qc_status" in work.columns:
            status = work["qc_status"].astype(str).str.lower()
            work = work[status == "success"].copy()
        if self._export_require_red_pass and "red_pass" in work.columns:
            work = work[work["red_pass"].fillna(False).astype(bool)].copy()
        if work.empty:
            return None
        sort_col = self._export_sort_column if self._export_sort_column in work.columns else "acquisition_score"
        if sort_col not in work.columns:
            return None
        work[sort_col] = pd.to_numeric(work[sort_col], errors="coerce")
        work = work[np.isfinite(work[sort_col].to_numpy(dtype=float))]
        if work.empty:
            return None
        work = work.sort_values(sort_col, ascending=False)
        if "smiles" in work.columns:
            work = work.drop_duplicates(subset=["smiles"], keep="first")
        top = work.head(self._export_top_k).copy()
        out_path = self._export_top_candidates_path or (self.results_dir / "top_red_candidates.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(out_path, index=False)
        return out_path

    def _write_run_summary(self, history_df: pd.DataFrame) -> Optional[Path]:
        if history_df.empty:
            return None
        summary = {
            "rows": int(len(history_df)),
            "iterations_completed": int(self.scheduler.iteration),
            "unique_smiles": int(history_df["smiles"].nunique()) if "smiles" in history_df.columns else None,
            "best_acquisition_score": None,
            "best_red_score": None,
            "red_pass_rate": None,
            "best_rl_reward_mean": None,
            "best_rl_validity": None,
            "best_rl_approx_kl": None,
            "best_rl_explained_variance": None,
            "rl_updates": int(self._rl_updates),
        }
        if "acquisition_score" in history_df.columns:
            vals = pd.to_numeric(history_df["acquisition_score"], errors="coerce")
            vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
            if not vals.empty:
                summary["best_acquisition_score"] = float(vals.max())
        if "red_score" in history_df.columns:
            vals = pd.to_numeric(history_df["red_score"], errors="coerce")
            vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
            if not vals.empty:
                summary["best_red_score"] = float(vals.max())
        if "red_pass" in history_df.columns and len(history_df) > 0:
            passes = history_df["red_pass"].fillna(False).astype(bool)
            summary["red_pass_rate"] = float(passes.mean())
        if self._live_history_rows:
            hist_live = pd.DataFrame(self._live_history_rows)
            if "rl_reward_mean" in hist_live.columns:
                vals = pd.to_numeric(hist_live["rl_reward_mean"], errors="coerce")
                vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
                if not vals.empty:
                    summary["best_rl_reward_mean"] = float(vals.max())
            if "rl_validity" in hist_live.columns:
                vals = pd.to_numeric(hist_live["rl_validity"], errors="coerce")
                vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
                if not vals.empty:
                    summary["best_rl_validity"] = float(vals.max())
            if "rl_approx_kl" in hist_live.columns:
                vals = pd.to_numeric(hist_live["rl_approx_kl"], errors="coerce")
                vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
                if not vals.empty:
                    summary["best_rl_approx_kl"] = float(vals.max())
            if "rl_explained_variance" in hist_live.columns:
                vals = pd.to_numeric(hist_live["rl_explained_variance"], errors="coerce")
                vals = vals[np.isfinite(vals.to_numpy(dtype=float))]
                if not vals.empty:
                    summary["best_rl_explained_variance"] = float(vals.max())
        out_path = self.results_dir / "run_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out_path

    def save_history(self) -> None:
        if not self.history:
            return
        path = self.results_dir / "active_learning_history.csv"
        hist_df = pd.concat(self.history, ignore_index=True)
        hist_df.to_csv(path, index=False)
        logger.info("Fertig, active learning history (ergebnnisse) in %s", path)
        top_path = self._export_top_candidates(hist_df)
        if top_path is not None:
            logger.info("Exported top candidates to %s", top_path)
        summary_path = self._write_run_summary(hist_df)
        if summary_path is not None:
            logger.info("Wrote run summary to %s", summary_path)

