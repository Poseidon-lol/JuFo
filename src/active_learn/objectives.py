"""Objective profile loading and application helpers for active learning."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate project root containing src/")


OBJECTIVE_MODES: Tuple[str, ...] = ("red", "blue", "general")


def normalize_objective_mode(mode: Optional[str], default: str = "red") -> str:
    raw = str(mode or default).strip().lower()
    if raw not in OBJECTIVE_MODES:
        raise ValueError(f"Unsupported objective_mode '{raw}'. Allowed: {OBJECTIVE_MODES}")
    return raw


def _resolve_profile_path(path: Optional[str]) -> Path:
    if path:
        profile_path = Path(path).expanduser()
        if not profile_path.is_absolute():
            profile_path = (Path.cwd() / profile_path).resolve()
        return profile_path
    return (PROJECT_ROOT / "configs" / "objectives.yaml").resolve()


def _as_sequence(value: Any, name: str) -> Optional[Sequence[Any]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"'{name}' must be a list/tuple, got string.")
    if not isinstance(value, Sequence):
        raise ValueError(f"'{name}' must be a list/tuple.")
    return value


def _validate_length_match(
    base: Optional[Sequence[Any]],
    base_name: str,
    candidate: Optional[Sequence[Any]],
    candidate_name: str,
) -> None:
    if base is None or candidate is None:
        return
    if len(base) != len(candidate):
        raise ValueError(
            f"Length mismatch in objective profile: {base_name}={len(base)} vs {candidate_name}={len(candidate)}"
        )


def _validate_profile(mode: str, profile: Mapping[str, Any]) -> None:
    if not isinstance(profile, Mapping):
        raise ValueError(f"Objective profile '{mode}' must be a mapping.")

    acq = profile.get("acquisition")
    if acq is not None and not isinstance(acq, Mapping):
        raise ValueError(f"Objective profile '{mode}.acquisition' must be a mapping.")
    if isinstance(acq, Mapping):
        targets = _as_sequence(acq.get("targets"), f"{mode}.acquisition.targets")
        tolerances = _as_sequence(acq.get("tolerances"), f"{mode}.acquisition.tolerances")
        weights = _as_sequence(acq.get("weights"), f"{mode}.acquisition.weights")
        maximise = _as_sequence(acq.get("maximise"), f"{mode}.acquisition.maximise")
        _validate_length_match(targets, "acquisition.targets", tolerances, "acquisition.tolerances")
        _validate_length_match(targets, "acquisition.targets", weights, "acquisition.weights")
        _validate_length_match(targets, "acquisition.targets", maximise, "acquisition.maximise")

    objective_score = profile.get("objective_score")
    if objective_score is not None and not isinstance(objective_score, Mapping):
        raise ValueError(f"Objective profile '{mode}.objective_score' must be a mapping.")
    if isinstance(objective_score, Mapping):
        cols = _as_sequence(objective_score.get("columns"), f"{mode}.objective_score.columns")
        targets = _as_sequence(objective_score.get("targets"), f"{mode}.objective_score.targets")
        tolerances = _as_sequence(objective_score.get("tolerances"), f"{mode}.objective_score.tolerances")
        weights = _as_sequence(objective_score.get("weights"), f"{mode}.objective_score.weights")
        _validate_length_match(cols, "objective_score.columns", targets, "objective_score.targets")
        _validate_length_match(cols, "objective_score.columns", tolerances, "objective_score.tolerances")
        _validate_length_match(cols, "objective_score.columns", weights, "objective_score.weights")

    for sec_name in ("optical", "oscillator"):
        sec = profile.get(sec_name)
        if sec is not None and not isinstance(sec, Mapping):
            raise ValueError(f"Objective profile '{mode}.{sec_name}' must be a mapping.")
        if not isinstance(sec, Mapping):
            continue
        cols = _as_sequence(sec.get("target_columns"), f"{mode}.{sec_name}.target_columns")
        targets = _as_sequence(sec.get("targets"), f"{mode}.{sec_name}.targets")
        tolerances = _as_sequence(sec.get("tolerances"), f"{mode}.{sec_name}.tolerances")
        weights = _as_sequence(sec.get("weights"), f"{mode}.{sec_name}.weights")
        _validate_length_match(cols, f"{sec_name}.target_columns", targets, f"{sec_name}.targets")
        _validate_length_match(cols, f"{sec_name}.target_columns", tolerances, f"{sec_name}.tolerances")
        _validate_length_match(cols, f"{sec_name}.target_columns", weights, f"{sec_name}.weights")


def load_objective_profile(mode: str, profile_path: Optional[str] = None) -> Tuple[Dict[str, Any], Path]:
    resolved_mode = normalize_objective_mode(mode)
    resolved_path = _resolve_profile_path(profile_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Objective profile file not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Objective profile file must be a mapping: {resolved_path}")
    profiles = payload.get("modes", payload)
    if not isinstance(profiles, Mapping):
        raise ValueError("Objective profile root must contain a mapping under 'modes'.")
    if resolved_mode not in profiles:
        raise KeyError(
            f"objective_mode '{resolved_mode}' missing in {resolved_path}. Available: {sorted(profiles.keys())}"
        )
    profile = profiles[resolved_mode]
    if not isinstance(profile, Mapping):
        raise ValueError(f"Profile '{resolved_mode}' must be a mapping.")
    profile_dict = dict(profile)
    _validate_profile(resolved_mode, profile_dict)
    return profile_dict, resolved_path


def _set_many(target: MutableMapping[str, Any], values: Mapping[str, Any], allowed: Sequence[str]) -> None:
    for key in allowed:
        if key in values:
            target[key] = values[key]


def apply_objective_profile(cfg: Any, mode: str, profile: Mapping[str, Any], profile_path: Path) -> Dict[str, Any]:
    """Apply profile values to config object in-place.

    Returns a dictionary with sections that were applied, useful for logging.
    """

    if "loop" not in cfg:
        cfg.loop = {}
    if "acquisition" not in cfg:
        cfg.acquisition = {}

    applied: Dict[str, Any] = {"mode": mode, "profile_path": str(profile_path)}
    loop_cfg = cfg.loop
    acq_cfg = cfg.acquisition
    loop_cfg["objective_mode"] = mode
    loop_cfg["objective_profile_path"] = str(profile_path)

    acquisition = profile.get("acquisition")
    if isinstance(acquisition, Mapping):
        _set_many(
            acq_cfg,
            acquisition,
            allowed=("kind", "beta", "xi", "targets", "tolerances", "weights", "maximise"),
        )
        applied["acquisition"] = dict(acquisition)

    objective_score = profile.get("objective_score")
    if isinstance(objective_score, Mapping):
        # Keep existing red_score fields for backwards compatibility.
        mapping = {
            "columns": "red_score_columns",
            "targets": "red_score_targets",
            "tolerances": "red_score_tolerances",
            "weights": "red_score_weights",
            "missing_penalty": "red_score_missing_penalty",
            "pass_threshold": "red_score_pass_threshold",
            "require_qc_success": "red_score_require_qc_success",
            "export_sort_column": "export_sort_column",
        }
        for src_key, dst_key in mapping.items():
            if src_key in objective_score:
                loop_cfg[dst_key] = objective_score[src_key]
        # Mirror in new neutral naming for later steps.
        if "columns" in objective_score:
            loop_cfg["objective_score_columns"] = objective_score["columns"]
        if "targets" in objective_score:
            loop_cfg["objective_score_targets"] = objective_score["targets"]
        if "tolerances" in objective_score:
            loop_cfg["objective_score_tolerances"] = objective_score["tolerances"]
        if "weights" in objective_score:
            loop_cfg["objective_score_weights"] = objective_score["weights"]
        if "missing_penalty" in objective_score:
            loop_cfg["objective_score_missing_penalty"] = objective_score["missing_penalty"]
        if "pass_threshold" in objective_score:
            loop_cfg["objective_score_pass_threshold"] = objective_score["pass_threshold"]
        if "require_qc_success" in objective_score:
            loop_cfg["objective_score_require_qc_success"] = objective_score["require_qc_success"]
        applied["objective_score"] = dict(objective_score)

    hard_gates = profile.get("hard_gates")
    if isinstance(hard_gates, Mapping):
        if "min_lambda_max_nm" in hard_gates:
            loop_cfg["min_lambda_max_nm"] = hard_gates["min_lambda_max_nm"]
        if "min_oscillator_strength" in hard_gates:
            loop_cfg["min_oscillator_strength"] = hard_gates["min_oscillator_strength"]
        # Keep future-ready generic gate keys.
        _set_many(
            loop_cfg,
            hard_gates,
            allowed=(
                "objective_gate_min_lambda_max_nm",
                "objective_gate_max_lambda_max_nm",
                "objective_gate_min_oscillator_strength",
                "objective_gate_max_oscillator_strength",
                "min_lambda_max_nm",
                "max_lambda_max_nm",
                "min_oscillator_strength",
                "max_oscillator_strength",
            ),
        )
        applied["hard_gates"] = dict(hard_gates)

    optical = profile.get("optical")
    if isinstance(optical, Mapping):
        mapping = {
            "score_weight": "optical_score_weight",
            "target_columns": "optical_target_columns",
            "targets": "optical_targets",
            "tolerances": "optical_tolerances",
            "weights": "optical_weights",
            "beta": "optical_beta",
        }
        for src_key, dst_key in mapping.items():
            if src_key in optical:
                loop_cfg[dst_key] = optical[src_key]
        applied["optical"] = dict(optical)

    oscillator = profile.get("oscillator")
    if isinstance(oscillator, Mapping):
        mapping = {
            "score_weight": "oscillator_score_weight",
            "target_columns": "oscillator_target_columns",
            "targets": "oscillator_targets",
            "tolerances": "oscillator_tolerances",
            "weights": "oscillator_weights",
            "beta": "oscillator_beta",
        }
        for src_key, dst_key in mapping.items():
            if src_key in oscillator:
                loop_cfg[dst_key] = oscillator[src_key]
        applied["oscillator"] = dict(oscillator)

    if "qc_extra_properties" in profile:
        loop_cfg["qc_extra_properties"] = profile["qc_extra_properties"]
        applied["qc_extra_properties"] = profile["qc_extra_properties"]

    if isinstance(profile.get("rl_reward_weights"), Mapping):
        loop_cfg["rl_reward_weights"] = dict(profile["rl_reward_weights"])
        applied["rl_reward_weights"] = dict(profile["rl_reward_weights"])

    return applied
