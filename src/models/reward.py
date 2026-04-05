"""Reward utilities for JT-VAE reinforcement learning."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "compute_objective_reward",
    "compute_reward_from_objective_profile",
]


_DEFAULT_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "red": ("lambda_max_nm", "oscillator_strength", "gap"),
    "blue": ("lambda_max_nm", "oscillator_strength", "gap"),
    "general": ("homo", "lumo", "gap"),
}

_DEFAULT_TARGETS: Dict[str, Tuple[float, ...]] = {
    "red": (680.0, 0.25, 1.8),
    "blue": (460.0, 0.28, 2.5),
    "general": (-5.3, -3.1, 1.8),
}

_DEFAULT_TOLERANCES: Dict[str, Tuple[float, ...]] = {
    "red": (60.0, 0.08, 0.45),
    "blue": (35.0, 0.10, 0.50),
    "general": (0.4, 0.5, 0.7),
}

_DEFAULT_WEIGHTS: Dict[str, Tuple[float, ...]] = {
    "red": (1.2, 1.6, 0.8),
    "blue": (1.6, 1.1, 0.9),
    "general": (1.0, 1.0, 1.0),
}


def _mode_key(mode: str) -> str:
    key = str(mode or "red").strip().lower()
    if key not in _DEFAULT_COMPONENTS:
        raise ValueError(f"Unsupported objective mode '{mode}'.")
    return key


def _gaussian_target_score(value: float, target: float, tolerance: float) -> float:
    tol = float(tolerance)
    if tol <= 1e-12:
        return -abs(float(value) - float(target))
    z = (float(value) - float(target)) / tol
    return float(np.exp(-0.5 * z * z))


def compute_objective_reward(
    predictions: Mapping[str, float],
    *,
    objective_mode: str = "red",
    components: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[float]] = None,
    tolerances: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    weight_overrides: Optional[Mapping[str, float]] = None,
    uncertainties: Optional[Mapping[str, float]] = None,
    uncertainty_weight: float = 0.0,
    missing_penalty: float = 1.0,
    invalid: bool = False,
    duplicate: bool = False,
    diversity: Optional[float] = None,
    novelty: Optional[bool] = None,
    invalid_penalty: float = 5.0,
    duplicate_penalty: float = 2.0,
    in_range_bonus: float = 0.0,
    diversity_weight: float = 0.0,
    novelty_bonus: float = 0.0,
    clip: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute scalar RL reward from predicted molecular properties.

    Returns ``(reward, components_dict)``.
    """

    mode = _mode_key(objective_mode)
    cols = tuple(components or _DEFAULT_COMPONENTS[mode])
    tars = tuple(float(v) for v in (targets or _DEFAULT_TARGETS[mode]))
    tols = tuple(float(v) for v in (tolerances or _DEFAULT_TOLERANCES[mode]))
    wts_list = [float(v) for v in (weights or _DEFAULT_WEIGHTS[mode])]
    if isinstance(weight_overrides, Mapping):
        for i, name in enumerate(cols):
            if name in weight_overrides:
                wts_list[i] = float(weight_overrides[name])
    wts = tuple(wts_list)

    if len(cols) != len(tars) or len(cols) != len(tols) or len(cols) != len(wts):
        raise ValueError(
            "components/targets/tolerances/weights length mismatch "
            f"({len(cols)}/{len(tars)}/{len(tols)}/{len(wts)})."
        )

    details: Dict[str, float] = {}
    reward = 0.0

    for name, target, tol, weight in zip(cols, tars, tols, wts):
        raw = predictions.get(name, None)
        if raw is None:
            penalty = float(missing_penalty) * abs(float(weight))
            reward -= penalty
            details[f"{name}_missing_penalty"] = -penalty
            continue
        value = float(raw)
        comp = _gaussian_target_score(value, target, tol) * float(weight)
        reward += comp
        details[f"{name}_score"] = float(comp)
        if in_range_bonus > 0.0 and abs(value - target) <= abs(float(tol)):
            bonus = float(in_range_bonus) * abs(float(weight))
            reward += bonus
            details[f"{name}_range_bonus"] = float(bonus)

    if uncertainties:
        unc_vals = [float(v) for k, v in uncertainties.items() if k in cols and np.isfinite(float(v))]
        if unc_vals:
            unc_pen = float(uncertainty_weight) * float(np.mean(unc_vals))
            reward -= unc_pen
            details["uncertainty_penalty"] = -float(unc_pen)

    if invalid:
        reward -= float(invalid_penalty)
        details["invalid_penalty"] = -float(invalid_penalty)
    if duplicate:
        reward -= float(duplicate_penalty)
        details["duplicate_penalty"] = -float(duplicate_penalty)
    if diversity is not None and np.isfinite(float(diversity)):
        diversity_clamped = float(np.clip(float(diversity), 0.0, 1.0))
        bonus = float(diversity_weight) * diversity_clamped
        reward += bonus
        details["diversity_bonus"] = float(bonus)
    if novelty is True and float(novelty_bonus) != 0.0:
        reward += float(novelty_bonus)
        details["novelty_bonus"] = float(novelty_bonus)

    if clip is not None:
        c = abs(float(clip))
        reward = float(np.clip(reward, -c, c))
        details["clipped"] = 1.0
    else:
        details["clipped"] = 0.0

    details["reward"] = float(reward)
    return float(reward), details


def compute_reward_from_objective_profile(
    predictions: Mapping[str, float],
    *,
    objective_mode: str,
    objective_profile: Optional[Mapping[str, Any]] = None,
    uncertainties: Optional[Mapping[str, float]] = None,
    invalid: bool = False,
    duplicate: bool = False,
    diversity: Optional[float] = None,
    novel: Optional[bool] = None,
    clip: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute reward using the loaded objective profile when available."""

    components = None
    targets = None
    tolerances = None
    weights = None
    weight_overrides: Optional[Dict[str, float]] = None
    uncertainty_weight = 0.0
    missing_penalty = 1.0
    invalid_penalty = 5.0
    duplicate_penalty = 2.0
    in_range_bonus = 0.0
    diversity_weight = 0.0
    novelty_bonus = 0.0

    if isinstance(objective_profile, Mapping):
        score_cfg = objective_profile.get("objective_score")
        if isinstance(score_cfg, Mapping):
            components = score_cfg.get("columns")
            targets = score_cfg.get("targets")
            tolerances = score_cfg.get("tolerances")
            weights = score_cfg.get("weights")
            if score_cfg.get("missing_penalty") is not None:
                missing_penalty = float(score_cfg.get("missing_penalty"))
        reward_cfg = objective_profile.get("rl_reward_weights")
        if isinstance(reward_cfg, Mapping):
            if reward_cfg.get("uncertainty_penalty") is not None:
                uncertainty_weight = float(reward_cfg.get("uncertainty_penalty"))
            if reward_cfg.get("invalid_penalty") is not None:
                invalid_penalty = float(reward_cfg.get("invalid_penalty"))
            if reward_cfg.get("duplicate_penalty") is not None:
                duplicate_penalty = float(reward_cfg.get("duplicate_penalty"))
            if reward_cfg.get("in_range_bonus") is not None:
                in_range_bonus = float(reward_cfg.get("in_range_bonus"))
            if reward_cfg.get("diversity_weight") is not None:
                diversity_weight = float(reward_cfg.get("diversity_weight"))
            if reward_cfg.get("novelty_bonus") is not None:
                novelty_bonus = float(reward_cfg.get("novelty_bonus"))
            weight_overrides = {}
            for key, value in reward_cfg.items():
                if key in {
                    "uncertainty_penalty",
                    "invalid_penalty",
                    "duplicate_penalty",
                    "in_range_bonus",
                    "diversity_weight",
                    "novelty_bonus",
                }:
                    continue
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    weight_overrides[str(key)] = float(value)
            if not weight_overrides:
                weight_overrides = None

    return compute_objective_reward(
        predictions,
        objective_mode=objective_mode,
        components=components,
        targets=targets,
        tolerances=tolerances,
        weights=weights,
        weight_overrides=weight_overrides,
        uncertainties=uncertainties,
        uncertainty_weight=uncertainty_weight,
        missing_penalty=missing_penalty,
        invalid=invalid,
        duplicate=duplicate,
        diversity=diversity,
        novelty=novel,
        invalid_penalty=invalid_penalty,
        duplicate_penalty=duplicate_penalty,
        in_range_bonus=in_range_bonus,
        diversity_weight=diversity_weight,
        novelty_bonus=novelty_bonus,
        clip=clip,
    )
