import numpy as np

from src.models.reward import compute_objective_reward


def test_reward_penalties_and_diversity_bonus():
    preds = {"lambda_max_nm": 680.0, "oscillator_strength": 0.25, "gap": 1.8}
    reward_ok, _ = compute_objective_reward(
        preds,
        objective_mode="red",
        diversity=0.9,
        novelty=True,
        diversity_weight=0.2,
        novelty_bonus=0.1,
        in_range_bonus=0.2,
    )
    reward_bad, _ = compute_objective_reward(
        preds,
        objective_mode="red",
        invalid=True,
        duplicate=True,
        invalid_penalty=6.0,
        duplicate_penalty=2.0,
    )
    assert reward_ok > reward_bad


def test_reward_weight_override_from_mapping():
    preds = {"lambda_max_nm": 680.0, "oscillator_strength": 0.10, "gap": 1.8}
    base, _ = compute_objective_reward(preds, objective_mode="red")
    boosted, _ = compute_objective_reward(
        preds,
        objective_mode="red",
        weight_overrides={"oscillator_strength": 3.0},
    )
    assert not np.isclose(boosted, base)
    assert np.isfinite(boosted)
