from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.active_learn.acq import AcquisitionConfig, acquisition_score
from src.active_learn.loop import ActiveLearningLoop


class _DummyLoop:
    def __init__(self, acq_cfg: AcquisitionConfig):
        self.config = SimpleNamespace(acquisition=acq_cfg)

    def _normalise_predictions(self, mean: np.ndarray, std: np.ndarray):
        n_targets = mean.shape[1]
        mus = np.zeros(n_targets, dtype=float)
        sigmas = np.ones(n_targets, dtype=float)
        directions = np.ones(n_targets, dtype=float)
        return mean, std, mus, sigmas, directions

    def _current_best(self):
        return None


class _DummyOpticalLoop:
    def __init__(
        self,
        *,
        optical_score_weight: float,
        optical_targets,
        optical_tolerances,
        optical_weights,
        optical_beta: float = 0.0,
    ):
        self.optical_surrogate = object()
        self._optical_score_weight = optical_score_weight
        self._optical_targets = optical_targets
        self._optical_tolerances = optical_tolerances
        self._optical_weights = optical_weights
        self._optical_beta = optical_beta
        self._optical_warned_no_targets = False


class _DummyWarmupLoop:
    def __init__(self, labelled: pd.DataFrame, min_labels: int, base_weight: float):
        self.optical_surrogate = object()
        self.labelled = labelled
        self._optical_target_columns = ("lambda_max_nm", "oscillator_strength")
        self._optical_retrain_on_success_only = True
        self._optical_score_weight = base_weight
        self._optical_base_weight = base_weight
        self._optical_weight_schedule = []
        self._min_labels_for_optical = min_labels
        self._optical_weight_unlocked = False
        self.scheduler = SimpleNamespace(iteration=0)

    def _count_optical_labels(self, frame: pd.DataFrame, *, require_success: bool) -> int:
        return ActiveLearningLoop._count_optical_labels(self, frame, require_success=require_success)


class _DummyRedScoreLoop:
    def __init__(self):
        self._red_score_columns = ("lambda_max_nm", "oscillator_strength")
        self._red_score_targets = (680.0, 0.25)
        self._red_score_tolerances = (60.0, 0.10)
        self._red_score_weights = (1.0, 1.0)
        self._red_score_missing_penalty = 3.0
        self._red_score_pass_threshold = -1.0
        self._red_score_require_qc_success = True


def test_score_candidates_forwards_target_config():
    mean = np.array(
        [
            [-5.4, -2.8, 2.1],
            [-5.1, -2.4, 1.8],
        ],
        dtype=float,
    )
    std = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.10, 0.20],
        ],
        dtype=float,
    )
    acq_cfg = AcquisitionConfig(
        kind="target",
        beta=0.0,
        targets=[-5.4, -2.8, 2.0],
        tolerances=[0.3, 0.4, 0.5],
        weights=[1.0, 0.8, 1.0],
    )

    loop = _DummyLoop(acq_cfg)
    observed = ActiveLearningLoop._score_candidates(loop, mean, std)
    expected = acquisition_score(mean, std, acq_cfg, best_so_far=None)

    assert np.allclose(observed, expected)


def test_score_optical_candidates_distance_weighted():
    mean = np.array(
        [
            [650.0, 0.20],
            [700.0, 0.10],
        ],
        dtype=float,
    )
    std = np.zeros_like(mean)
    loop = _DummyOpticalLoop(
        optical_score_weight=1.0,
        optical_targets=(650.0, 0.20),
        optical_tolerances=(50.0, 0.10),
        optical_weights=(1.0, 1.0),
    )

    observed = ActiveLearningLoop._score_optical_candidates(loop, mean, std, n_candidates=2)
    expected = np.array([0.0, -2.0], dtype=float)

    assert np.allclose(observed, expected)


def test_score_optical_candidates_returns_zero_when_weight_zero():
    mean = np.array([[650.0, 0.20]], dtype=float)
    std = np.array([[0.1, 0.1]], dtype=float)
    loop = _DummyOpticalLoop(
        optical_score_weight=0.0,
        optical_targets=(650.0, 0.20),
        optical_tolerances=(50.0, 0.10),
        optical_weights=(1.0, 1.0),
    )

    observed = ActiveLearningLoop._score_optical_candidates(loop, mean, std, n_candidates=1)

    assert np.allclose(observed, np.zeros(1, dtype=float))


def test_effective_optical_weight_respects_warmup_threshold():
    labelled = pd.DataFrame(
        {
            "lambda_max_nm": [700.0, 690.0],
            "oscillator_strength": [0.22, 0.25],
            "qc_status": ["success", "success"],
        }
    )
    loop = _DummyWarmupLoop(labelled=labelled, min_labels=3, base_weight=0.4)

    w_before = ActiveLearningLoop._effective_optical_weight(loop)
    assert np.isclose(w_before, 0.0)

    loop.labelled = pd.concat(
        [
            loop.labelled,
            pd.DataFrame(
                {"lambda_max_nm": [710.0], "oscillator_strength": [0.24], "qc_status": ["success"]}
            ),
        ],
        ignore_index=True,
    )
    w_after = ActiveLearningLoop._effective_optical_weight(loop)
    assert np.isclose(w_after, 0.4)


def test_compute_red_score_sets_columns_and_qc_guard():
    frame = pd.DataFrame(
        {
            "lambda_max_nm": [680.0, 560.0, 690.0],
            "oscillator_strength": [0.25, 0.05, 0.30],
            "qc_status": ["success", "success", "error"],
        }
    )
    loop = _DummyRedScoreLoop()

    ActiveLearningLoop._compute_red_score(loop, frame)

    assert "red_score" in frame.columns
    assert "red_pass" in frame.columns
    assert "red_reason" in frame.columns

    assert np.isclose(float(frame.loc[0, "red_score"]), 0.0)
    assert bool(frame.loc[0, "red_pass"]) is True
    assert str(frame.loc[0, "red_reason"]) == "ok"

    assert bool(frame.loc[1, "red_pass"]) is False
    assert "score_below_threshold" in str(frame.loc[1, "red_reason"])

    assert pd.isna(frame.loc[2, "red_score"])
    assert bool(frame.loc[2, "red_pass"]) is False
    assert str(frame.loc[2, "red_reason"]) == "qc_not_success"


def test_effective_optical_weight_uses_stage_schedule():
    labelled = pd.DataFrame(
        {
            "lambda_max_nm": [700.0, 690.0, 710.0],
            "oscillator_strength": [0.22, 0.25, 0.24],
            "qc_status": ["success", "success", "success"],
        }
    )
    loop = _DummyWarmupLoop(labelled=labelled, min_labels=0, base_weight=0.4)
    loop._optical_weight_schedule = [(2, 0.1), (4, 0.3), (999, 0.6)]

    loop.scheduler.iteration = 0
    w1 = ActiveLearningLoop._effective_optical_weight(loop)
    loop.scheduler.iteration = 2
    w3 = ActiveLearningLoop._effective_optical_weight(loop)
    loop.scheduler.iteration = 5
    w6 = ActiveLearningLoop._effective_optical_weight(loop)

    assert np.isclose(w1, 0.1)
    assert np.isclose(w3, 0.3)
    assert np.isclose(w6, 0.6)
