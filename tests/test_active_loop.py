from types import SimpleNamespace

import numpy as np

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
