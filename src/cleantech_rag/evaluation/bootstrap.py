"""Percentile bootstrap confidence intervals over per-question values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_RESAMPLES = 1000
CONFIDENCE = 0.95
SEED = 0


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = N_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> dict[str, float | int] | None:
    """Mean and percentile-bootstrap CI of ``values`` (resampling questions with replacement).

    Deterministic for a given ``seed``. Returns ``None`` for no values; with a single value
    there is nothing to resample, so the interval collapses to that value.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return None
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    alpha = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [alpha, 100 - alpha])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": n}
