"""Scoring.

The central subtlety of this reproduction lives in this file, so it is worth
stating plainly at the top.

The paper reports R2 values such as 0.8764. Those are *not* R2 over a test set
of sequences. They are R2 computed across the eight entries of a single
property vector: one designed sequence, its eight predicted property values,
against the eight target values that were asked for. The released notebook
makes this explicit - `r2_score(req, prop)` inside `generate_new()`, where both
arguments have length 8.

This is a self-consistency measure: does the model's own forward pass agree
with the target it was asked to design for? It is a reasonable thing to
measure, and the paper describes it as such. It is not a measure of
generalisation to unseen sequences, and it should not be read as one. We
therefore compute two families of scores and never mix them:

    r2_within_vector(...)  - reproduces the paper's number exactly.
    r2_across_dataset(...) - the ordinary regression R2, per property, over a
                             held-out set of real sequences. The paper does not
                             report this.

One more consequence of the within-vector definition: R2 is measured against
the variance of the eight target values. For a target set whose eight entries
are tightly clustered, that denominator is small and R2 is correspondingly
brittle. target_spread() reports the denominator so this can be seen rather
than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# Within-vector scores (the paper's definition)
# --------------------------------------------------------------------------


def r2_within_vector(target: Sequence[float], predicted: Sequence[float]) -> float:
    """R2 across the 8 entries of one property vector.

    Matches sklearn.metrics.r2_score(target, predicted) but is written out so
    the argument order is unambiguous: `target` is the ground truth, the thing
    that supplies the variance in the denominator.
    """
    t = np.asarray(target, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0.0:
        # Degenerate: every target entry identical. sklearn returns 1.0 for a
        # perfect fit and 0.0 otherwise; we mirror that rather than invent a
        # convention.
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def target_spread(target: Sequence[float]) -> float:
    """The R2 denominator: sum of squared deviations of the 8 target values.

    Small values mean the reported R2 is highly sensitive to small absolute
    errors. Reported alongside every within-vector R2 in our tables.
    """
    t = np.asarray(target, dtype=float)
    return float(np.sum((t - t.mean()) ** 2))


def mae(target: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(target) - np.asarray(predicted))))


def rmse(target: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(target) - np.asarray(predicted)) ** 2)))


# --------------------------------------------------------------------------
# Across-dataset scores (ordinary regression R2)
# --------------------------------------------------------------------------


def r2_across_dataset(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column R2 over N samples. Shapes (N, K) -> (K,).

    This is the metric a reader is likely to assume when they see "R2" in a
    property-prediction paper. We report it separately and label it clearly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    out = np.where(ss_tot == 0, np.where(ss_res == 0, 1.0, 0.0), 1.0 - ss_res / np.maximum(ss_tot, 1e-300))
    return out


# --------------------------------------------------------------------------
# Best-of-N
# --------------------------------------------------------------------------


def best_of_n_curve(
    scores: Sequence[float],
    n_values: Sequence[int],
    n_bootstrap: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[int, dict[str, float]]:
    """Expected best score when you draw N candidates and keep the maximum.

    This is the quantity the paper leaves unstated. Its headline R2 is the
    maximum over a pool of sampled candidates; the released notebook's default
    pool is 32 x 64 = 2048. Reporting max-over-2048 without reporting N makes
    the number impossible to interpret, so we sweep N and show the whole curve.

    Estimated by resampling with replacement from the observed pool, which
    assumes the pool is a fair sample of the generator's output distribution.
    That assumption holds by construction here: the pool *is* independent
    samples at fixed decoding settings.

    Returns {n: {mean, std, p05, p50, p95}}.
    """
    s = np.asarray([x for x in scores if np.isfinite(x)], dtype=float)
    if s.size == 0:
        raise ValueError("no finite scores")
    rng = rng or np.random.default_rng(0)

    out: dict[int, dict[str, float]] = {}
    for n in n_values:
        draws = rng.choice(s, size=(n_bootstrap, int(n)), replace=True).max(axis=1)
        out[int(n)] = {
            "mean": float(draws.mean()),
            "std": float(draws.std(ddof=1)) if n_bootstrap > 1 else 0.0,
            "p05": float(np.percentile(draws, 5)),
            "p50": float(np.percentile(draws, 50)),
            "p95": float(np.percentile(draws, 95)),
        }
    return out


def n_required_to_reach(scores: Sequence[float], threshold: float) -> float | None:
    """Median number of samples needed before one exceeds `threshold`.

    Derived from the empirical hit rate p: for a geometric distribution the
    median is ceil(log(0.5) / log(1 - p)). Returns None if nothing in the pool
    reaches the threshold, which is itself informative - it means the reported
    value was not reachable at this sampling budget in our run.
    """
    s = np.asarray([x for x in scores if np.isfinite(x)], dtype=float)
    hits = int(np.sum(s >= threshold))
    if hits == 0:
        return None
    p = hits / s.size
    if p >= 1.0:
        return 1.0
    return float(np.ceil(np.log(0.5) / np.log(1.0 - p)))


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


@dataclass
class ParseStats:
    """How many generations came back usable.

    The released notebook wraps parsing in a bare `except:` and moves on, so
    the failure rate is invisible. We count it. A high rate would mean the
    effective sampling budget is smaller than the nominal one, which changes
    how the best-of-N numbers should be read.
    """

    attempted: int = 0
    parsed: int = 0
    rejected_unparseable: int = 0
    rejected_invalid_aa: int = 0
    rejected_not_novel: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.attempted if self.attempted else float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parse_rate"] = self.parse_rate
        return d
