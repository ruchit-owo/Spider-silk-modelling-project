"""A composition-only baseline for the forward (sequence -> properties) task.

Purpose: establish how much of SilkomeGPT's forward performance needs a
253.6M-parameter transformer. If ridge regression on amino-acid counts gets
close, that reframes what the transformer is contributing - not as a criticism
of the paper, which does not claim otherwise, but because the comparison is
absent and a reader cannot make it themselves.

One comparison rule has to be stated before any numbers are read.

The paper fine-tunes on "all known pairs of sequence and properties"
(Experimental Section). There is no held-out split. So when the released
checkpoint predicts properties for a sequence in the Silkome 1,033, it is
predicting on data it was trained on. Any comparison must respect that:

    in-sample     baseline fit on all 1,033 and scored on all 1,033.
                  This is the like-for-like comparison against the released
                  checkpoint. Both numbers are optimistic; they are optimistic
                  in the same way.

    cross-validated  baseline scored by K-fold CV.
                  This is the honest generalisation estimate for the baseline.
                  There is no equivalent number for SilkomeGPT, because
                  producing one would require retraining the model with a
                  held-out fold, which we do not do.

Reporting only the CV baseline against the in-sample transformer would
understate the baseline; reporting only in-sample would overstate it. Both are
produced and both are labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .ablations import CANONICAL_MOTIFS, find_motif_spans
from .protparam import AA_ORDER

DIPEPTIDES = ["".join(p) for p in product(AA_ORDER, repeat=2)]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def composition_features(sequence: str) -> dict[str, float]:
    """20 amino-acid fractions plus length. No order information at all."""
    n = len(sequence)
    out = {f"aa_{a}": sequence.count(a) / n for a in AA_ORDER}
    out["log_length"] = float(np.log10(n))
    return out


def dipeptide_features(sequence: str) -> dict[str, float]:
    """400 dipeptide fractions. Minimal order information: adjacency only."""
    n = max(len(sequence) - 1, 1)
    counts = dict.fromkeys(DIPEPTIDES, 0)
    for i in range(len(sequence) - 1):
        d = sequence[i : i + 2]
        if d in counts:
            counts[d] += 1
    return {f"dp_{d}": c / n for d, c in counts.items()}


def motif_density_features(sequence: str) -> dict[str, float]:
    """Occurrences per 100 residues of each canonical spidroin motif."""
    n = len(sequence)
    return {
        f"motif_{name}": 100.0 * len(find_motif_spans(sequence, pat)) / n
        for name, pat in CANONICAL_MOTIFS.items()
    }


FEATURE_SETS = {
    "composition": lambda s: composition_features(s),
    "composition+motifs": lambda s: {**composition_features(s), **motif_density_features(s)},
    "dipeptide": lambda s: {**composition_features(s), **dipeptide_features(s)},
}


def build_feature_matrix(sequences: list[str], feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; have {list(FEATURE_SETS)}")
    fn = FEATURE_SETS[feature_set]
    return pd.DataFrame([fn(s) for s in sequences])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass
class BaselineScores:
    feature_set: str
    model: str
    r2_in_sample: np.ndarray      # per property, fit and scored on everything
    r2_cv_mean: np.ndarray        # per property, K-fold CV
    r2_cv_std: np.ndarray
    r2_trivial: np.ndarray        # predict the training mean, CV'd

    def to_frame(self, property_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "property": property_names,
                "feature_set": self.feature_set,
                "model": self.model,
                "r2_in_sample": self.r2_in_sample,
                "r2_cv_mean": self.r2_cv_mean,
                "r2_cv_std": self.r2_cv_std,
                "r2_cv_predict_mean_baseline": self.r2_trivial,
            }
        )


def _make_model(name: str, seed: int):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 30)),
        )
    if name == "gbm":
        return MultiOutputRegressor(
            GradientBoostingRegressor(random_state=seed, n_estimators=300, max_depth=3)
        )
    if name == "mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown model {name!r}")


def evaluate_baseline(
    sequences: list[str],
    targets: np.ndarray,
    feature_set: str = "composition",
    model: str = "ridge",
    n_splits: int = 5,
    seed: int = 0,
) -> BaselineScores:
    """Fit and score a baseline both in-sample and by K-fold CV.

    `targets` is (N, 8) in the config.PROPERTY_NAMES order.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    from .metrics import r2_across_dataset

    X = build_feature_matrix(sequences, feature_set).to_numpy()
    y = np.asarray(targets, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{X.shape[0]} sequences but {y.shape[0]} target rows")

    est = _make_model(model, seed)
    est.fit(X, y)
    r2_in = r2_across_dataset(y, est.predict(X))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_r2, fold_r2_trivial = [], []
    for train_idx, test_idx in kf.split(X):
        m = clone(_make_model(model, seed))
        m.fit(X[train_idx], y[train_idx])
        fold_r2.append(r2_across_dataset(y[test_idx], m.predict(X[test_idx])))

        d = clone(_make_model("mean", seed))
        d.fit(X[train_idx], y[train_idx])
        fold_r2_trivial.append(r2_across_dataset(y[test_idx], d.predict(X[test_idx])))

    fold_r2 = np.vstack(fold_r2)
    return BaselineScores(
        feature_set=feature_set,
        model=model,
        r2_in_sample=r2_in,
        r2_cv_mean=fold_r2.mean(axis=0),
        r2_cv_std=fold_r2.std(axis=0, ddof=1),
        r2_trivial=np.vstack(fold_r2_trivial).mean(axis=0),
    )


def within_vector_baseline(
    sequences: list[str],
    targets: np.ndarray,
    feature_set: str = "composition",
    model: str = "ridge",
    seed: int = 0,
) -> np.ndarray:
    """Baseline predictions scored the paper's way, one R2 per sequence.

    Lets the baseline be placed on the same axis as the paper's Table 1
    numbers, which the across-dataset R2 cannot do.
    """
    from .metrics import r2_within_vector

    X = build_feature_matrix(sequences, feature_set).to_numpy()
    y = np.asarray(targets, dtype=float)
    est = _make_model(model, seed)
    est.fit(X, y)
    pred = est.predict(X)
    return np.array([r2_within_vector(y[i], pred[i]) for i in range(len(y))])
