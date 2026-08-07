"""How much of the forward task's accuracy is memorised training labels?

Motivation. scripts/04 measures the forward task against the 1,033 pairs and
gets a respectable in-sample R2. But the paper fine-tunes on all 1,033 pairs
and holds nothing out, so a model that had simply memorised the label attached
to each training sequence would score exactly the same way. The two are
indistinguishable from the aggregate R2 alone.

They are distinguishable per-row. If the model has memorised a pair, its
prediction will not merely be close to the label, it will be the label - all
eight values, to the three decimals the model emits. If it is computing
something from the sequence, its predictions will be close but not exact, and
the error will look like regression error rather than a spike at zero.

So this script splits the dataset by whether the label was reproduced exactly,
and reports accuracy separately on each part.

HOW TO READ THE RESULT, INCLUDING WHAT IT CANNOT SHOW.

The accuracy on non-exactly-reproduced rows is NOT an unbiased estimate of
generalisation. Those rows are a selected subset: sequences the model failed to
memorise may be systematically harder, rarer, longer, or from
under-represented taxa. Selecting on the outcome biases the estimate downward
by an unknown amount. It is a diagnostic, not a held-out score, and the only
way to get a held-out score is to retrain with a fold held out, which this
project does not do.

What the split does establish is the composition of the aggregate number: how
much of it rests on rows where the model returned the training label verbatim.

This also bears on nothing the paper claims. The paper reports self-consistency
between its inverse and forward passes, describes its fine-tuning set as all
known pairs, and does not claim held-out predictive accuracy. This script
characterises the model, not a shortfall against a stated claim.

Usage:
    python scripts/04b_memorisation_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, metrics, runinfo  # noqa: E402

# The model emits three decimals, so two values agreeing to within 5e-4 are the
# same number as far as the model can express it.
EXACT_TOL = 5e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=EXACT_TOL)
    args = ap.parse_args()

    path = config.RESULTS / "forward_eval_predictions.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/04_forward_eval_dataset.py first."
        )
    d = pd.read_csv(path)
    y_true = d[[f"true_{n}" for n in config.PROPERTY_NAMES]].to_numpy(dtype=float)
    y_pred = d[[f"pred_{n}" for n in config.PROPERTY_NAMES]].to_numpy(dtype=float)

    exact = np.all(np.abs(y_true - y_pred) < args.tol, axis=1)
    n, n_exact = len(d), int(exact.sum())

    print(f"{n} scored fine-tuning pairs")
    print(f"label reproduced exactly (all 8 within {args.tol:g}): "
          f"{n_exact} = {n_exact / n:.1%}")
    print(f"mean |error| on those rows      : "
          f"{np.abs(y_true[exact] - y_pred[exact]).mean():.6f}")
    print(f"mean |error| on the other {n - n_exact:>4} rows: "
          f"{np.abs(y_true[~exact] - y_pred[~exact]).mean():.4f}")

    r2_all = metrics.r2_across_dataset(y_true, y_pred)
    r2_rest = (
        metrics.r2_across_dataset(y_true[~exact], y_pred[~exact])
        if (~exact).sum() > 10
        else np.full(8, np.nan)
    )

    table = pd.DataFrame(
        {
            "property": config.PROPERTY_NAMES,
            "r2_all_rows": r2_all,
            "r2_non_memorised_rows": r2_rest,
        }
    )
    print("\nR2 over all rows, and over rows whose label was NOT reproduced "
          "exactly\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    mech_all = float(r2_all[config.MECHANICAL_IDX].mean())
    mech_rest = float(r2_rest[config.MECHANICAL_IDX].mean())
    print(f"\nmean over the four mechanical properties:")
    print(f"  all rows            {mech_all:+.4f}")
    print(f"  non-memorised rows  {mech_rest:+.4f}")
    print(f"\nA value below zero means worse than predicting the dataset mean.")

    # Are the non-memorised rows systematically different? If they are, that is
    # the selection effect the docstring warns about, made visible.
    lengths = d["sequence"].astype(str).str.len().to_numpy()
    print(f"\nselection check - are the two groups comparable?")
    print(f"  median sequence length, memorised     : "
          f"{np.median(lengths[exact]):.0f}")
    print(f"  median sequence length, non-memorised : "
          f"{np.median(lengths[~exact]):.0f}")
    print("  If these differ substantially, the non-memorised rows are a")
    print("  biased subset and their R2 understates generalisation by an")
    print("  unknown amount.")

    runinfo.write_result(
        "04b_memorisation_audit",
        {
            "n_rows": n,
            "n_exact": n_exact,
            "exact_fraction": n_exact / n,
            "tolerance": args.tol,
            "mean_abs_error_memorised": float(
                np.abs(y_true[exact] - y_pred[exact]).mean()
            ),
            "mean_abs_error_other": float(
                np.abs(y_true[~exact] - y_pred[~exact]).mean()
            ),
            "r2_all_rows": r2_all.tolist(),
            "r2_non_memorised_rows": r2_rest.tolist(),
            "mean_r2_mechanical_all": mech_all,
            "mean_r2_mechanical_non_memorised": mech_rest,
            "median_length_memorised": float(np.median(lengths[exact])),
            "median_length_non_memorised": float(np.median(lengths[~exact])),
            "caveat": (
                "Rows are selected on the outcome, so the non-memorised R2 is "
                "a diagnostic, not an unbiased generalisation estimate. A "
                "held-out score would require retraining, which this project "
                "does not do."
            ),
        },
        script="scripts/04b_memorisation_audit.py",
        params={"tol": args.tol},
    )
    print(f"\nwrote {config.RESULTS / '04b_memorisation_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
