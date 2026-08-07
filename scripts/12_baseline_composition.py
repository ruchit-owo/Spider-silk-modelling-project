"""Extension 3: how far does amino-acid composition alone get you?

Fits ridge and gradient-boosted regressors on hand-made features - amino-acid
fractions, dipeptide fractions, canonical motif densities, sequence length -
and scores them on the same 1,033 pairs the transformer was fine-tuned on.

Two numbers are produced for each baseline and they answer different questions:

    in-sample     fit on all 1,033, scored on all 1,033. This is the
                  like-for-like comparison against SilkomeGPT's forward task,
                  because the paper fine-tunes on all known pairs and so its
                  predictions on these sequences are also in-sample. Both
                  numbers are optimistic in the same way.

    5-fold CV     the honest generalisation estimate for the baseline. There
                  is no equivalent number for SilkomeGPT in this project.

Reporting only one of the two would mislead in one direction or the other, so
both are always printed together.

This is a control, not a competitor. Nothing here shows the transformer is
unnecessary: the baseline cannot generate sequences, which is the paper's
actual contribution, and it needs the labelled dataset at fit time. What it
establishes is a floor, so that the forward task's accuracy can be read
against something.

Runs on CPU.

Usage:
    python scripts/12_baseline_composition.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import baseline, config, dataio, metrics, runinfo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    df = dataio.load_pairs()
    norm = dataio.load_normalisation()
    sequences = df["sequence"].astype(str).tolist()
    y = dataio.normalised_targets(df, norm)
    print(f"{len(sequences)} pairs, normalisation from {norm.source}\n")

    # Grouped CV would be the stricter choice, since five sequences appear
    # twice with different labels and near-identical spidroins from related
    # species are common. We report plain K-fold to match the simplest
    # reading, and additionally a species-grouped split, because leakage
    # between close relatives would inflate the plain number.
    results = []
    frames = []
    for feature_set in ("composition", "composition+motifs", "dipeptide"):
        for model in ("ridge", "gbm"):
            print(f"fitting {model} on {feature_set} ...")
            sc = baseline.evaluate_baseline(
                sequences,
                y,
                feature_set=feature_set,
                model=model,
                n_splits=args.folds,
                seed=args.seed,
            )
            frame = sc.to_frame(config.PROPERTY_NAMES)
            frames.append(frame)
            results.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "mean_r2_in_sample_mechanical": float(
                        sc.r2_in_sample[config.MECHANICAL_IDX].mean()
                    ),
                    "mean_r2_cv_mechanical": float(
                        sc.r2_cv_mean[config.MECHANICAL_IDX].mean()
                    ),
                    "mean_r2_in_sample_all8": float(sc.r2_in_sample.mean()),
                    "mean_r2_cv_all8": float(sc.r2_cv_mean.mean()),
                    "per_property": frame.to_dict(orient="records"),
                }
            )

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(config.RESULTS / "baseline_per_property.csv", index=False)

    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "per_property"} for r in results]
    )
    summary.to_csv(config.RESULTS / "baseline_summary.csv", index=False)

    print("\nbaseline R2, averaged over the four mechanical properties")
    print("(SD slots excluded - they vary little and inflate the average)\n")
    print(
        summary[
            [
                "feature_set",
                "model",
                "mean_r2_in_sample_mechanical",
                "mean_r2_cv_mechanical",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nper-property detail for the best cross-validated baseline:")
    best = summary.loc[summary["mean_r2_cv_mechanical"].idxmax()]
    detail = full[
        (full["feature_set"] == best["feature_set"]) & (full["model"] == best["model"])
    ]
    print(f"  {best['model']} on {best['feature_set']}")
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- within-vector scores, for comparability with Table 1 -------------
    wv = baseline.within_vector_baseline(
        sequences, y, feature_set="composition", model="ridge", seed=args.seed
    )
    print("\nwithin-vector R2 of the composition ridge baseline on real pairs")
    print("(the paper's R2 definition, so directly comparable to Table 1):")
    print(f"  mean {wv.mean():.4f}   median {np.median(wv):.4f}   "
          f"max {wv.max():.4f}   frac>0 {np.mean(wv > 0):.3f}")

    # --- species-grouped CV ----------------------------------------------
    grouped = None
    if "species" in df.columns:
        from sklearn.model_selection import GroupKFold
        from sklearn.base import clone

        X = baseline.build_feature_matrix(sequences, "composition").to_numpy()
        groups = (df["genus"].astype(str) + "_" + df["species"].astype(str)).to_numpy()
        gkf = GroupKFold(n_splits=min(args.folds, len(set(groups))))
        fold_r2 = []
        for tr, te in gkf.split(X, y, groups):
            m = clone(baseline._make_model("ridge", args.seed))
            m.fit(X[tr], y[tr])
            fold_r2.append(metrics.r2_across_dataset(y[te], m.predict(X[te])))
        fold_r2 = np.vstack(fold_r2)
        grouped = fold_r2.mean(axis=0)
        print("\nspecies-grouped CV (ridge on composition), so that no species "
              "appears in both train and test:")
        for n, v in zip(config.PROPERTY_NAMES, grouped):
            print(f"  {n:14s} {v:+.4f}")
        print(f"  mean over mechanical properties: "
              f"{grouped[config.MECHANICAL_IDX].mean():+.4f}")
        print("\n  The gap between this and the plain K-fold number is the "
              "leakage between related species.")

    path = runinfo.write_result(
        "12_baseline_composition",
        {
            "n_pairs": len(sequences),
            "normalisation_source": norm.source,
            "results": results,
            "within_vector_composition_ridge": {
                "mean": float(wv.mean()),
                "median": float(np.median(wv)),
                "max": float(wv.max()),
                "frac_positive": float(np.mean(wv > 0)),
            },
            "species_grouped_cv_ridge_composition": (
                None if grouped is None else grouped.tolist()
            ),
        },
        script="scripts/12_baseline_composition.py",
        params={"folds": args.folds, "seed": args.seed},
    )
    print(f"\nwrote {config.RESULTS / 'baseline_summary.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
