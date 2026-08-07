"""Forward-task accuracy over the reconstructed 1,033-pair dataset.

The paper reports R2 only in its within-vector sense: one designed sequence,
its eight predicted values against the eight values that were requested. It
does not report the ordinary regression question - given a real spidroin
sequence, how close is the predicted toughness to the measured toughness,
across many sequences?

This script answers that, per property, over all 1,033 pairs.

The result must be read with one fact in front of it. The paper states that
"the fine-tuning training set includes all known pairs of sequence and
properties". There is no held-out split. So every sequence here was seen
during fine-tuning, and these numbers are training-set numbers. They are an
upper bound on what the model would do on a new spidroin, not an estimate of
it. Nothing in this project can produce that estimate, because doing so would
require retraining with a held-out fold, which we do not do.

Reported per property rather than pooled, because pooling over the four
mechanical properties and their four standard deviations hides the case where
a model does well on the SD slots (which vary little) and poorly on the
properties themselves.

Usage:
    python scripts/04_forward_eval_dataset.py --limit 1033
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, dataio, metrics, runinfo  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N pairs (for a quick check)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    df = dataio.load_pairs()
    norm = dataio.load_normalisation()
    print(f"{len(df)} sequence/property pairs, normalisation from {norm.source}")

    if args.limit:
        df = df.head(args.limit).copy()

    y_true = dataio.normalised_targets(df, norm)
    check = dataio.check_normalisation_consistency(y_true)
    print(f"normalised targets within [0,1]: {check['in_unit_interval']}")

    model = SilkomeGPT()
    fwd = GenerationSettings.forward_default()

    sequences = df["sequence"].astype(str).tolist()
    t0 = time.time()
    preds = model.predict_properties_batch(
        sequences, settings=fwd, seed=args.seed, batch_size=args.batch_size
    )
    print(f"forward pass over {len(sequences)} sequences in {time.time() - t0:.0f}s")

    ok = np.array([p is not None for p in preds])
    print(f"parsed {ok.sum()}/{len(preds)} ({ok.mean():.1%})")
    if ok.sum() < 10:
        raise SystemExit("too few parseable predictions to score")

    y_pred = np.vstack([p for p in preds if p is not None])
    y_obs = y_true[ok]

    r2 = metrics.r2_across_dataset(y_obs, y_pred)
    mae = np.mean(np.abs(y_obs - y_pred), axis=0)
    # R2 of the trivial predictor that always emits the dataset mean is 0 by
    # construction; included so the sign of r2 is easy to read.
    corr = np.array(
        [np.corrcoef(y_obs[:, k], y_pred[:, k])[0, 1] for k in range(8)]
    )

    table = pd.DataFrame(
        {
            "property": config.PROPERTY_NAMES,
            "r2_in_sample": r2,
            "pearson_r": corr,
            "mae_normalised": mae,
            "target_sd": y_obs.std(axis=0, ddof=1),
            "pred_sd": y_pred.std(axis=0, ddof=1),
        }
    )

    print("\nforward-task accuracy over the fine-tuning set")
    print("(in-sample: the model was trained on every one of these pairs)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    mech = table.iloc[config.MECHANICAL_IDX]
    print(f"\nmean R2 over the four mechanical properties: "
          f"{mech['r2_in_sample'].mean():.4f}")
    print(f"mean R2 over the four SD slots:                "
          f"{table.iloc[config.SD_IDX]['r2_in_sample'].mean():.4f}")

    # The within-vector score, for comparability with Table 1.
    wv = np.array(
        [metrics.r2_within_vector(y_obs[i], y_pred[i]) for i in range(len(y_obs))]
    )
    print(f"\nwithin-vector R2 (the paper's definition), over real pairs:")
    print(f"  mean {wv.mean():.4f}   median {np.median(wv):.4f}   "
          f"max {wv.max():.4f}   frac>0 {np.mean(wv > 0):.3f}")

    table.to_csv(config.RESULTS / "forward_eval_dataset.csv", index=False)
    out = pd.DataFrame(
        {
            "sequence": np.array(sequences)[ok],
            **{f"true_{n}": y_obs[:, i] for i, n in enumerate(config.PROPERTY_NAMES)},
            **{f"pred_{n}": y_pred[:, i] for i, n in enumerate(config.PROPERTY_NAMES)},
            "within_vector_r2": wv,
        }
    )
    out.to_csv(config.RESULTS / "forward_eval_predictions.csv", index=False)

    path = runinfo.write_result(
        "04_forward_eval_dataset",
        {
            "n_pairs": int(len(df)),
            "n_parsed": int(ok.sum()),
            "parse_rate": float(ok.mean()),
            "normalisation_source": norm.source,
            "per_property": table.to_dict(orient="records"),
            "mean_r2_mechanical": float(mech["r2_in_sample"].mean()),
            "mean_r2_sd_slots": float(table.iloc[config.SD_IDX]["r2_in_sample"].mean()),
            "within_vector": {
                "mean": float(wv.mean()),
                "median": float(np.median(wv)),
                "max": float(wv.max()),
                "frac_positive": float(np.mean(wv > 0)),
            },
            "in_sample_warning": (
                "The paper fine-tunes on all known pairs, so every sequence "
                "scored here was in the training set. These are training-set "
                "numbers, not generalisation estimates."
            ),
        },
        script="scripts/04_forward_eval_dataset.py",
        params={"limit": args.limit, "seed": args.seed, "forward": fwd.to_dict()},
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
