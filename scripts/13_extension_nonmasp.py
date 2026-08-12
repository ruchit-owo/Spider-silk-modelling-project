"""Extension 4: what does the model do with spidroins it was not trained on?

SilkomeGPT is fine-tuned on MaSp only. The silkome database holds eight other
spidroin families - MiSp, Flag, AcSp, AgSp, PySp, CySp and others - recovered
from the same individuals whose fibres were tested. The forward task accepts
them without complaint.

Two questions.

  Q1. Does the prediction depend on which family the input comes from, or does
      the model largely emit the MaSp training marginal regardless? If Flag and
      MaSp inputs produce indistinguishable outputs, the forward task is
      reading less from the sequence than it appears to. This question needs no
      ground truth and carries no caveat.

  Q2. How does accuracy against the measured properties compare across
      families?

Q2 needs a caveat stated before the numbers, not after. The silkome mechanical
properties are measured on *dragline* fibre, and dragline is made of MaSp. A
Flag sequence in this table is therefore paired with the same spider's dragline
properties, not with the properties of its flagelliform silk. The paper is
explicit that its modelling assumes "the MaSp governs the mechanics of spider
silks", and restricting to MaSp follows from that assumption.

So Q2 is not "can this model predict flagelliform silk properties" - no
analysis of this dataset could answer that. It is the narrower question of
whether a non-MaSp sequence from an individual still recovers that
individual's dragline properties, which probes how much the model leans on
species-level signal rather than MaSp-specific sequence features.

Usage:
    python scripts/13_extension_nonmasp.py \
        --fasta path/to/spider-silkome-database.v1.prot.fasta \
        --mech  path/to/mechanical_properties.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, metrics, runinfo, silkome  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--mech", type=Path, required=True)
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    table = silkome.join_sequences_to_properties(args.fasta, args.mech)
    mins, maxs = silkome.normalisation_from(table)
    print(f"{len(table)} sequence/property rows across all spidroin types\n")

    counts = table["family_group"].value_counts()
    print("rows per family:")
    print(counts.to_string())

    # The model was fine-tuned for sequences up to 768 residues; longer inputs
    # would confound family effects with truncation effects.
    table = table[table["sequence"].str.len().between(80, 768)]

    rng = np.random.default_rng(args.seed)
    keep = []
    for fam, grp in table.groupby("family_group"):
        if fam == "other" or len(grp) < 10:
            continue
        idx = rng.choice(len(grp), size=min(args.per_family, len(grp)), replace=False)
        keep.append(grp.iloc[idx])
    sample = pd.concat(keep, ignore_index=True)
    print(f"\nsampled {len(sample)} sequences across "
          f"{sample['family_group'].nunique()} families "
          f"(length 80-768)")

    model = SilkomeGPT()
    preds = model.predict_properties_batch(
        sample["sequence"].astype(str).tolist(),
        settings=GenerationSettings.forward_default(),
        seed=args.seed,
        batch_size=args.batch_size,
    )
    ok = np.array([p is not None for p in preds])
    print(f"parsed {ok.sum()}/{len(preds)}\n")

    sample = sample[ok].reset_index(drop=True)
    y_pred = np.vstack([p for p in preds if p is not None])
    y_true = (sample[config.PROPERTY_NAMES].to_numpy(dtype=float) - mins) / (maxs - mins)

    for i, n in enumerate(config.PROPERTY_NAMES):
        sample[f"pred_{n}"] = y_pred[:, i]
        sample[f"true_{n}"] = y_true[:, i]
    sample.to_csv(config.RESULTS / "nonmasp_predictions.csv", index=False)

    # ---- Q1 -------------------------------------------------------------
    print("Q1  mean prediction by family")
    print("    (if family were ignored, every row here would coincide)\n")
    mech_names = [config.PROPERTY_NAMES[i] for i in config.MECHANICAL_IDX]
    rows = []
    for fam, grp in sample.groupby("family_group"):
        # Both spreads must be taken over the same four properties. Computing
        # `within` over all eight while `between` uses only the mechanical four
        # makes the ratio compare two different quantities - the SD slots vary
        # more, so it inflates the denominator and understates the ratio.
        p_mech = grp[[f"pred_{n}" for n in mech_names]].to_numpy(dtype=float)
        p_all = grp[[f"pred_{n}" for n in config.PROPERTY_NAMES]].to_numpy(dtype=float)
        rows.append(
            {
                "family": fam,
                "n": len(grp),
                **{n: float(grp[f"pred_{n}"].mean()) for n in mech_names},
                "within_family_sd": float(p_mech.std(axis=0).mean()),
                "within_family_sd_all8": float(p_all.std(axis=0).mean()),
            }
        )
    fam_df = pd.DataFrame(rows).sort_values("family")
    print(fam_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    between = float(fam_df[mech_names].to_numpy(dtype=float).std(axis=0).mean())
    within = float(fam_df["within_family_sd"].mean())
    ratio = between / within if within > 0 else float("inf")
    print(f"\n    between-family SD of the mean prediction : {between:.4f}")
    print(f"    mean within-family SD                    : {within:.4f}")
    print(f"    ratio                                    : {ratio:.3f}")
    print(f"    (both over the four mechanical properties)")
    print("\n    A ratio well below 1 means family membership shifts the")
    print("    prediction far less than sequence-to-sequence variation within")
    print("    a family does.")

    # ---- Q2 -------------------------------------------------------------
    print("\nQ2  accuracy against measured dragline properties, by family")
    print("    (read the caveat in this script's docstring first)\n")
    acc = []
    for fam, grp in sample.groupby("family_group"):
        yt = grp[[f"true_{n}" for n in config.PROPERTY_NAMES]].to_numpy(dtype=float)
        yp = grp[[f"pred_{n}" for n in config.PROPERTY_NAMES]].to_numpy(dtype=float)
        r2 = metrics.r2_across_dataset(yt, yp)
        acc.append(
            {
                "family": fam,
                "n": len(grp),
                "in_training_distribution": fam == "MaSp",
                "mean_r2_mechanical": float(r2[config.MECHANICAL_IDX].mean()),
                "mean_r2_all8": float(r2.mean()),
                "mean_abs_error": float(np.mean(np.abs(yt - yp))),
            }
        )
    acc_df = pd.DataFrame(acc).sort_values("mean_r2_mechanical", ascending=False)
    print(acc_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    acc_df.to_csv(config.RESULTS / "nonmasp_accuracy.csv", index=False)

    path = runinfo.write_result(
        "13_extension_nonmasp",
        {
            "n_evaluated": int(len(sample)),
            "family_counts": {k: int(v) for k, v in counts.items()},
            "mean_prediction_by_family": fam_df.to_dict(orient="records"),
            "between_family_sd": between,
            "within_family_sd": within,
            "between_over_within": ratio,
            "spread_basis": "four mechanical properties, both numerator and denominator",
            "accuracy_by_family": acc_df.to_dict(orient="records"),
            "caveat": (
                "Silkome mechanical properties are measured on dragline fibre, "
                "which is MaSp. Non-MaSp sequences are paired with the same "
                "individual's dragline properties, not with their own silk "
                "type's properties."
            ),
        },
        script="scripts/13_extension_nonmasp.py",
        params={"per_family": args.per_family, "seed": args.seed},
    )
    print(f"\nwrote {config.RESULTS / 'nonmasp_accuracy.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
