"""Which residues distinguish high- from low-performing silks?

A model-free, data-side analysis: sort the samples by each mechanical property,
take the extremes, and ask how their amino-acid composition differs. No model
is involved, so it says something about the dataset rather than about
SilkomeGPT - which is exactly why it is a useful companion to the ablations in
scripts/11, where the model showed no motif-specific sensitivity.

WHAT THIS REPRODUCES, AND WHAT IT DOES NOT.

This follows Supplementary Note 8 of

    A. Pandey, W. Chen, S. Keten, "Primary sequence-based data-constrained
    machine learning framework to predict mechanical properties of the spider
    silk", Commun. Mater. 2024.

It is *not* Figure 7 of Lu, Kaplan & Buehler, which this project's other
scripts reproduce. Lu et al.'s Figure 7 is a motif count and positional
analysis using the motif definitions of its Table S4; see
scripts/06_motif_analysis.py. The two are different analyses of different
quantities and are kept separate.

METHOD (their Supplementary Equations 2 and 3):

    C_a      = mean fractional composition of residue a over a set of samples
    C_a^t    = that mean over the top-10 samples by the property
    C_a^b    = that mean over the bottom-10 samples
    C_diff   = |C_a^b - C_a^t| normalised by the maximum such difference
               among residues in the SAME group (hydrophobic/polar/charged)

Two notes on the transcription, neither of which changes the result:

  - Their Supplementary Equation 2 as printed is a sum over samples of
    (count / length) with no 1/n factor, while the text calls it an average.
    We use the mean. Since C_diff compares two slices of equal size (10 and
    10) and then normalises, a constant factor cancels either way.
  - The normalisation is within-group, so the grouping determines the bar
    heights directly.

THE ASSUMPTION THIS ANALYSIS FORCES.

THE SOURCE DOES NOT SPECIFY the hydrophobic/polar/charged assignment table.
Because the normalisation is within-group, that choice changes the output. We
therefore run two standard schemes (see src/silkrepro/aa_groups.py) and report
both. Where they agree, the ambiguity did not matter. Where they disagree, the
disagreement is the result - we do not select the scheme that better matches
the published residue list.

Usage:
    python scripts/14_composition_association.py --spidroin MaSp1 --top-n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import aa_groups, config, dataio, runinfo  # noqa: E402
from silkrepro.protparam import AA_ORDER  # noqa: E402


def mean_composition(sequences: list[str]) -> dict[str, float]:
    """C_a: mean fractional composition of each residue over a set of samples."""
    if not sequences:
        return dict.fromkeys(AA_ORDER, np.nan)
    fracs = np.array(
        [[s.count(a) / len(s) for a in AA_ORDER] for s in sequences], dtype=float
    )
    return dict(zip(AA_ORDER, fracs.mean(axis=0)))


def c_diff(top: dict[str, float], bottom: dict[str, float],
           scheme: dict[str, set[str]]) -> pd.DataFrame:
    """Within-group normalised |C_b - C_t| (their Supplementary Equation 3)."""
    rows = []
    for aa in AA_ORDER:
        rows.append(
            {
                "residue": aa,
                "group": aa_groups.group_of(aa, scheme),
                "C_top": top[aa],
                "C_bottom": bottom[aa],
                "abs_diff": abs(bottom[aa] - top[aa]),
            }
        )
    df = pd.DataFrame(rows)
    df["C_diff"] = df.groupby("group")["abs_diff"].transform(
        lambda s: s / s.max() if s.max() > 0 else s * 0.0
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spidroin", default="MaSp1",
                    help="type label to restrict to; the source focuses on MaSp1")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--select-per-group", type=int, default=2,
                    help="residues per group to call 'selected', by C_diff. "
                         "Kept small deliberately: see the null below")
    ap.add_argument("--n-null", type=int, default=2000,
                    help="permutation replicates for the null model")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    for name, scheme in aa_groups.SCHEMES.items():
        aa_groups.validate(scheme)
    diffs = aa_groups.differences(aa_groups.SCHEME_A, aa_groups.SCHEME_B)
    print("amino-acid schemes differ on: "
          + ", ".join(f"{aa} ({d['scheme_a']}/{d['scheme_b']})"
                      for aa, d in diffs.items()))

    df = dataio.load_pairs()
    if args.spidroin:
        df = df[df["type"] == args.spidroin]
    print(f"\n{len(df)} {args.spidroin} sequence rows")

    # Properties are measured per individual, so the "samples" being ranked are
    # individuals, not sequences. One individual can contribute several
    # sequences; all of them go into that individual's composition.
    by_ind = (
        df.groupby("idv_id")
        .agg({**{p: "first" for p in config.PROPERTY_NAMES},
              "sequence": lambda s: list(s)})
        .reset_index()
    )
    print(f"{len(by_ind)} individuals with {args.spidroin} sequences")
    if len(by_ind) < 2 * args.top_n:
        raise SystemExit(
            f"only {len(by_ind)} individuals; need at least {2 * args.top_n}"
        )

    mech = [config.PROPERTY_NAMES[i] for i in config.MECHANICAL_IDX]
    all_rows, selections = [], []

    for prop in mech:
        ranked = by_ind.sort_values(prop)
        bottom_seqs = [s for lst in ranked.head(args.top_n)["sequence"] for s in lst]
        top_seqs = [s for lst in ranked.tail(args.top_n)["sequence"] for s in lst]

        c_top = mean_composition(top_seqs)
        c_bot = mean_composition(bottom_seqs)

        print(f"\n{prop}: bottom-{args.top_n} = "
              f"{ranked.head(args.top_n)[prop].min():.3g}"
              f"..{ranked.head(args.top_n)[prop].max():.3g}, "
              f"top-{args.top_n} = "
              f"{ranked.tail(args.top_n)[prop].min():.3g}"
              f"..{ranked.tail(args.top_n)[prop].max():.3g} "
              f"({len(bottom_seqs)} / {len(top_seqs)} sequences)")

        for scheme_name, scheme in aa_groups.SCHEMES.items():
            cd = c_diff(c_top, c_bot, scheme)
            cd["property"] = prop
            cd["scheme"] = scheme_name
            all_rows.append(cd)

            picked = (
                cd.sort_values("C_diff", ascending=False)
                .groupby("group")
                .head(args.select_per_group)["residue"]
                .tolist()
            )
            selections.append(
                {"property": prop, "scheme": scheme_name, "selected": sorted(picked)}
            )

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(config.RESULTS / "composition_association.csv", index=False)

    # --- residues selected, per scheme, pooled over properties -----------
    print("\n\nresidues with the largest within-group composition difference")
    print(f"(top {args.select_per_group} per group, per property)\n")

    summary = {}
    for scheme_name in aa_groups.SCHEMES:
        pooled: dict[str, int] = {}
        for s in selections:
            if s["scheme"] != scheme_name:
                continue
            for aa in s["selected"]:
                pooled[aa] = pooled.get(aa, 0) + 1
        chosen = sorted([a for a, c in pooled.items() if c >= 2])
        summary[scheme_name] = {"counts": pooled, "selected_in_2plus_properties": chosen}
        print(f"  {scheme_name}: {''.join(chosen)}")

    a = set(summary["A_lehninger_style"]["selected_in_2plus_properties"])
    b = set(summary["B_alternative"]["selected_in_2plus_properties"])
    print(f"\n  agreed by both schemes : {''.join(sorted(a & b))}")
    print(f"  scheme A only          : {''.join(sorted(a - b))}")
    print(f"  scheme B only          : {''.join(sorted(b - a))}")

    # ---- the null model, without which none of this means anything ------
    #
    # The selection rule takes the top k residues from each group. Groups are
    # small - the charged group has 5 members in scheme A and 4 in scheme B -
    # so with k=5 every charged residue is selected automatically, whatever
    # the data says. Measured: with k=5, uniform random C_diff recovers 12.7
    # of the 14 published residues on average, selecting 17.8 of 20. The real
    # data recovered 12. The overlap statistic was therefore carrying no
    # information at all, and an earlier version of this project reported it
    # as convergent evidence. It was not.
    #
    # The fix is a null, reported alongside the observed value every time.
    keten = set(aa_groups.KETEN_SELECTED)
    rng = np.random.default_rng(args.seed)

    def null_distribution(scheme, k, n_rep):
        out = []
        for _ in range(n_rep):
            pooled: dict[str, int] = {}
            for _prop in mech:
                noise = {aa: rng.random() for aa in AA_ORDER}
                df_n = pd.DataFrame(
                    {
                        "residue": AA_ORDER,
                        "group": [aa_groups.group_of(x, scheme) for x in AA_ORDER],
                        "C_diff": [noise[x] for x in AA_ORDER],
                    }
                )
                picked = (
                    df_n.sort_values("C_diff", ascending=False)
                    .groupby("group")
                    .head(k)["residue"]
                    .tolist()
                )
                for aa in picked:
                    pooled[aa] = pooled.get(aa, 0) + 1
            chosen = {x for x, c in pooled.items() if c >= 2}
            out.append((len(chosen & keten), len(chosen)))
        return np.array(out)

    print(f"\n  Pandey/Chen/Keten reported: {''.join(sorted(keten))}")
    print(f"\n  overlap with their list, against a permutation null "
          f"({args.n_null} replicates,\n  identical selection rule applied to "
          f"uniform random C_diff):\n")

    null_report = {}
    for nm, sel, scheme in (
        ("A_lehninger_style", a, aa_groups.SCHEME_A),
        ("B_alternative", b, aa_groups.SCHEME_B),
    ):
        null = null_distribution(scheme, args.select_per_group, args.n_null)
        obs = len(sel & keten)
        p = float(np.mean(null[:, 0] >= obs))
        null_report[nm] = {
            "observed_overlap": obs,
            "observed_n_selected": len(sel),
            "null_overlap_mean": float(null[:, 0].mean()),
            "null_overlap_p5": float(np.percentile(null[:, 0], 5)),
            "null_overlap_p95": float(np.percentile(null[:, 0], 95)),
            "null_n_selected_mean": float(null[:, 1].mean()),
            "p_value": p,
        }
        print(f"    {nm}")
        print(f"      observed : {obs}/{len(keten)} recovered, "
              f"{len(sel)}/20 residues selected")
        print(f"      null     : {null[:, 0].mean():.1f}/{len(keten)} recovered "
              f"(p5-p95 {np.percentile(null[:, 0], 5):.0f}-"
              f"{np.percentile(null[:, 0], 95):.0f}), "
              f"{null[:, 1].mean():.1f}/20 selected")
        print(f"      p        : {p:.3f}"
              + ("   (not distinguishable from chance)" if p > 0.05 else ""))

    print("\n  Read the p-value, not the raw overlap. The selection rule picks")
    print("  a fixed number of residues per group regardless of the data, so a")
    print("  high overlap is expected even from noise.")

    path = runinfo.write_result(
        "14_composition_association",
        {
            "reproduces": (
                "Pandey, Chen & Keten, Commun. Mater. 2024, Supplementary "
                "Note 8 / Supplementary Figure 7. NOT Figure 7 of Lu, Kaplan "
                "& Buehler, which is a motif analysis (see scripts/06)."
            ),
            "spidroin": args.spidroin,
            "n_sequence_rows": int(len(df)),
            "n_individuals": int(len(by_ind)),
            "top_n": args.top_n,
            "schemes": {
                k: {g: sorted(v) for g, v in s.items()}
                for k, s in aa_groups.SCHEMES.items()
            },
            "scheme_disagreements": diffs,
            "selections_per_property": selections,
            "summary": summary,
            "agreed_by_both_schemes": sorted(a & b),
            "keten_reported": sorted(keten),
            "permutation_null": null_report,
            "null_note": (
                "The top-k-per-group selection rule picks a near-fixed number "
                "of residues whatever the data says, because the groups are "
                "small. Overlap with the published list must be read against "
                "the null, not on its own."
            ),
            "assumption": (
                "The source does not give the hydrophobic/polar/charged "
                "assignment table, and the normalisation is within-group, so "
                "the grouping determines the output. Two standard schemes are "
                "reported; neither was chosen to improve agreement."
            ),
        },
        script="scripts/14_composition_association.py",
        params={"spidroin": args.spidroin, "top_n": args.top_n,
                "select_per_group": args.select_per_group},
    )
    print(f"\nwrote {config.RESULTS / 'composition_association.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
