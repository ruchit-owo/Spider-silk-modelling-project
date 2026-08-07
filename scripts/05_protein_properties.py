"""Reproduce Figure 5: sequence descriptors for generated vs natural sequences.

Figure 5 compares each generated sequence against its two closest BLAST hits on
sequence length, molecular weight, instability index, isoelectric point, amino
acid composition and secondary-structure fractions.

We cannot reproduce it exactly. Table S1, which lists the paper's five
generated sequences and their ten BLAST neighbours, is in the Supporting
Information we were unable to obtain (docs/discrepancies.md D7). So instead of
the paper's sequences we use our own: the best-scoring generation for each
property set from scripts/02_reproduce_table1.py, and for the "natural
comparison" role, the nearest neighbours in the silkome reference set by 6-mer
containment.

That substitution is stated wherever the output is used. It changes the
comparison from "the paper's generations vs their BLAST hits" to "our
generations vs their nearest silkome neighbours", which tests the same
question - do generated sequences look like natural spidroins on standard
descriptors - with different specimens.

Both secondary-structure conventions are reported. See docs/discrepancies.md
D1 for why that is necessary.

Usage:
    python scripts/05_protein_properties.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, novelty, protparam, runinfo  # noqa: E402


nearest_neighbours = novelty.nearest_neighbours


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(config.SELFCONSISTENCY_SETS))
    args = ap.parse_args()

    known = novelty.load_known_sequences()
    rows = []
    provenance = []

    for set_name in args.sets:
        pool_path = config.RESULTS / "pools" / f"pool_{set_name}.csv"
        if not pool_path.exists():
            print(f"{set_name}: no pool; run scripts/02_reproduce_table1.py first")
            continue
        pool = pd.read_csv(pool_path)
        scored = pool[pool["forward_parsed"] & pool["novel_exact"]].copy()
        scored = scored[np.isfinite(scored["r2"])]
        if scored.empty:
            print(f"{set_name}: no scoreable candidates")
            continue

        best = scored.loc[scored["r2"].idxmax()]
        gen_seq = str(best["sequence"])
        print(f"{set_name}: best generation, length {len(gen_seq)}, "
              f"R2 {best['r2']:.4f}")

        try:
            d = protparam.descriptors(gen_seq)
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue
        rows.append({"set": set_name, "role": "generated", "name": f"{set_name}_gen", **d})

        for i, (sim, ref) in enumerate(
            nearest_neighbours(gen_seq, known, n=2), start=1
        ):
            try:
                dn = protparam.descriptors(ref)
            except ValueError:
                continue
            rows.append(
                {
                    "set": set_name,
                    "role": f"neighbour{i}",
                    "name": f"{set_name}_nb{i}",
                    "kmer_containment_to_generated": sim,
                    **dn,
                }
            )
            provenance.append(
                {"set": set_name, "neighbour": i, "containment": sim, "length": len(ref)}
            )

    if not rows:
        raise SystemExit("nothing to report; run scripts/02_reproduce_table1.py first")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "protein_properties.csv", index=False)

    key = [
        "length",
        "molecular_weight",
        "instability_index",
        "isoelectric_point",
        "frac_helix_legacy",
        "frac_sheet_legacy",
        "frac_helix_corrected",
        "frac_sheet_corrected",
    ]
    print("\nFigure 5a descriptors (legacy = the paper's caption convention)\n")
    print(df[["set", "role"] + key].to_string(index=False,
                                              float_format=lambda x: f"{x:.3f}"))

    # How close are generated sequences to their natural neighbours, per
    # descriptor? Reported as the relative gap, so descriptors on different
    # scales can be compared.
    gaps = []
    for set_name, grp in df.groupby("set"):
        gen = grp[grp["role"] == "generated"]
        nbs = grp[grp["role"].str.startswith("neighbour")]
        if gen.empty or nbs.empty:
            continue
        for col in ("molecular_weight", "instability_index", "isoelectric_point",
                    "frac_helix_legacy", "frac_sheet_legacy"):
            g = float(gen.iloc[0][col])
            n = float(nbs[col].mean())
            denom = abs(n) if abs(n) > 1e-9 else 1.0
            gaps.append({"set": set_name, "descriptor": col,
                         "generated": g, "neighbour_mean": n,
                         "relative_gap": (g - n) / denom})
    gap_df = pd.DataFrame(gaps)
    if not gap_df.empty:
        print("\nrelative gap between generated and neighbour mean, by descriptor")
        print(gap_df.pivot(index="descriptor", columns="set", values="relative_gap")
              .to_string(float_format=lambda x: f"{x:+.3f}"))

    path = runinfo.write_result(
        "05_protein_properties",
        {
            "comparison_basis": (
                "our best generation per property set vs its two nearest "
                "silkome neighbours by 6-mer containment; NOT the paper's "
                "Table S1 sequences and NOT BLAST (see discrepancies D7)"
            ),
            "descriptors": df.to_dict(orient="records"),
            "neighbour_provenance": provenance,
            "relative_gaps": gaps,
        },
        script="scripts/05_protein_properties.py",
    )
    print(f"\nwrote {config.RESULTS / 'protein_properties.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
