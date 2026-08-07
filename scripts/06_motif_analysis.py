"""Figure 7: motif counts and positional distributions.

Whether this script reproduces Figure 7 or merely resembles it depends on one
input.

  With data/raw/table_s4_motifs.csv present, it uses the paper's own motif
  definitions and the analysis is a reproduction.

  Without it, it falls back to canonical spidroin motifs (poly-A, GGX, GPGXX,
  GA, QQ). That is a different motif set, so it is a different analysis. Every
  output is stamped motif_source = canonical_fallback, and nothing produced
  this way is described as reproducing Figure 7.

We were unable to obtain the Supporting Information, so the fallback is what
runs here. See docs/discrepancies.md D7.

Two reporting choices, both stated because Figure 7 does not make them:

  - Raw counts scale with sequence length, and generated sequences are often
    shorter than the natural sequences they are compared against, so raw
    counts confound motif usage with length. Figure 7a plots raw counts; we
    report raw counts *and* counts per 100 residues.
  - Matches are counted non-overlapping. For run motifs like A{4,} that is
    correct (one tract of nine alanines is one poly-A tract). Overlapping
    counts are also computed for comparison.

Usage:
    python scripts/06_motif_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, motifs, novelty, runinfo  # noqa: E402
from silkrepro.metrics import r2_within_vector  # noqa: F401  (kept for parity)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(config.SELFCONSISTENCY_SETS))
    ap.add_argument("--n-neighbours", type=int, default=2)
    args = ap.parse_args()

    motif_set = motifs.load_motif_set()
    print(f"motif source: {motif_set.source}")
    if motif_set.source == "canonical_fallback":
        print("  Table S4 unavailable. This is NOT a reproduction of Figure 7;")
        print("  it is the same analysis run on canonical spidroin motifs.")
    print(f"  motifs: {list(motif_set.patterns)}\n")

    known = novelty.load_known_sequences()
    nearest_neighbours = novelty.nearest_neighbours
    named: dict[str, str] = {}
    roles = []

    for set_name in args.sets:
        pool_path = config.RESULTS / "pools" / f"pool_{set_name}.csv"
        if not pool_path.exists():
            continue
        pool = pd.read_csv(pool_path)
        scored = pool[pool["forward_parsed"] & pool["novel_exact"]]
        scored = scored[np.isfinite(scored["r2"])]
        if scored.empty:
            continue
        gen = str(scored.loc[scored["r2"].idxmax(), "sequence"])
        named[f"{set_name}_gen"] = gen
        roles.append({"name": f"{set_name}_gen", "set": set_name, "role": "generated"})
        for i, (sim, ref) in enumerate(
            nearest_neighbours(gen, known, n=args.n_neighbours), start=1
        ):
            named[f"{set_name}_nb{i}"] = ref
            roles.append(
                {
                    "name": f"{set_name}_nb{i}",
                    "set": set_name,
                    "role": f"neighbour{i}",
                    "containment": sim,
                }
            )

    if not named:
        raise SystemExit("no pools found; run scripts/02_reproduce_table1.py first")

    counts = motifs.motif_count_table(named, motif_set, per_100=True)
    counts = counts.merge(pd.DataFrame(roles), on="name", how="left")
    counts.to_csv(config.RESULTS / "motif_counts.csv", index=False)

    positions = motifs.motif_position_table(named, motif_set)
    positions = positions.merge(pd.DataFrame(roles), on="name", how="left")
    positions.to_csv(config.RESULTS / "motif_positions.csv", index=False)

    count_cols = [c for c in counts.columns if c.startswith("count_")]
    per100_cols = [c for c in counts.columns if c.startswith("per100_")]

    print("motif counts (raw)\n")
    print(counts[["name", "role", "length"] + count_cols].to_string(index=False))
    print("\nmotif density (per 100 residues)\n")
    print(
        counts[["name", "role", "length"] + per100_cols].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )

    # Does the generated sequence resemble its natural neighbours in motif
    # usage? Compared on density, not raw count, for the length reason above.
    agreement = []
    for set_name, grp in counts.groupby("set"):
        gen = grp[grp["role"] == "generated"]
        nbs = grp[grp["role"].str.startswith("neighbour")]
        if gen.empty or nbs.empty:
            continue
        g = gen.iloc[0][per100_cols].to_numpy(dtype=float)
        n = nbs[per100_cols].to_numpy(dtype=float).mean(axis=0)
        if np.all(np.isfinite(g)) and np.all(np.isfinite(n)) and n.std() > 0:
            corr = float(np.corrcoef(g, n)[0, 1])
        else:
            corr = float("nan")
        agreement.append(
            {
                "set": set_name,
                "pearson_r_density_profile": corr,
                "mean_abs_density_gap": float(np.mean(np.abs(g - n))),
            }
        )

    if agreement:
        print("\nagreement between generated and neighbour motif-density profiles")
        print(pd.DataFrame(agreement).to_string(index=False,
                                                float_format=lambda x: f"{x:.3f}"))
        print("\n  Each correlation is over "
              f"{len(per100_cols)} motifs, so it is a weak statistic; it is "
              "reported as a summary, not a test.")

    path = runinfo.write_result(
        "06_motif_analysis",
        {
            "motif_source": motif_set.source,
            "is_figure7_reproduction": motif_set.source == "table_s4",
            "motifs": motif_set.patterns,
            "counts": counts.to_dict(orient="records"),
            "agreement": agreement,
            "n_position_records": int(len(positions)),
        },
        script="scripts/06_motif_analysis.py",
        params={"sets": args.sets, "n_neighbours": args.n_neighbours},
    )
    print(f"\nwrote {config.RESULTS / 'motif_counts.csv'}")
    print(f"wrote {config.RESULTS / 'motif_positions.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
