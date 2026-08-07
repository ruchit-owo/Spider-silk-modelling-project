"""How novel are the generated sequences, and how many are verbatim copies?

The paper's pipeline discards non-novel candidates before scoring, where novel
means "not a byte-identical member of ALL_SILK_SEQ.csv". It reports the
surviving sequences but not how many were discarded.

That rate is worth knowing for two reasons. It sets the effective sampling
budget - if half the generations are copies, a nominal 2,048 candidates is
really about 1,000 - and it characterises what the inverse task does by
default, which is relevant background to the novelty argument of Section 2.2.

This script also reports the softer k-mer measures (assumption A4) for the
sequences that do pass the exact-match test, because passing it is cheap: one
substituted residue in a 400-residue sequence is enough.

Runs on CPU from the saved pools.

Usage:
    python scripts/07_novelty_audit.py --sample 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, novelty, runinfo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150,
                    help="novel sequences per set to run k-mer measures on "
                         "(these are O(reference set) each, so not all of them)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    known = novelty.load_known_sequences()
    print(f"reference set: {len(known)} known silk sequences\n")

    rng = np.random.default_rng(args.seed)
    rows, kmer_rows = [], []

    for p in sorted((config.RESULTS / "pools").glob("pool_*.csv")):
        set_name = p.stem.replace("pool_", "")
        if set_name not in config.ALL_SETS:
            continue
        pool = pd.read_csv(p)
        n_total = len(pool)
        n_novel = int(pool["novel_exact"].sum())
        n_copy = n_total - n_novel

        # Among the verbatim copies, how concentrated are they? If the model
        # returns the same handful of sequences repeatedly, that is a different
        # behaviour from returning many different training sequences.
        copies = pool[~pool["novel_exact"]]["sequence"]
        n_distinct_copies = int(copies.nunique()) if len(copies) else 0
        top_share = (
            float(copies.value_counts().iloc[0] / len(copies)) if len(copies) else 0.0
        )

        rows.append(
            {
                "set": set_name,
                "n_parsed": n_total,
                "n_verbatim_copies": n_copy,
                "copy_rate": n_copy / n_total if n_total else np.nan,
                "n_novel": n_novel,
                "n_distinct_copied_sequences": n_distinct_copies,
                "most_repeated_copy_share": top_share,
                "n_distinct_novel": int(
                    pool[pool["novel_exact"]]["sequence"].nunique()
                ),
            }
        )

        nov = pool[pool["novel_exact"]]["sequence"].astype(str).tolist()
        if nov:
            take = min(args.sample, len(nov))
            idx = rng.choice(len(nov), size=take, replace=False)
            for i in idx:
                s = nov[i]
                kmer_rows.append(
                    {
                        "set": set_name,
                        "length": len(s),
                        "max_containment_k6": novelty.max_kmer_containment(s, 6, known),
                        "max_jaccard_k6": novelty.max_kmer_jaccard(s, 6, known),
                    }
                )
        print(f"{set_name}: {n_copy}/{n_total} verbatim copies "
              f"({n_copy / n_total:.1%}), {n_novel} novel")

    if not rows:
        raise SystemExit("no pools found")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "novelty_audit.csv", index=False)

    print("\nexact-match novelty by property set\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    overall_copy = float(df["n_verbatim_copies"].sum() / df["n_parsed"].sum())
    print(f"\noverall verbatim-copy rate: {overall_copy:.1%}")
    print(f"nominal budget 2048 -> effective budget "
          f"{int(2048 * (1 - overall_copy))} scoreable candidates per set")

    kdf = pd.DataFrame(kmer_rows)
    kmer_summary = None
    if not kdf.empty:
        kdf.to_csv(config.RESULTS / "novelty_kmer.csv", index=False)
        print("\nk-mer similarity of the sequences that DO pass the exact-match")
        print("test, against the same reference set (assumption A4):\n")
        kmer_summary = (
            kdf.groupby("set")[["max_containment_k6", "max_jaccard_k6"]]
            .agg(["mean", "median", "max"])
            .round(3)
        )
        print(kmer_summary.to_string())
        # Flatten the MultiIndex columns: json.dump cannot use tuple keys.
        kmer_summary = kmer_summary.copy()
        kmer_summary.columns = [f"{a}_{b}" for a, b in kmer_summary.columns]
        kmer_summary = kmer_summary.reset_index()
        overall_cont = float(kdf["max_containment_k6"].mean())
        print(f"\nmean best 6-mer containment across all sets: {overall_cont:.3f}")
        print("  Containment near 1 means nearly every 6-mer of the generated")
        print("  sequence also occurs in some known sequence. That is compatible")
        print("  with a novel arrangement of familiar parts, which is what one")
        print("  would expect of a repetitive protein family; it is not by")
        print("  itself evidence of copying, and it is not percent identity.")

    path = runinfo.write_result(
        "07_novelty_audit",
        {
            "per_set": rows,
            "overall_copy_rate": overall_copy,
            "kmer_summary": None if kmer_summary is None
            else kmer_summary.to_dict(orient="records"),
            "kmer_note": (
                "6-mer containment/Jaccard against ALL_SILK_SEQ; offline "
                "stand-in for BLAST, not an alignment statistic (A4)"
            ),
        },
        script="scripts/07_novelty_audit.py",
        params={"sample": args.sample, "seed": args.seed},
    )
    print(f"\nwrote {config.RESULTS / 'novelty_audit.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
