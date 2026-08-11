"""Submit the paper's own published sequences to the forward task.

This is the sharpest check in the project. Table S1 gives the five sequences
the paper generated for its five self-consistency property sets, and Table 1
gives the R2 each achieved. Both the input and the expected output are
therefore known, so the forward task can be checked directly - no sampling, no
search, no candidate pool.

It separates two things that our Table 1 reproduction could not:

  - if we recover the published R2 on the published sequence, the forward task
    reproduces exactly, and the shortfall reported in RESULTS.md section 8 is
    entirely about the inverse task's search finding a different candidate;
  - if we do not, something differs in the scoring path itself, and the search
    explanation is insufficient.

Also reproduces Figures 5 and 7 against the paper's specimens rather than
nearest-neighbour substitutes: descriptors and motif counts for the generated
sequence and its two BLAST neighbours, per property set.

Usage:
    python scripts/09_verify_published_sequences.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, metrics, motifs, novelty, protparam, runinfo  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402

SET_KEY = {1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-s1", type=Path,
                    default=config.DATA_RAW / "table_s1_sequences.csv")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if not args.table_s1.exists():
        raise SystemExit(
            f"{args.table_s1} not found. Run scripts/08_extract_table_s1.py first."
        )
    # dtype=str on `id`: values like "1.1" are read as float64 otherwise,
    # which turns "1.10" into 1.1 and breaks every join on it.
    s1 = pd.read_csv(args.table_s1, dtype={"id": str})
    print(f"{len(s1)} sequences from Table S1\n")

    model = SilkomeGPT()
    fwd = GenerationSettings.forward_default()

    # ---- the headline check ---------------------------------------------
    print("=" * 78)
    print("forward task on the paper's own generated sequences")
    print("=" * 78)

    gen = s1[s1["role"] == "generated"].sort_values("set")
    rows = []
    for _, r in gen.iterrows():
        set_name = SET_KEY[int(r["set"])]
        target = config.SELFCONSISTENCY_SETS[set_name]
        paper_r2 = config.SELFCONSISTENCY_R2_PAPER[set_name]
        seq = str(r["sequence"])

        n_tok = len(model.tokenizer.encode(seq))
        pred = model.predict_properties(seq, fwd, seed=args.seed)

        entry = {
            "set": set_name,
            "id": r["id"],
            "length": len(seq),
            "n_tokens": n_tok,
            "paper_r2": paper_r2,
            "novel_exact": novelty.is_novel_exact(seq),
        }
        if pred is None:
            entry["our_r2"] = np.nan
            entry["parse_failed"] = True
            print(f"{set_name}: forward task produced no parseable output "
                  f"(length {len(seq)}, {n_tok} tokens)")
        else:
            our_r2 = metrics.r2_within_vector(target, pred)
            entry.update(
                {
                    "our_r2": our_r2,
                    "parse_failed": False,
                    "difference": our_r2 - paper_r2,
                    **{f"pred_{n}": float(pred[i])
                       for i, n in enumerate(config.PROPERTY_NAMES)},
                }
            )
            print(f"{set_name}: paper {paper_r2:.4f}   ours {our_r2:+.4f}   "
                  f"difference {our_r2 - paper_r2:+.4f}   "
                  f"(len {len(seq)}, {n_tok} tokens)")
            print(f"     target    {[round(v, 3) for v in target]}")
            print(f"     predicted {[round(float(v), 3) for v in pred]}")
        rows.append(entry)

    verify = pd.DataFrame(rows)
    verify.to_csv(config.RESULTS / "published_sequence_verification.csv", index=False)

    ok = verify.dropna(subset=["our_r2"])
    if not ok.empty:
        print(f"\nmean |difference| from the published R2: "
              f"{ok['difference'].abs().mean():.4f}")
        n_close = int((ok["difference"].abs() < 0.01).sum())
        print(f"{n_close}/{len(ok)} within 0.01 of the published value")

    # ---- Figure 5 and 7, against the paper's own specimens --------------
    print("\n" + "=" * 78)
    print("Figures 5 and 7 against the paper's specimens")
    print("=" * 78)

    motif_set = motifs.load_motif_set()
    print(f"motif source: {motif_set.source}\n")

    named = {str(r["id"]): str(r["sequence"]) for _, r in s1.iterrows()}
    counts = motifs.motif_count_table(named, motif_set, per_100=True)
    counts = counts.merge(
        s1[["id", "set", "role", "protparam_safe"]].rename(columns={"id": "name"}),
        on="name", how="left",
    )
    counts.to_csv(config.RESULTS / "published_motif_counts.csv", index=False)

    per100 = [c for c in counts.columns if c.startswith("per100_")]
    agreement = []
    for set_no, grp in counts.groupby("set"):
        g = grp[grp["role"] == "generated"]
        nbs = grp[grp["role"].isin(["blast1", "blast2"])]
        if g.empty or nbs.empty:
            continue
        gv = g.iloc[0][per100].to_numpy(dtype=float)
        nv = nbs[per100].to_numpy(dtype=float).mean(axis=0)
        corr = (
            float(np.corrcoef(gv, nv)[0, 1])
            if np.std(gv) > 0 and np.std(nv) > 0 else np.nan
        )
        agreement.append(
            {
                "set": SET_KEY[int(set_no)],
                "pearson_r_density_profile": corr,
                "mean_abs_density_gap": float(np.mean(np.abs(gv - nv))),
            }
        )
    agree_df = pd.DataFrame(agreement)
    print("motif-density agreement, generated vs its two BLAST neighbours")
    print("(the paper's own specimens, its own motif set)\n")
    print(agree_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # descriptors, skipping sequences with ambiguity codes
    safe = s1[s1["protparam_safe"]]
    desc = protparam.descriptor_table(
        {str(r["id"]): str(r["sequence"]) for _, r in safe.iterrows()}
    )
    desc = desc.merge(
        s1[["id", "set", "role"]].rename(columns={"id": "name"}), on="name", how="left"
    )
    desc.to_csv(config.RESULTS / "published_protein_properties.csv", index=False)
    print(f"\ndescriptors computed for {len(desc)}/{len(s1)} sequences "
          f"({len(s1) - len(desc)} excluded for ambiguity codes)")

    runinfo.write_result(
        "09_verify_published_sequences",
        {
            "forward_check": rows,
            "mean_abs_difference": (
                None if ok.empty else float(ok["difference"].abs().mean())
            ),
            "motif_source": motif_set.source,
            "motif_agreement": agreement,
            "n_descriptors": len(desc),
        },
        script="scripts/09_verify_published_sequences.py",
        params={"seed": args.seed, "forward": fwd.to_dict()},
    )
    print(f"\nwrote {config.RESULTS / 'published_sequence_verification.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
