"""Extension 2: how much of the reported R2 is the selection procedure?

The paper's headline numbers are, in its own words, "the results with the
highest R2 values are selected from a set of sampling attempts". It does not
say how large that set is. The released notebook does: 32 samples per step,
64 repeats, then `np.argmax` over the pooled R2 values. That is a maximum over
up to 2,048 candidates.

A maximum over N draws is a statistic about the *procedure*, not only about
the model, and it grows with N without any change to the model. This script
takes the candidate pools written by 02_reproduce_table1.py and separates the
two contributions:

    what the model does        the distribution of R2 over the pool
    what the selection does    how the maximum grows with N

Neither is a criticism. Best-of-N selection is a legitimate design procedure -
if you can generate 2,000 candidate sequences cheaply and screen them, taking
the best is exactly what you should do. The point is that a reader given only
the maximum cannot tell how much sampling produced it, cannot compare it to
another method's single-shot number, and cannot estimate the compute a
practitioner would need to hit the same value. This script supplies all three.

It runs entirely on the saved pools; no GPU needed.

Usage:
    python scripts/10_ablation_bestofn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, metrics, runinfo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    pool_dir = config.RESULTS / "pools"
    pools = sorted(pool_dir.glob("pool_*.csv"))
    if not pools:
        raise SystemExit(
            f"no candidate pools in {pool_dir}. "
            f"Run scripts/02_reproduce_table1.py first."
        )

    rng = np.random.default_rng(args.seed)
    n_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    per_set = []
    curve_rows = []

    for p in pools:
        set_name = p.stem.replace("pool_", "")
        if set_name not in config.ALL_SETS:
            continue
        df = pd.read_csv(p)
        scored = df[df["forward_parsed"] & df["novel_exact"]]
        r2 = scored["r2"].to_numpy(dtype=float)
        r2 = r2[np.isfinite(r2)]
        if r2.size < 4:
            print(f"{set_name}: only {r2.size} scored candidates, skipping")
            continue

        paper = config.ALL_R2_PAPER[set_name]
        # Capped at pool/10: past that the bootstrap converges on the pool
        # maximum instead of estimating best-of-N independently of it.
        ns = metrics.reliable_n_grid(r2.size, n_grid)
        curve = metrics.best_of_n_curve(r2, ns, rng=rng)

        entry = {
            "set": set_name,
            "n_pool": int(r2.size),
            "paper_r2": paper,
            "r2_single_sample_mean": float(r2.mean()),
            "r2_single_sample_median": float(np.median(r2)),
            "r2_pool_max": float(r2.max()),
            "frac_at_or_above_paper": float(np.mean(r2 >= paper)),
            "n_to_reach_paper_median": metrics.n_required_to_reach(r2, paper),
            "best_of_n": curve,
        }
        # How much of the pool maximum is attributable to selection rather
        # than to a typical draw.
        entry["selection_gain"] = entry["r2_pool_max"] - entry["r2_single_sample_mean"]
        per_set.append(entry)

        for n, stats in curve.items():
            curve_rows.append({"set": set_name, "n": n, **stats})

    if not per_set:
        raise SystemExit("no usable pools")

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(config.RESULTS / "bestofn_curves.csv", index=False)

    tbl = pd.DataFrame(
        [{k: v for k, v in e.items() if k != "best_of_n"} for e in per_set]
    )
    tbl.to_csv(config.RESULTS / "bestofn_summary.csv", index=False)

    print("\nR2 by property set: what one sample gives, vs the pool maximum\n")
    print(
        tbl[
            [
                "set",
                "n_pool",
                "paper_r2",
                "r2_single_sample_mean",
                "r2_single_sample_median",
                "r2_pool_max",
                "selection_gain",
                "frac_at_or_above_paper",
                "n_to_reach_paper_median",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nexpected best-of-N (mean over bootstrap resamples of the pool)\n")
    pivot = curves.pivot(index="set", columns="n", values="mean")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    path = runinfo.write_result(
        "10_ablation_bestofn",
        {"per_set": per_set},
        script="scripts/10_ablation_bestofn.py",
        params={"seed": args.seed, "n_grid": n_grid},
    )
    print(f"\nwrote {config.RESULTS / 'bestofn_summary.csv'}")
    print(f"wrote {config.RESULTS / 'bestofn_curves.csv'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
