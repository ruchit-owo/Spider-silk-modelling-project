"""Reproduce Table 1 / Figures 3 and 4: the cycle-consistent design loop.

Procedure, following the authors' released notebook (generate_new and
generate_new_and_find_best):

    for each target property set t:
        sample N candidate sequences from GenerateSilkContent<t>
        keep those that parse and are novel by exact match
        for each kept sequence s:
            predict properties p = CalculateSilkContent<s>
            score R2(t, p) across the eight entries
        the paper reports max(R2)

We record the whole pool, not just the maximum. The maximum is what the paper
reports and is printed alongside its published value; the mean, median and
best-of-N curve are what make that maximum interpretable, and the paper does
not give them because it does not state N.

The output of this script feeds 10_ablation_bestofn.py and the figures. It
writes incrementally so a long run can be interrupted and resumed.

Usage:
    python scripts/02_reproduce_table1.py --n 128            # pilot
    python scripts/02_reproduce_table1.py --n 2048           # paper's budget
    python scripts/02_reproduce_table1.py --sets S1 S4 --n 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, metrics, novelty, runinfo  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402


def pool_path(set_name: str) -> Path:
    return config.RESULTS / "pools" / f"pool_{set_name}.csv"


def load_existing_pool(set_name: str) -> pd.DataFrame:
    p = pool_path(set_name)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def run_one_set(
    model: SilkomeGPT,
    set_name: str,
    target: list[float],
    n_samples: int,
    batch_size: int,
    seed: int,
    resume: bool,
    faithful_sampling: bool = False,
    fwd_batch_size: int = 16,
) -> pd.DataFrame:
    existing = load_existing_pool(set_name) if resume else pd.DataFrame()
    already = len(existing)
    if already >= n_samples:
        print(f"  {set_name}: {already} candidates already on disk, skipping")
        return existing

    to_generate = n_samples - already
    print(f"  {set_name}: target {target}")
    print(f"  {set_name}: generating {to_generate} candidates "
          f"({already} already present)")

    t0 = time.time()
    seqs, stats = model.design_sequences(
        target,
        n=to_generate,
        seed=seed + already,
        batch_size=batch_size,
    )
    t_gen = time.time() - t0
    print(f"  {set_name}: {len(seqs)}/{to_generate} parsed as valid sequences "
          f"in {t_gen:.0f}s  ({stats.rejected_unparseable} unparseable, "
          f"{stats.rejected_invalid_aa} non-amino-acid)")

    t0 = time.time()
    fwd = (
        GenerationSettings.forward_notebook()
        if faithful_sampling
        else GenerationSettings.forward_default()
    )
    preds = model.predict_properties_batch(
        seqs, settings=fwd, seed=seed + 10_000, batch_size=fwd_batch_size
    )
    print(f"  {set_name}: forward pass over {len(seqs)} candidates in "
          f"{time.time() - t0:.0f}s "
          f"({sum(p is None for p in preds)} unparseable)")

    rows = []
    for i, (s, pred) in enumerate(zip(seqs, preds)):
        is_novel = novelty.is_novel_exact(s)
        if pred is None:
            stats.notes.append(f"forward parse failure at candidate {i}")
        rows.append(
            {
                "set": set_name,
                "candidate_index": already + i,
                "sequence": s,
                "length": len(s),
                "novel_exact": is_novel,
                "forward_parsed": pred is not None,
                **(
                    {f"pred_{p}": float(pred[j]) for j, p in enumerate(config.PROPERTY_NAMES)}
                    if pred is not None
                    else {f"pred_{p}": np.nan for p in config.PROPERTY_NAMES}
                ),
                "r2": metrics.r2_within_vector(target, pred) if pred is not None else np.nan,
                "mae": metrics.mae(target, pred) if pred is not None else np.nan,
                "rmse": metrics.rmse(target, pred) if pred is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    out = pd.concat([existing, df], ignore_index=True) if len(existing) else df
    pool_path(set_name).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pool_path(set_name), index=False)
    print(f"  {set_name}: wrote {len(out)} rows to {pool_path(set_name)}")
    return out


def summarise(set_name: str, target: list[float], pool: pd.DataFrame) -> dict:
    """Everything we can say about one property set."""
    scored = pool[pool["forward_parsed"] & pool["novel_exact"]]
    r2 = scored["r2"].to_numpy(dtype=float)
    r2 = r2[np.isfinite(r2)]

    paper_r2 = config.ALL_R2_PAPER[set_name]

    out: dict = {
        "set": set_name,
        "target": target,
        "target_spread": metrics.target_spread(target),
        "paper_r2": paper_r2,
        "n_generated": int(len(pool)),
        "n_novel": int(pool["novel_exact"].sum()),
        "n_forward_parsed": int(pool["forward_parsed"].sum()),
        "n_scored": int(len(r2)),
    }

    if len(r2) == 0:
        out["error"] = "no candidate produced a scoreable prediction"
        return out

    out.update(
        {
            "r2_max": float(r2.max()),
            "r2_mean": float(r2.mean()),
            "r2_median": float(np.median(r2)),
            "r2_std": float(r2.std(ddof=1)) if len(r2) > 1 else 0.0,
            "r2_p05": float(np.percentile(r2, 5)),
            "r2_p95": float(np.percentile(r2, 95)),
            "r2_min": float(r2.min()),
            "frac_above_paper_r2": float(np.mean(r2 >= paper_r2)),
            "n_required_to_reach_paper_r2": metrics.n_required_to_reach(r2, paper_r2),
            "mae_at_best": float(scored.loc[scored["r2"].idxmax(), "mae"]),
            "best_sequence": str(scored.loc[scored["r2"].idxmax(), "sequence"]),
            "best_length": int(scored.loc[scored["r2"].idxmax(), "length"]),
        }
    )

    n_values = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) if n <= len(r2)]
    if n_values:
        out["best_of_n_curve"] = metrics.best_of_n_curve(
            r2, n_values, rng=np.random.default_rng(config.SEED)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128,
                    help="candidates per property set (notebook default is 2048)")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="subset of F1 F2 F3 S1..S5; default is all eight")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="generation batch size; lower this if VRAM is tight")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--fwd-batch-size", type=int, default=16)
    ap.add_argument(
        "--faithful-sampling",
        action="store_true",
        help="use the released notebook's sampled forward decoding (T=0.01) "
             "instead of our greedy default; sensitivity check for A5",
    )
    args = ap.parse_args()

    names = args.sets or list(config.ALL_SETS)
    unknown = [n for n in names if n not in config.ALL_SETS]
    if unknown:
        raise SystemExit(f"unknown set names: {unknown}")

    print(f"Reproducing Table 1 for {names} with N={args.n} per set")
    print(f"(the released notebook's budget is "
          f"{config.PAPER_BUDGET_PER_STEP} x {config.PAPER_BUDGET_REPEATS} = "
          f"{config.PAPER_BUDGET_TOTAL})\n")

    model = SilkomeGPT()
    print(f"model on {model.device} as {model.dtype}\n")

    summaries = []
    for name in names:
        target = config.ALL_SETS[name]
        pool = run_one_set(
            model,
            name,
            target,
            args.n,
            args.batch_size,
            args.seed,
            not args.no_resume,
            faithful_sampling=args.faithful_sampling,
            fwd_batch_size=args.fwd_batch_size,
        )
        s = summarise(name, target, pool)
        summaries.append(s)
        if "error" in s:
            print(f"  {name}: {s['error']}\n")
            continue
        print(
            f"  {name}: paper {s['paper_r2']:.4f} | "
            f"ours max {s['r2_max']:.4f}  mean {s['r2_mean']:.4f}  "
            f"median {s['r2_median']:.4f}  (n={s['n_scored']})\n"
        )

    df = pd.DataFrame(
        [{k: v for k, v in s.items() if k not in ("best_of_n_curve", "target")} for s in summaries]
    )
    csv = config.RESULTS / "table1_reproduction.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)

    path = runinfo.write_result(
        "02_table1_reproduction",
        {"summaries": summaries},
        script="scripts/02_reproduce_table1.py",
        params={
            "n_per_set": args.n,
            "sets": names,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "generation": GenerationSettings.inverse_default().to_dict(),
            "forward": (
                GenerationSettings.forward_notebook()
                if args.faithful_sampling
                else GenerationSettings.forward_default()
            ).to_dict(),
            "faithful_sampling": args.faithful_sampling,
        },
    )
    print(f"wrote {csv}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
