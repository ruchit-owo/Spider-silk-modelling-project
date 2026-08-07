"""Extension 1: what does the forward task actually read from the sequence?

SilkomeGPT's forward task maps a sequence to eight numbers. This script asks
which properties of the sequence that mapping depends on, by editing sequences
in controlled ways and measuring how far the prediction moves.

The ablations, and what each isolates:

    repeat            same sequence, resampled. Not an ablation - this is the
                      noise floor. The forward task decodes with
                      do_sample=True, so it is stochastic even at temperature
                      0.01. Every other effect below is reported as a multiple
                      of this floor. An effect below the floor is not evidence.

    shuffle           composition preserved exactly, all order destroyed.
    shuffle_blocks    composition and local motifs preserved, arrangement
                      destroyed.
    reverse           composition preserved, N-to-C direction destroyed.
    knockout_polyA    poly-alanine tracts excised (beta-sheet formers).
    knockout_GGX      GGX repeats excised.
    knockout_GPGXX    GPGXX repeats excised (MaSp2 elastic motif).
    length_control    same number of residues removed at random positions.
                      The control for the three knockouts.
    termini_only      repetitive core removed.
    core_only         terminal domains removed.
    random_matched    uniform random residues, matched length. The floor at
                      the other end: what the model does with no silk signal.

Interpretation limits, stated here because they bound every conclusion drawn
from this script:

  - A shuffled spidroin is far outside the fine-tuning distribution. A large
    prediction shift shows the model is sensitive to order, but does not show
    it uses order in any biologically meaningful way; it may simply be
    reacting to text that does not look like its training data.
  - The knockouts change length as well as content, which is why
    length_control is run on the same sequences with the same number of
    residues removed. Where a knockout and its length control move the
    prediction equally, the knockout has shown nothing about the motif.
  - This measures the model, not spider silk. Nothing here is a claim about
    real mechanical properties.

Usage:
    python scripts/11_ablation_sequence_content.py --n-sequences 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import ablations, config, dataio, novelty, runinfo  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402


def pick_probe_sequences(n: int, seed: int, max_len: int = 700) -> tuple[list[str], str]:
    """Sequences to ablate.

    Prefer the curated MaSp pairs, since those are what the model was
    fine-tuned on and the forward task is defined for. Fall back to the silk
    sequences shipped with the authors' repo, recording which was used.
    """
    rng = np.random.default_rng(seed)
    try:
        df = dataio.load_pairs()
        seqs = df["sequence"].astype(str).tolist()
        source = "silkome_masp_pairs"
    except dataio.MissingInput:
        seqs = list(novelty.load_known_sequences())
        source = "ALL_SILK_SEQ (MaSp pairs unavailable)"

    # The model was fine-tuned for sequences up to 768 residues; longer inputs
    # would confound the ablation with truncation effects.
    seqs = [s for s in seqs if 200 <= len(s) <= max_len and set(s) <= set(ablations.AA_ORDER)]
    if not seqs:
        raise SystemExit("no probe sequences in the usable length range")
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    return [seqs[i] for i in idx], source


def build_edits(seq: str, rng: np.random.Generator) -> list[tuple[str, str, int]]:
    """(ablation name, edited sequence, residues changed) for one sequence."""
    edits: list[tuple[str, str, int]] = []

    edits.append(("shuffle", ablations.shuffle_residues(seq, rng), len(seq)))
    edits.append(("shuffle_blocks", ablations.shuffle_blocks(seq, rng, block=20), len(seq)))
    edits.append(("reverse", ablations.reverse_sequence(seq), len(seq)))

    for motif in ("polyA", "GGX", "GPGXX"):
        pat = ablations.CANONICAL_MOTIFS[motif]
        edited, removed = ablations.knockout_motif_delete(seq, pat)
        if removed == 0:
            continue  # motif absent; nothing to knock out, and no control needed
        edits.append((f"knockout_{motif}", edited, removed))
        edits.append(
            (
                f"length_control_{motif}",
                ablations.length_matched_deletion(seq, removed, rng),
                removed,
            )
        )
        sub, n_sub = ablations.knockout_motif_substitute(seq, pat, rng)
        edits.append((f"substitute_{motif}", sub, n_sub))

    t = ablations.keep_termini(seq)
    c = ablations.keep_core(seq)
    if t and t != seq:
        edits.append(("termini_only", t, len(seq) - len(t)))
    if c:
        edits.append(("core_only", c, len(seq) - len(c)))

    edits.append(("random_matched", ablations.random_sequence_matched(seq, rng), len(seq)))
    return edits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=30)
    ap.add_argument("--noise-repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    probes, probe_source = pick_probe_sequences(args.n_sequences, args.seed)
    print(f"{len(probes)} probe sequences from {probe_source}")
    print(f"lengths {min(len(s) for s in probes)}-{max(len(s) for s in probes)}\n")

    model = SilkomeGPT()
    fwd = GenerationSettings.forward_default()

    def predict(seq: str, seed: int):
        return model.predict_properties(seq, fwd, seed=seed)

    # --- noise floor -----------------------------------------------------
    print(f"measuring noise floor ({args.noise_repeats} repeats per sequence)")
    floor = ablations.repeat_noise_floor(
        predict, probes[: min(10, len(probes))], n_repeats=args.noise_repeats, seed=args.seed
    )
    if "error" in floor:
        raise SystemExit(f"could not measure noise floor: {floor['error']}")
    print(f"  pooled mean |deviation|  {floor['pooled_mean_abs_dev']:.5f}")
    print(f"  pooled p95  |deviation|  {floor['pooled_p95_abs_dev']:.5f}")
    print(f"  distinct outputs per sequence: "
          f"{[p['n_distinct_outputs'] for p in floor['per_sequence']]}\n")

    # --- ablations -------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    rows = []
    n_pred_fail = 0

    for si, seq in enumerate(probes):
        base = predict(seq, args.seed + si)
        if base is None:
            n_pred_fail += 1
            continue

        for name, edited, changed in build_edits(seq, rng):
            if not edited:
                continue
            pred = predict(edited, args.seed + 100_000 + si)
            if pred is None:
                n_pred_fail += 1
                continue
            res = ablations.AblationResult(
                name=name,
                original_sequence=seq,
                edited_sequence=edited,
                original_pred=base,
                edited_pred=pred,
                n_residues_changed=changed,
            )
            row = res.to_row(config.PROPERTY_NAMES)
            row["sequence_index"] = si
            rows.append(row)

        print(f"  sequence {si + 1}/{len(probes)} done")

    df = pd.DataFrame(rows)
    out_csv = config.RESULTS / "ablation_sequence_content.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # --- summary ---------------------------------------------------------
    floor_mean = floor["pooled_mean_abs_dev"]
    summary_rows = []
    for name, grp in df.groupby("ablation"):
        d = grp["abs_delta_mean"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "ablation": name,
                "n": len(d),
                "mean_abs_delta": float(d.mean()),
                "median_abs_delta": float(np.median(d)),
                "sd_abs_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "vs_noise_floor": float(d.mean() / floor_mean) if floor_mean > 0 else np.inf,
                "frac_below_noise_floor": float(np.mean(d < floor_mean)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_abs_delta", ascending=False)

    # Paired comparison of each knockout against its own length control.
    paired = []
    for motif in ("polyA", "GGX", "GPGXX"):
        a = df[df["ablation"] == f"knockout_{motif}"].set_index("sequence_index")["abs_delta_mean"]
        b = df[df["ablation"] == f"length_control_{motif}"].set_index("sequence_index")["abs_delta_mean"]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            continue
        diff = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
        entry = {
            "motif": motif,
            "n_paired": int(len(common)),
            "mean_knockout": float(a.loc[common].mean()),
            "mean_length_control": float(b.loc[common].mean()),
            "mean_difference": float(diff.mean()),
        }
        try:
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(diff)
            entry["wilcoxon_p"] = float(p)
        except Exception:
            # scipy is optional; report a sign test instead of nothing.
            entry["frac_knockout_larger"] = float(np.mean(diff > 0))
        paired.append(entry)

    print("\nablation effect, mean |change| in the 8D prediction")
    print(f"(noise floor = {floor_mean:.5f})\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if paired:
        print("\nmotif knockout vs its length-matched control:")
        for e in paired:
            print(f"  {e['motif']:8s} knockout {e['mean_knockout']:.5f} "
                  f"vs control {e['mean_length_control']:.5f} "
                  f"(difference {e['mean_difference']:+.5f})")

    summary.to_csv(config.RESULTS / "ablation_sequence_content_summary.csv", index=False)
    path = runinfo.write_result(
        "11_ablation_sequence_content",
        {
            "probe_source": probe_source,
            "n_probes": len(probes),
            "n_prediction_failures": n_pred_fail,
            "noise_floor": floor,
            "summary": summary_rows,
            "knockout_vs_length_control": paired,
        },
        script="scripts/11_ablation_sequence_content.py",
        params={
            "n_sequences": args.n_sequences,
            "noise_repeats": args.noise_repeats,
            "seed": args.seed,
            "forward": fwd.to_dict(),
        },
    )
    print(f"\nwrote {out_csv}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
