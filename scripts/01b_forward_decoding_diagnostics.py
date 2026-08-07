"""Diagnostics for the forward task's decoding settings.

Three questions, each of which changes numbers downstream, and none of which
the paper answers:

  Q1. Does the forward task always emit the property vector immediately?
      No. For some inputs the model first continues the amino-acid sequence,
      then closes it with '>', and only then emits the bracketed values. With
      the released notebook's max_new_tokens=64 the answer can be cut off
      before it appears, and the notebook counts that as nothing at all (it is
      caught by a bare `except:`). This script measures the parse-failure rate
      as a function of the token budget.

  Q2. Is sampling at temperature 0.01 equivalent to greedy decoding?
      Nearly, but the residual stochasticity lands exactly where it does most
      damage: in how many extra residues the model emits before the values.
      This script measures how often sampled and greedy decoding disagree.

  Q3. Does batch composition change results?
      With do_sample=True it can, because all rows of a batch draw from one
      RNG stream, so a sequence's result depends on what it was batched with.
      With greedy decoding it cannot.

The outcome decides the project's default, recorded as assumption A5.

Usage:
    python scripts/01b_forward_decoding_diagnostics.py --n 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, novelty, runinfo, tasks  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402
from silkrepro.paper_examples import (  # noqa: E402
    PAPER_EXAMPLE_PROPERTIES,
    PAPER_EXAMPLE_SEQUENCE,
)


def probe_sequences(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    known = [
        s
        for s in novelty.load_known_sequences()
        if 80 <= len(s) <= 700 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
    ]
    idx = rng.choice(len(known), size=min(n, len(known)), replace=False)
    return [known[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    model = SilkomeGPT()
    seqs = probe_sequences(args.n, args.seed)
    print(f"{len(seqs)} probe sequences, lengths "
          f"{min(len(s) for s in seqs)}-{max(len(s) for s in seqs)}\n")

    report: dict = {}

    # --- Q1: token budget -----------------------------------------------
    print("Q1  parse rate vs max_new_tokens (greedy decoding)")
    print("    budget   parsed   rate     extra residues before answer")
    budget_rows = []
    for budget in (32, 64, 96, 128, 192):
        s = GenerationSettings(
            temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=budget, do_sample=False
        )
        parsed = 0
        extras = []
        for seq in seqs:
            prompt = tasks.forward_prompt(seq)
            text = model.generate(prompt, s, 1, seed=args.seed)[0]
            tail = text[len(prompt):]
            v = tasks.parse_property_output(tail)
            if v is not None:
                parsed += 1
                # Residues emitted before the bracketed answer, if any.
                head = tail[: tail.find("[")]
                extras.append(sum(c.isalpha() for c in head))
        rate = parsed / len(seqs)
        med_extra = float(np.median(extras)) if extras else float("nan")
        max_extra = int(np.max(extras)) if extras else 0
        print(f"    {budget:6d}   {parsed:6d}   {rate:.3f}    "
              f"median {med_extra:.0f}, max {max_extra}")
        budget_rows.append(
            {
                "max_new_tokens": budget,
                "n_parsed": parsed,
                "parse_rate": rate,
                "median_extra_residues": med_extra,
                "max_extra_residues": max_extra,
            }
        )
    report["budget_sweep"] = budget_rows

    # --- Q2: sampled vs greedy ------------------------------------------
    print("\nQ2  sampled (T=0.01, the notebook's setting) vs greedy, budget 192")
    sampled_settings = GenerationSettings(
        temperature=config.FWD_TEMPERATURE,
        top_k=config.FWD_TOP_K,
        top_p=config.FWD_TOP_P,
        max_new_tokens=192,
        do_sample=True,
    )
    greedy_settings = GenerationSettings(
        temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=192, do_sample=False
    )

    agree = disagree = only_one_parsed = neither = 0
    max_abs_diff = 0.0
    for i, seq in enumerate(seqs):
        prompt = tasks.forward_prompt(seq)
        a = tasks.parse_property_output(
            model.generate(prompt, sampled_settings, 1, seed=args.seed + i)[0][len(prompt):]
        )
        b = tasks.parse_property_output(
            model.generate(prompt, greedy_settings, 1, seed=args.seed + i)[0][len(prompt):]
        )
        if a is None and b is None:
            neither += 1
        elif a is None or b is None:
            only_one_parsed += 1
        elif np.allclose(a, b, atol=1e-9):
            agree += 1
        else:
            disagree += 1
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))))

    n = len(seqs)
    print(f"    identical           {agree}/{n} ({agree / n:.1%})")
    print(f"    differing values    {disagree}/{n}"
          + (f"  (max |difference| {max_abs_diff:.3f})" if disagree else ""))
    print(f"    only one parsed     {only_one_parsed}/{n}")
    print(f"    neither parsed      {neither}/{n}")
    report["sampled_vs_greedy"] = {
        "n": n,
        "identical": agree,
        "differing": disagree,
        "only_one_parsed": only_one_parsed,
        "neither_parsed": neither,
        "max_abs_difference": max_abs_diff,
    }

    # --- Q3: batch invariance under greedy ------------------------------
    print("\nQ3  batch-size invariance under greedy decoding")
    subset = seqs[:16]
    g = GenerationSettings(
        temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=192, do_sample=False
    )
    b1 = model.predict_properties_batch(subset, settings=g, batch_size=1)
    b8 = model.predict_properties_batch(subset, settings=g, batch_size=8)
    mismatches = 0
    for a, b in zip(b1, b8):
        if (a is None) != (b is None):
            mismatches += 1
        elif a is not None and not np.allclose(a, b, atol=1e-6):
            mismatches += 1
    print(f"    {len(subset) - mismatches}/{len(subset)} identical "
          f"between batch_size 1 and 8")
    report["batch_invariance_greedy"] = {
        "n": len(subset),
        "mismatches": mismatches,
    }

    # --- the paper's worked example -------------------------------------
    print("\npaper's worked example (Experimental Section)")
    pred = tasks.parse_property_output(
        model.generate(
            tasks.forward_prompt(PAPER_EXAMPLE_SEQUENCE), g, 1, seed=0
        )[0][len(tasks.forward_prompt(PAPER_EXAMPLE_SEQUENCE)):]
    )
    ok = pred is not None and np.allclose(pred, PAPER_EXAMPLE_PROPERTIES, atol=1e-3)
    print(f"    paper : {PAPER_EXAMPLE_PROPERTIES}")
    print(f"    model : {None if pred is None else [round(float(x), 3) for x in pred]}")
    print(f"    match : {ok}")
    report["paper_worked_example"] = {
        "expected": PAPER_EXAMPLE_PROPERTIES,
        "predicted": None if pred is None else pred.tolist(),
        "match": bool(ok),
    }

    path = runinfo.write_result(
        "01b_forward_decoding_diagnostics",
        report,
        script="scripts/01b_forward_decoding_diagnostics.py",
        params={"n": args.n, "seed": args.seed},
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
