"""End-to-end smoke test of the three tasks, before any measurement.

Checks that the prompt formats in tasks.py actually elicit the expected output
shape from the released checkpoint. If the forward task does not return eight
bracketed numbers, everything downstream is measuring a parse failure rather
than the model, so this runs first and prints raw output.

Also times a single generation, which sets expectations for how long a full
2,048-candidate run will take.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, novelty, runinfo, tasks  # noqa: E402
from silkrepro.model import GenerationSettings, SilkomeGPT  # noqa: E402

# A real MaSp-like sequence from the paper's own Experimental Section example
# for the "CalculateSilkContent" task.
PAPER_EXAMPLE = (
    "AAAGGAGQGGYGGQGAGQGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGSGQGG"
    "YGGQGAGAAAAAAAAGGAGQGGYGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQGAGQGSG"
    "AAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGGQGAGQGAAAAAAGGAGQGGYGGQ"
    "GAGQGAGAAAAAAGGAGQGGYGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQGAGAAAAAA"
    "GGAGQGGYGGQGAGQGAAAAAAGGAGQGGYGGQGAGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQ"
    "GAGQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGGQGAGAAAAAAGGAGQG"
    "GYGGQGAGQGGYGGQGSGAAAAAAAAGGAGQGGYGGLGSQGAGQGAGAAAAAAGGAGQGGYGGQGA"
    "GQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGG"
    "YGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGQGAGAAAASAAASRLSSPEASSGLSGCDVLVQA"
    "LLEVVSALIHILGSSSIGPVNYGSASQSTQIVGQSVYQALG"
)


def main() -> int:
    model = SilkomeGPT()
    print(f"model on {model.device} as {model.dtype}\n")
    report: dict = {}

    # --- forward task ----------------------------------------------------
    print("=" * 70)
    print("forward task: CalculateSilkContent")
    print("=" * 70)
    prompt = tasks.forward_prompt(PAPER_EXAMPLE)
    print(f"prompt is {len(prompt)} characters, "
          f"{len(model.tokenizer.encode(prompt))} tokens")

    t0 = time.time()
    raw = model.generate(prompt, GenerationSettings.forward_default(), 1, seed=0)[0]
    dt = time.time() - t0
    tail = raw[len(prompt):]
    print(f"raw continuation: {tail[:200]!r}")
    pred = tasks.parse_property_output(tail)
    print(f"parsed: {pred}")
    print(f"time: {dt:.2f} s")
    report["forward"] = {
        "parsed": pred is not None,
        "prediction": None if pred is None else pred.tolist(),
        "seconds": dt,
        "raw_tail": tail[:300],
    }

    if pred is not None:
        print("\npredicted properties:")
        for name, v in zip(config.PROPERTY_NAMES, pred):
            print(f"  {name:14s} {v:.3f}")

    # --- repeatability ---------------------------------------------------
    print("\n" + "=" * 70)
    print("forward task repeatability (same input, 5 draws)")
    print("=" * 70)
    print("The forward task decodes with do_sample=True, so it need not be")
    print("deterministic. This is the noise floor every ablation is measured")
    print("against.\n")
    draws = []
    for i in range(5):
        p = model.predict_properties(PAPER_EXAMPLE, seed=100 + i)
        draws.append(None if p is None else p.tolist())
        print(f"  draw {i}: {p}")
    distinct = len({tuple(d) for d in draws if d is not None})
    print(f"\n{distinct} distinct outputs across 5 draws")
    report["repeatability"] = {"draws": draws, "n_distinct": distinct}

    # --- inverse task ----------------------------------------------------
    print("\n" + "=" * 70)
    print("inverse task: GenerateSilkContent")
    print("=" * 70)
    target = config.SELFCONSISTENCY_SETS["S1"]
    print(f"target (set S1): {target}")
    print(f"prompt: {tasks.inverse_prompt(target)}\n")

    t0 = time.time()
    seqs, stats = model.design_sequences(target, n=4, seed=0, batch_size=4)
    dt = time.time() - t0
    print(f"{len(seqs)}/4 parsed as valid protein in {dt:.1f} s "
          f"({dt / 4:.1f} s per candidate)")
    print(f"parse stats: {stats.to_dict()}")

    for i, s in enumerate(seqs):
        print(f"\n  candidate {i}: length {len(s)}, "
              f"novel={novelty.is_novel_exact(s)}")
        print(f"    {s[:120]}{'...' if len(s) > 120 else ''}")
        p = model.predict_properties(s, seed=200 + i)
        if p is not None:
            from silkrepro import metrics

            print(f"    predicted: {[round(float(x), 3) for x in p]}")
            print(f"    R2 vs target: {metrics.r2_within_vector(target, p):.4f}")

    report["inverse"] = {
        "n_requested": 4,
        "n_valid": len(seqs),
        "parse_stats": stats.to_dict(),
        "seconds": dt,
        "seconds_per_candidate": dt / 4,
        "lengths": [len(s) for s in seqs],
    }

    # --- projected cost --------------------------------------------------
    per_candidate = dt / 4 + report["forward"]["seconds"]
    total_hours = per_candidate * config.PAPER_BUDGET_TOTAL * 8 / 3600
    print("\n" + "=" * 70)
    print(f"projected cost of the full paper budget "
          f"({config.PAPER_BUDGET_TOTAL} candidates x 8 property sets):")
    print(f"  {per_candidate:.1f} s per candidate -> {total_hours:.1f} hours")
    print("=" * 70)
    report["projected_full_run_hours"] = total_hours

    path = runinfo.write_result(
        "01_smoke_test", report, script="scripts/01_smoke_test.py"
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
