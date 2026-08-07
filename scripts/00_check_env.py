"""Record the environment and verify the checkpoint matches the paper's spec.

Run this first. It downloads the released checkpoint (about 1 GB), reports the
resolved revision so later runs can be pinned, and checks the loaded
architecture against the numbers stated in the paper's Experimental Section.

A mismatch here would mean the published checkpoint is not the model the paper
describes, which would need to be reported rather than worked around. Nothing
in this script corrects anything; it only compares and prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, runinfo  # noqa: E402
from silkrepro.model import SilkomeGPT  # noqa: E402


def main() -> int:
    print("Loading", config.MODEL_NAME, "(first run downloads ~1 GB)")
    m = SilkomeGPT()

    arch = m.arch_summary()
    n_params = m.n_parameters()
    n_params_m = n_params / 1e6

    print(f"\ndevice          : {m.device}")
    print(f"dtype           : {m.dtype}")
    print(f"parameters      : {n_params:,} ({n_params_m:.1f} M)")
    print(f"paper states    : {config.EXPECTED_PARAMS_M} M")
    print("\narchitecture:")
    for k, v in arch.items():
        print(f"  {k:26s} {v}")

    mismatches = []
    for k, expected in config.EXPECTED_ARCH.items():
        got = arch.get(k)
        if got != expected:
            mismatches.append((k, expected, got))

    # 0.5 M tolerance on the parameter count: the paper rounds to one decimal
    # and it is not stated whether the count includes tied embeddings.
    param_ok = abs(n_params_m - config.EXPECTED_PARAMS_M) < 0.5

    print("\narchitecture vs paper:")
    if not mismatches:
        print("  all four stated hyperparameters match")
    else:
        for k, e, g in mismatches:
            print(f"  MISMATCH {k}: paper says {e}, checkpoint has {g}")
    print(f"  parameter count {'matches' if param_ok else 'DIFFERS'} "
          f"({n_params_m:.2f} M vs {config.EXPECTED_PARAMS_M} M)")

    # Resolve the revision actually downloaded, so the run is reproducible.
    revision = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(config.MODEL_NAME).sha
        print(f"\nresolved revision: {revision}")
        (config.RESULTS).mkdir(parents=True, exist_ok=True)
        (config.RESULTS / "model_revision.txt").write_text(revision, encoding="utf-8")
        print("written to results/model_revision.txt "
              "(export SILKOME_REVISION=<sha> to pin later runs)")
    except Exception as exc:  # network or auth issue; not fatal
        print(f"\ncould not resolve revision: {exc}")

    path = runinfo.write_result(
        "00_env_and_checkpoint",
        {
            "n_parameters": n_params,
            "n_parameters_millions": n_params_m,
            "paper_parameters_millions": config.EXPECTED_PARAMS_M,
            "parameter_count_matches": param_ok,
            "architecture": arch,
            "architecture_mismatches": [
                {"field": k, "paper": e, "checkpoint": g} for k, e, g in mismatches
            ],
            "resolved_revision": revision,
        },
        script="scripts/00_check_env.py",
        params={"model_name": config.MODEL_NAME},
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
