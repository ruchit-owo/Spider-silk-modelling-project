"""Provenance for every result file.

Every script in scripts/ writes a manifest next to its output. The rule is that
any number appearing in the writeup must be traceable to a manifest that says
which commit, which model revision, which seed and which package versions
produced it. Without that, a result that changes between runs is impossible to
diagnose.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict:
    v: dict[str, str | None] = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "numpy", "pandas", "sklearn", "Bio", "matplotlib"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            v[mod] = None
    return v


def _gpu() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            ),
            "torch_cuda": torch.version.cuda,
        }
    except Exception:
        return {"cuda": None}


def manifest(script: str, params: dict | None = None) -> dict:
    return {
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "versions": _versions(),
        "gpu": _gpu(),
        "params": params or {},
    }


def write_result(
    name: str,
    payload: dict,
    script: str,
    params: dict | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Write payload + manifest as one JSON file under results/."""
    from . import config

    results_dir = results_dir or config.RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    doc = {"manifest": manifest(script, params), "result": payload}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_result(name: str, results_dir: Path | None = None) -> dict:
    from . import config

    results_dir = results_dir or config.RESULTS
    with open(results_dir / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)
