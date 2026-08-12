"""Keep the prose honest about the code.

Every check here corresponds to something that was actually wrong in this
repository at some point: counts quoted in the README that had drifted from
reality, a figure whose colour rule tested for an ablation name the code never
emits, and cross-references to scripts that do not exist.

None of these are deep, and all of them are the kind of thing a reader
reasonably takes at face value.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Counts quoted in prose
# --------------------------------------------------------------------------


def count_assumptions() -> int:
    return len(re.findall(r"^## A\d+", read("ASSUMPTIONS.md"), re.M))


def count_discrepancies() -> int:
    return len(re.findall(r"^## D\d+", read("docs/discrepancies.md"), re.M))


def test_assumption_ids_are_contiguous():
    ids = sorted(
        int(m) for m in re.findall(r"^## A(\d+)", read("ASSUMPTIONS.md"), re.M)
    )
    assert ids == list(range(1, len(ids) + 1)), f"gaps or duplicates in {ids}"


def test_discrepancy_ids_are_contiguous():
    ids = sorted(
        int(m) for m in re.findall(r"^## D(\d+)", read("docs/discrepancies.md"), re.M)
    )
    assert ids == list(range(1, len(ids) + 1)), f"gaps or duplicates in {ids}"


def test_readme_assumption_count_matches():
    readme = read("README.md")
    m = re.search(r"all (\d+) choices the (?:source |)paper", readme)
    assert m, "README no longer states an assumption count in the expected form"
    assert int(m.group(1)) == count_assumptions()


def test_readme_discrepancy_count_matches():
    readme = read("README.md")
    m = re.search(r"(\d+) discrepancies and open points", readme)
    assert m, "README no longer states a discrepancy count in the expected form"
    assert int(m.group(1)) == count_discrepancies()


def test_readme_property_set_count_matches_config():
    from silkrepro import config

    readme = read("README.md")
    assert str(len(config.ALL_SETS)) in readme
    assert len(config.ALL_SETS) == 8


# --------------------------------------------------------------------------
# Cross-references
# --------------------------------------------------------------------------


SCRIPT_RE = re.compile(r"scripts/(\d[\w]*\.py)")


def test_referenced_scripts_exist():
    """Docstrings and docs must not point at scripts that were renamed away."""
    missing = []
    for path in list((ROOT / "src" / "silkrepro").glob("*.py")) + \
                list((ROOT / "scripts").glob("*.py")) + \
                [ROOT / "README.md", ROOT / "RESULTS.md", ROOT / "ASSUMPTIONS.md",
                 ROOT / "docs" / "methods.md", ROOT / "docs" / "discrepancies.md"]:
        text = path.read_text(encoding="utf-8")
        for name in SCRIPT_RE.findall(text):
            if not (ROOT / "scripts" / name).exists():
                missing.append(f"{path.name} -> scripts/{name}")
    assert not missing, "references to non-existent scripts: " + "; ".join(missing)


# --------------------------------------------------------------------------
# Claims about behaviour that the config controls
# --------------------------------------------------------------------------


def test_no_stale_claims_that_the_forward_task_samples():
    """config.FWD_DO_SAMPLE is False (assumption A5). Module docstrings that
    still describe the forward task as sampling would mislead a reader about
    why the noise floor behaves as it does."""
    from silkrepro import config

    assert config.FWD_DO_SAMPLE is False

    offenders = []
    for rel in ("src/silkrepro/ablations.py",
                "scripts/11_ablation_sequence_content.py",
                "docs/methods.md",
                "README.md"):
        text = read(rel)
        if re.search(r"forward task decodes with do_sample=True", text):
            offenders.append(rel)
    assert not offenders, f"stale sampling claim in {offenders}"


def test_figure4_control_rule_matches_emitted_ablation_names():
    """The figure greys the control bars. It must test for names the ablation
    script actually produces - it once tested for 'length_control', which the
    script never emits, so no bar was ever greyed while the title said they
    were."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "figs", ROOT / "scripts" / "99_make_figures.py"
    )
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)

    ablation_src = read("scripts/11_ablation_sequence_content.py")
    emitted = set(re.findall(r'\(\s*f?"([a-z_]+(?:_\{motif\})?)"', ablation_src))
    emitted |= set(re.findall(r'f"(\w+)_\{motif\}"', ablation_src))

    controls = {n for n in emitted if "control" in n}
    assert controls, "no control ablations found in scripts/11"
    for name in controls:
        assert figs.is_control(name + "_polyA" if name.endswith("control") else name), (
            f"figure colour rule does not recognise control ablation {name!r}"
        )

    for name in ("shuffle", "reverse", "knockout_polyA", "point_mutation"):
        assert not figs.is_control(name), f"{name} wrongly treated as a control"


def test_figure_module_guards_the_noise_floor():
    """The figure must not draw a reference line at a floor that scripts/11
    itself refuses to divide by."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "figs2", ROOT / "scripts" / "99_make_figures.py"
    )
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    assert hasattr(figs, "PREDICTION_RESOLUTION")
    assert figs.PREDICTION_RESOLUTION == pytest.approx(1e-4)
