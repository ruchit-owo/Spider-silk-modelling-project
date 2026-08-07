"""Tests for the sequence descriptors used in Figure 5.

The pointed test here is test_legacy_helix_equals_current_biopython_sheet. It
pins down the discrepancy documented in docs/discrepancies.md D1: the group
the paper's caption calls "helix" is the group current Biopython calls "sheet".
If that ever stops being true, the discrepancy write-up is stale and this test
says so.
"""

import pytest

from silkrepro import protparam

SEQ = "GAGAGSGAAGYGGLGSQGAGRGGQGAAAAAAGGAGQGGYGGLGSQGAGQEMLVIYFWNPGS"


def test_legacy_groups_match_the_figure5_caption():
    """helix (V I Y F W L), turn (N P G S), sheet (E M A L)."""
    assert protparam.LEGACY_HELIX_AA == set("VIYFWL")
    assert protparam.LEGACY_TURN_AA == set("NPGS")
    assert protparam.LEGACY_SHEET_AA == set("EMAL")


def test_legacy_helix_equals_current_biopython_sheet():
    """The core of discrepancy D1.

    Current Biopython computes its sheet fraction from V I Y F W L T. Dropping
    the T it added in v1.82 leaves exactly the paper's "helix" group. So the
    paper's helix bars are sheet propensities under the corrected assignment.
    """
    assert protparam.LEGACY_HELIX_AA == protparam.CORRECTED_SHEET_AA - {"T"}
    assert protparam.LEGACY_SHEET_AA == protparam.CORRECTED_HELIX_AA - {"K"}


def test_legacy_fractions_agree_with_pre_1_82_biopython_arithmetic():
    """Independent check of our reimplementation.

    Pre-1.82 Biopython computed each fraction as the summed composition of its
    group. We count residues directly. The two must agree.
    """
    frac = protparam.amino_acid_fractions(SEQ)
    legacy = protparam.secondary_structure_legacy(SEQ)
    assert legacy["frac_helix_legacy"] == pytest.approx(
        sum(frac[r] for r in "VIYFWL"), abs=1e-12
    )
    assert legacy["frac_turn_legacy"] == pytest.approx(
        sum(frac[r] for r in "NPGS"), abs=1e-12
    )
    assert legacy["frac_sheet_legacy"] == pytest.approx(
        sum(frac[r] for r in "EMAL"), abs=1e-12
    )


def test_corrected_fractions_agree_with_installed_biopython():
    """Guards against Biopython changing its groupings again underneath us."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    pa = ProteinAnalysis(SEQ)
    try:
        helix, turn, sheet = pa.secondary_structure_fraction()
    except Exception:  # pragma: no cover - method removed upstream
        pytest.skip("Biopython no longer exposes secondary_structure_fraction")

    c = protparam.secondary_structure_corrected(SEQ)
    if helix == pytest.approx(c["frac_helix_corrected"], abs=1e-9):
        assert turn == pytest.approx(c["frac_turn_corrected"], abs=1e-9)
        assert sheet == pytest.approx(c["frac_sheet_corrected"], abs=1e-9)
    else:
        # Installed Biopython predates 1.82; it should then match the legacy
        # convention instead. Either way we must match one of the two.
        legacy = protparam.secondary_structure_legacy(SEQ)
        assert helix == pytest.approx(legacy["frac_helix_legacy"], abs=1e-9), (
            "installed Biopython matches neither the legacy nor the corrected "
            "grouping; docs/discrepancies.md D1 needs revisiting"
        )


def test_structure_fractions_do_not_sum_to_one():
    """The groups overlap (L is in both helix and sheet under the legacy
    definition), so these are propensity fractions, not a partition. Stated in
    the writeup; asserted here so it cannot change silently."""
    m = protparam.secondary_structure_legacy(SEQ)
    total = sum(m.values())
    assert total != pytest.approx(1.0)


def test_composition_percentages_sum_to_100():
    d = protparam.descriptors(SEQ)
    total = sum(d[f"pct_{a}"] for a in protparam.AA_ORDER)
    assert total == pytest.approx(100.0, abs=1e-6)


def test_amino_acid_fractions_sum_to_one():
    f = protparam.amino_acid_fractions(SEQ)
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-12)


def test_molecular_weight_is_plausible():
    """Average residue mass is roughly 110 Da; poly-glycine sits well below.
    A magnitude check to catch unit errors."""
    d = protparam.descriptors("G" * 100)
    assert 5000 < d["molecular_weight"] < 6500


def test_nonstandard_residues_raise_rather_than_being_substituted():
    with pytest.raises(ValueError, match="non-standard"):
        protparam.descriptors("GAGAXAGA")


def test_descriptor_table_carries_both_conventions():
    df = protparam.descriptor_table({"a": SEQ, "b": "GAGAGA"})
    assert len(df) == 2
    for col in (
        "instability_index",
        "pct_A",
        "frac_helix_legacy",
        "frac_sheet_legacy",
        "frac_helix_corrected",
        "frac_sheet_corrected",
    ):
        assert col in df.columns, col
