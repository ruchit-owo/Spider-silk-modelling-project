"""Checks that the transcribed constants are internally consistent.

The eight property sets in config.py were typed in from Table 1 of the paper.
Typos there would propagate into every result silently, so the paper's own
prose descriptions of those sets are used as an independent check: Section 2
and the Figure 2 caption both describe each set in words, and those words
imply particular entries.

One of these tests documents an inconsistency in the paper rather than in our
transcription. See test_set3_strength_departs_from_the_stated_median.
"""

import pytest

from silkrepro import config


def test_all_sets_are_eight_dimensional():
    for name, values in config.ALL_SETS.items():
        assert len(values) == 8, name


def test_all_values_in_unit_interval():
    """The Experimental Section states all eight values are normalised to
    0...1."""
    for name, values in config.ALL_SETS.items():
        assert all(0.0 <= v <= 1.0 for v in values), name


def test_property_name_ordering_matches_equation_1():
    assert config.PROPERTY_NAMES == [
        "toughness",
        "toughness_sd",
        "E",
        "E_sd",
        "strength",
        "strength_sd",
        "strain",
        "strain_sd",
    ]
    assert [config.PROPERTY_NAMES[i] for i in config.MECHANICAL_IDX] == [
        "toughness",
        "E",
        "strength",
        "strain",
    ]


@pytest.mark.parametrize(
    "name,toughness,E",
    [
        ("S1", 0.25, 0.20),   # "common ... pair (0.2 and 0.25)"
        ("S2", 0.55, 0.75),   # "relatively high E (0.75 and 0.55)"
        ("S3", 1.00, 0.45),   # "high toughness (0.45 and 1.00)"
        ("S4", 0.90, 0.80),   # "elevated ... (0.8 and 0.9)"
    ],
)
def test_selfconsistency_sets_match_prose_description(name, toughness, E):
    v = config.SELFCONSISTENCY_SETS[name]
    assert v[0] == pytest.approx(toughness), f"{name} toughness"
    assert v[2] == pytest.approx(E), f"{name} E"


def test_set5_matches_its_strength_toughness_description():
    """"Set 5 instead centered around common strength and toughness value
    pairs (0.27 and 0.24)"."""
    v = config.SELFCONSISTENCY_SETS["S5"]
    assert v[4] == pytest.approx(0.27)  # strength
    assert v[0] == pytest.approx(0.24)  # toughness


def test_medians_are_shared_across_selfconsistency_sets():
    """The paper says "the medians are employed" for the property values not
    being varied. The four SD slots are never varied, so they must be
    identical across all five sets - and they are."""
    sd_slots = config.SD_IDX
    ref = config.SELFCONSISTENCY_SETS["S1"]
    for name, v in config.SELFCONSISTENCY_SETS.items():
        for i in sd_slots:
            assert v[i] == pytest.approx(ref[i]), f"{name} slot {i}"


def test_set3_strength_departs_from_the_stated_median():
    """Documents an inconsistency in Table 1, not in our transcription.

    The paper says that for each self-consistency set, values other than the
    two being varied use the dataset medians. Sets S1, S2, S4 and S5 all
    carry strength = 0.310, except S5 where strength is one of the two varied
    values. S3 varies toughness and E, so its strength should also be 0.310 -
    but Table 1 and the Figure 2 caption both give 0.200 for S3.

    Both places in the paper agree with each other, so this is not a
    typesetting slip in one location. We keep the published value and flag the
    inconsistency; see docs/discrepancies.md, entry D2. If 0.310 were used
    instead, S3's target spread and therefore its R2 would change.
    """
    assert config.SELFCONSISTENCY_SETS["S3"][4] == pytest.approx(0.200)
    others = [
        config.SELFCONSISTENCY_SETS[n][4] for n in ("S1", "S2", "S4")
    ]
    assert all(o == pytest.approx(0.310) for o in others)


def test_paper_r2_values_present_for_every_set():
    assert set(config.ALL_R2_PAPER) == set(config.ALL_SETS)
    assert all(0.0 <= r <= 1.0 for r in config.ALL_R2_PAPER.values())


def test_notebook_budget_constant():
    assert config.PAPER_BUDGET_TOTAL == 2048
