"""Tests for prompt formatting and output parsing.

The prompt strings are checked character by character against the examples
printed in the paper's Experimental Section, because a silently malformed
prompt would degrade conditioning without raising anything.
"""

import numpy as np
import pytest

from silkrepro import config, tasks


def test_forward_prompt_matches_paper_example():
    seq = "AAAGGAGQGGYGG"
    assert tasks.forward_prompt(seq) == "CalculateSilkContent<AAAGGAGQGGYGG>"


def test_inverse_prompt_matches_paper_example():
    props = [0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]
    assert tasks.inverse_prompt(props) == (
        "GenerateSilkContent<0.327,0.356,0.261,0.287,0.437,0.190,0.220,0.301>"
    )


def test_property_vector_always_three_decimals():
    # The fine-tuning data used three decimals; 0.2 must render as 0.200.
    assert tasks.format_property_vector([0.2] * 8) == ",".join(["0.200"] * 8)


def test_property_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        tasks.format_property_vector([0.1, 0.2, 0.3])


def test_parse_property_output_happy_path():
    v = tasks.parse_property_output("[0.100,0.200,0.300,0.400,0.500,0.600,0.700,0.800]")
    assert v is not None
    assert np.allclose(v, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


@pytest.mark.parametrize(
    "text",
    [
        "",                                    # nothing
        "0.1,0.2,0.3",                         # no brackets
        "[0.1,0.2,0.3]",                       # too few values
        "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",  # too many
        "[a,b,c,d,e,f,g,h]",                   # not numbers
        "[0.1,0.2,0.3,0.4,0.5,0.6,0.7",        # unterminated
    ],
)
def test_parse_property_output_returns_none_not_raises(text):
    """Parse failures must be countable, not exceptions swallowed by a bare
    except. The failure rate is reported in every run manifest."""
    assert tasks.parse_property_output(text) is None


def test_parse_sequence_output_accepts_valid_protein():
    assert tasks.parse_sequence_output("[GAGAGAGSAAAA]") == "GAGAGAGSAAAA"


@pytest.mark.parametrize(
    "text",
    [
        "[GAGA1234]",       # digits
        "[GAGA GAGA]",      # whitespace inside
        "[]",               # empty
        "[GAGAXBZ]",        # X, B, Z are not among the twenty standard residues
        "no brackets here",
    ],
)
def test_parse_sequence_output_rejects_non_protein(text):
    assert tasks.parse_sequence_output(text) is None


def test_first_bracket_group_is_taken():
    # Model output can run on past the answer; we take the first group only.
    assert tasks.parse_sequence_output("[GAGA] trailing [OTHER]") == "GAGA"


def test_task_of():
    assert tasks.task_of(tasks.forward_prompt("GA")) == config.FORWARD_TASK
    assert tasks.task_of(tasks.inverse_prompt([0.1] * 8)) == config.INVERSE_TASK
    assert tasks.task_of("not a task") is None


def test_valid_aa_is_the_twenty_standard_residues():
    assert len(tasks.VALID_AA) == 20
    assert "X" not in tasks.VALID_AA and "U" not in tasks.VALID_AA
