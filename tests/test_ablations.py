"""Tests for the ablation operators.

Each operator claims to destroy one kind of information while preserving
another. These tests check the preservation claims, because an operator that
quietly changes composition as well as order would make its ablation
uninterpretable.
"""

from collections import Counter

import numpy as np
import pytest

from silkrepro import ablations

SEQ = (
    "MGQGGYGGLGSQGAGRGGQGAGAAAAAAGGAGQGGYGGLGSQGAGQGAGAAAAAAGGAGQGGYGG"
    "QGAGQGAGAAAAAAGPGGYGPGQQGPGGYGPGQQGPSGPGSAAAAAAAASRLSSPEASSRVSSAV"
)


def test_shuffle_preserves_composition_exactly():
    rng = np.random.default_rng(0)
    out = ablations.shuffle_residues(SEQ, rng)
    assert Counter(out) == Counter(SEQ)
    assert len(out) == len(SEQ)
    assert out != SEQ  # astronomically unlikely to be equal


def test_block_shuffle_preserves_composition():
    rng = np.random.default_rng(0)
    out = ablations.shuffle_blocks(SEQ, rng, block=20)
    assert Counter(out) == Counter(SEQ)


def test_block_shuffle_preserves_more_local_structure_than_uniform_shuffle():
    """The point of having both: block shuffle should retain more adjacent
    residue pairs (dipeptides) than a uniform shuffle does."""
    rng = np.random.default_rng(1)

    def dipeptides(s):
        return Counter(s[i : i + 2] for i in range(len(s) - 1))

    orig = dipeptides(SEQ)

    def overlap(s):
        d = dipeptides(s)
        return sum(min(orig[k], d[k]) for k in orig)

    block_overlaps = [overlap(ablations.shuffle_blocks(SEQ, rng, 20)) for _ in range(20)]
    unif_overlaps = [overlap(ablations.shuffle_residues(SEQ, rng)) for _ in range(20)]
    assert np.mean(block_overlaps) > np.mean(unif_overlaps)


def test_reverse_is_involutive_and_preserves_composition():
    out = ablations.reverse_sequence(SEQ)
    assert Counter(out) == Counter(SEQ)
    assert ablations.reverse_sequence(out) == SEQ


def test_polya_knockout_removes_all_polya_runs():
    edited, removed = ablations.knockout_motif_delete(SEQ, ablations.CANONICAL_MOTIFS["polyA"])
    assert removed > 0
    assert ablations.find_motif_spans(edited, ablations.CANONICAL_MOTIFS["polyA"]) == []
    assert len(edited) == len(SEQ) - removed


def test_length_matched_deletion_removes_exactly_n():
    rng = np.random.default_rng(0)
    out = ablations.length_matched_deletion(SEQ, 17, rng)
    assert len(out) == len(SEQ) - 17
    # It is a subsequence of the original, in order.
    it = iter(SEQ)
    assert all(c in it for c in out)


def test_substitution_knockout_preserves_length():
    rng = np.random.default_rng(0)
    edited, n = ablations.knockout_motif_substitute(
        SEQ, ablations.CANONICAL_MOTIFS["polyA"], rng
    )
    assert len(edited) == len(SEQ)
    assert n > 0


def test_termini_and_core_partition_the_sequence():
    long_seq = SEQ * 5
    t = ablations.keep_termini(long_seq, 130, 100)
    c = ablations.keep_core(long_seq, 130, 100)
    assert len(t) + len(c) == len(long_seq)
    assert t[:130] + c + t[130:] == long_seq


def test_short_sequences_are_returned_unchanged_by_termini_split():
    short = "GAGAGA"
    assert ablations.keep_termini(short, 130, 100) == short
    assert ablations.keep_core(short, 130, 100) == ""


def test_random_matched_control_has_right_length_and_alphabet():
    rng = np.random.default_rng(0)
    out = ablations.random_sequence_matched(SEQ, rng)
    assert len(out) == len(SEQ)
    assert set(out) <= set(ablations.AA_ORDER)


def test_noise_floor_detects_a_deterministic_predictor():
    """A predictor that always returns the same vector must show a zero floor."""
    fixed = np.arange(8) / 10.0
    out = ablations.repeat_noise_floor(lambda s, seed: fixed, [SEQ], n_repeats=5)
    assert out["pooled_mean_abs_dev"] == pytest.approx(0.0)
    assert out["per_sequence"][0]["n_distinct_outputs"] == 1


def test_noise_floor_detects_a_noisy_predictor():
    rng = np.random.default_rng(0)
    out = ablations.repeat_noise_floor(
        lambda s, seed: rng.normal(0.5, 0.1, size=8), [SEQ], n_repeats=20
    )
    assert out["pooled_mean_abs_dev"] > 0.01
    assert out["per_sequence"][0]["n_distinct_outputs"] == 20
