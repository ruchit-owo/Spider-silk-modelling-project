"""Batched forward prediction must equal unbatched forward prediction.

The reproduction batches the forward task for speed, which requires
left-padding sequences of different lengths into one tensor. If padding
perturbed the result, every property prediction in the project would be
slightly wrong in a way no other test would catch, because there is no
reference value to compare against.

So it is checked directly, on real sequences of deliberately unequal length
(padding only does anything when lengths differ).

Marked slow: needs the checkpoint and a GPU. Run with
    pytest -m slow
and re-run after any transformers or torch upgrade.
"""

import numpy as np
import pytest

from silkrepro import novelty

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    from silkrepro.model import SilkomeGPT

    return SilkomeGPT()


@pytest.fixture(scope="module")
def sequences():
    """Real silk sequences spanning a wide length range, so that the batch is
    heavily padded. Equal-length inputs would make the test vacuous."""
    known = [s for s in novelty.load_known_sequences() if 80 <= len(s) <= 600]
    known = sorted(known, key=len)
    picks = [known[0], known[len(known) // 4], known[len(known) // 2], known[-1]]
    assert len({len(s) for s in picks}) > 1, "test needs unequal lengths"
    return picks


def test_batched_matches_unbatched(model, sequences):
    single = [model.predict_properties(s, seed=0) for s in sequences]
    batched = model.predict_properties_batch(sequences, seed=0, batch_size=len(sequences))

    assert len(single) == len(batched)
    for i, (a, b) in enumerate(zip(single, batched)):
        assert (a is None) == (b is None), f"parse disagreement at index {i}"
        if a is None:
            continue
        assert np.allclose(a, b, atol=1e-6), (
            f"sequence {i} (length {len(sequences[i])}): "
            f"unbatched {a} vs batched {b}"
        )


def test_batch_size_does_not_change_results(model, sequences):
    """Different padding widths must not change anything either."""
    b1 = model.predict_properties_batch(sequences, seed=0, batch_size=1)
    b4 = model.predict_properties_batch(sequences, seed=0, batch_size=4)
    for i, (a, b) in enumerate(zip(b1, b4)):
        assert (a is None) == (b is None), i
        if a is not None:
            assert np.allclose(a, b, atol=1e-6), f"index {i}: {a} vs {b}"


def test_forward_reproduces_the_papers_worked_example(model):
    """The Experimental Section prints a sequence and the property vector the
    fine-tuning data pairs with it. The released checkpoint should return that
    vector. This is the single strongest check that we have the right model
    and the right prompt format."""
    from silkrepro.paper_examples import (
        PAPER_EXAMPLE_PROPERTIES,
        PAPER_EXAMPLE_SEQUENCE,
    )

    pred = model.predict_properties(PAPER_EXAMPLE_SEQUENCE, seed=0)
    assert pred is not None
    assert np.allclose(pred, PAPER_EXAMPLE_PROPERTIES, atol=1e-3), (
        f"expected {PAPER_EXAMPLE_PROPERTIES}, got {pred.tolist()}"
    )
