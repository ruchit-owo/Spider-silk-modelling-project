"""Sequence-content ablations.

The question these answer: when SilkomeGPT's forward task maps a sequence to
eight numbers, what in the sequence is it reading? Composition alone? Local
motifs? Residue order? Length?

Each ablation is a controlled edit that destroys one kind of information while
holding others fixed. The prediction shift it produces is only meaningful against a reference, and
this project uses two.

`repeat_noise_floor` resubmits the identical prompt. Under the project's greedy
decoding (assumption A5) that floor is zero by construction and can never
reject anything; it is kept only to demonstrate the determinism, not as a
safeguard. It would be a live check under the released notebook's sampled
decoding.

`conservative_point_mutation` provides the reference that does work: the
smallest edit that changes the sequence at all, one residue swapped for a
chemically similar one. An ablation that moves the prediction no further than a
single conservative substitution has not shown anything.

Confounds are named in each docstring rather than left for the reader to spot.
The two that recur:

  - Deleting residues changes length as well as content. Where a motif knockout
    deletes, a length-matched control is run alongside it.
  - Shuffling preserves composition exactly but produces a sequence far outside
    the training distribution. A model that behaves oddly on shuffled input may
    be reacting to the distribution shift rather than to the loss of order.
    This limits what the shuffle ablation can conclude, and the writeup says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .protparam import AA_ORDER

# --------------------------------------------------------------------------
# Motifs
# --------------------------------------------------------------------------

# Canonical spidroin motifs. These are the well-established ones from the silk
# literature, written as regexes. They are NOT the Table S4 motif set from the
# paper's Supporting Information - that set is loaded separately from
# data/raw/table_s4_motifs.csv when available (see motifs.py). These are used
# for the knockout ablation, where what matters is that the motif is real and
# frequent, not that it matches the paper's list.
CANONICAL_MOTIFS = {
    "polyA": r"A{4,}",            # beta-sheet nanocrystal former
    "GGX": r"(?:GG[ALQSY])+",     # 3-10 helix / amorphous
    "GPGXX": r"(?:GPG[A-Z]{2})+",  # elastic beta-spiral, MaSp2
    "GA": r"(?:GA){3,}",          # alternating Gly-Ala
    "QQ": r"Q{2,}",
}


def find_motif_spans(sequence: str, pattern: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(pattern, sequence)]


# --------------------------------------------------------------------------
# Ablation operators
# --------------------------------------------------------------------------


def shuffle_residues(sequence: str, rng: np.random.Generator) -> str:
    """Uniform permutation. Composition and length exactly preserved; all
    order information destroyed, local and global."""
    chars = list(sequence)
    rng.shuffle(chars)
    return "".join(chars)


def shuffle_blocks(sequence: str, rng: np.random.Generator, block: int = 20) -> str:
    """Permute contiguous blocks. Local motifs survive inside blocks; the
    arrangement of blocks is destroyed. Separates 'motif content' from 'motif
    arrangement', which the uniform shuffle cannot."""
    blocks = [sequence[i : i + block] for i in range(0, len(sequence), block)]
    order = rng.permutation(len(blocks))
    return "".join(blocks[i] for i in order)


def reverse_sequence(sequence: str) -> str:
    """Reverse. Composition and all unordered k-mer *multisets* are preserved
    up to reversal; N-to-C directionality is destroyed. Deterministic, so it
    needs no seed and is a clean single-factor test."""
    return sequence[::-1]


def knockout_motif_delete(sequence: str, pattern: str) -> tuple[str, int]:
    """Excise every match of `pattern`. Returns (edited, n_removed_residues).

    Changes length. Always pair with `length_matched_deletion` below.
    """
    spans = find_motif_spans(sequence, pattern)
    if not spans:
        return sequence, 0
    keep, prev = [], 0
    removed = 0
    for a, b in spans:
        keep.append(sequence[prev:a])
        removed += b - a
        prev = b
    keep.append(sequence[prev:])
    return "".join(keep), removed


def length_matched_deletion(
    sequence: str, n_remove: int, rng: np.random.Generator
) -> str:
    """Remove the same number of residues, but at random positions.

    The control for knockout_motif_delete: if the motif knockout and this
    control shift the prediction by the same amount, the shift was about
    length, not about the motif.
    """
    if n_remove <= 0 or n_remove >= len(sequence):
        return sequence
    idx = set(rng.choice(len(sequence), size=n_remove, replace=False).tolist())
    return "".join(c for i, c in enumerate(sequence) if i not in idx)


def contiguous_block_deletion(
    sequence: str, n_remove: int, rng: np.random.Generator
) -> str:
    """Remove `n_remove` residues as one contiguous block at a random position.

    The second control for knockout_motif_delete, and the more important of the
    two. `length_matched_deletion` removes the same *number* of residues but
    scatters them across the sequence, which creates many small local
    disruptions; a motif knockout removes a few contiguous runs. Comparing a
    knockout only against the scattered control confounds "which residues were
    removed" with "how the removal was distributed".

    Removing one block of the same size isolates the first question. Neither
    control is perfect on its own - the motif knockout removes several runs,
    not one block - but a knockout that sits between the two controls is
    telling a different story from one that sits outside both.
    """
    if n_remove <= 0 or n_remove >= len(sequence):
        return sequence
    start = int(rng.integers(0, len(sequence) - n_remove + 1))
    return sequence[:start] + sequence[start + n_remove :]


def knockout_motif_substitute(
    sequence: str, pattern: str, rng: np.random.Generator
) -> tuple[str, int]:
    """Replace matched residues with residues drawn from the sequence's own
    composition. Length preserved, motif destroyed, composition approximately
    preserved. Complements the deletion variant; if both agree, length is not
    driving the result."""
    spans = find_motif_spans(sequence, pattern)
    if not spans:
        return sequence, 0
    pool = list(sequence)
    chars = list(sequence)
    n = 0
    for a, b in spans:
        for i in range(a, b):
            chars[i] = pool[int(rng.integers(len(pool)))]
            n += 1
    return "".join(chars), n


def keep_termini(sequence: str, n_term: int = 130, c_term: int = 100) -> str:
    """Keep only the N- and C-terminal domains, drop the repetitive core.

    Spidroin terminal domains are roughly 130 (N) and 100 (C) residues. THE
    PAPER DOES NOT SPECIFY domain boundaries, and this project does not run a
    domain annotator, so these are nominal literature values used as a coarse
    split. The complementary `keep_core` uses the same cut points, so the two
    together partition the sequence and the comparison between them is
    internally consistent even if the boundary is imprecise.
    """
    if len(sequence) <= n_term + c_term:
        return sequence
    return sequence[:n_term] + sequence[-c_term:]


def keep_core(sequence: str, n_term: int = 130, c_term: int = 100) -> str:
    """Complement of keep_termini: the repetitive core only."""
    if len(sequence) <= n_term + c_term:
        return ""
    return sequence[n_term:-c_term]


def truncate_to(sequence: str, length: int) -> str:
    return sequence[:length]


def random_sequence_matched(sequence: str, rng: np.random.Generator) -> str:
    """Uniform random residues, matched length. The furthest-out control:
    neither composition nor order preserved. Establishes what the model does
    with input carrying no silk signal at all."""
    return "".join(rng.choice(AA_ORDER, size=len(sequence)))


# --------------------------------------------------------------------------
# Running an ablation
# --------------------------------------------------------------------------


@dataclass
class AblationResult:
    name: str
    original_sequence: str
    edited_sequence: str
    original_pred: np.ndarray
    edited_pred: np.ndarray
    n_residues_changed: int

    @property
    def delta(self) -> np.ndarray:
        return self.edited_pred - self.original_pred

    @property
    def abs_delta_mean(self) -> float:
        return float(np.mean(np.abs(self.delta)))

    def to_row(self, property_names: list[str]) -> dict:
        row = {
            "ablation": self.name,
            "orig_len": len(self.original_sequence),
            "edit_len": len(self.edited_sequence),
            "n_residues_changed": self.n_residues_changed,
            "abs_delta_mean": self.abs_delta_mean,
        }
        for i, p in enumerate(property_names):
            row[f"orig_{p}"] = float(self.original_pred[i])
            row[f"edit_{p}"] = float(self.edited_pred[i])
            row[f"delta_{p}"] = float(self.delta[i])
        return row


def conservative_point_mutation(
    sequence: str, rng: np.random.Generator
) -> tuple[str, int]:
    """Substitute one residue for a chemically similar one.

    The smallest edit that changes the sequence at all, and a far more useful
    reference than resubmitting the identical prompt. Under greedy decoding the
    repeat floor is exactly zero by construction, so it can never reject
    anything; a one-residue conservative substitution gives a floor an effect
    can actually fall below.

    "Conservative" means within the same hydrophobic / polar / charged class,
    so the edit is minimal in composition as well as in length. Positions are
    chosen uniformly; the returned index lets a caller check the choice.
    """
    from .aa_groups import SCHEME_A, group_of

    if not sequence:
        return sequence, -1
    for _ in range(50):  # a few tries in case a class has no alternative
        i = int(rng.integers(len(sequence)))
        aa = sequence[i]
        if aa not in AA_ORDER:
            continue
        peers = sorted(SCHEME_A[group_of(aa, SCHEME_A)] - {aa})
        if not peers:
            continue
        repl = peers[int(rng.integers(len(peers)))]
        return sequence[:i] + repl + sequence[i + 1 :], i
    return sequence, -1


def repeat_noise_floor(
    predict: Callable[[str, int], np.ndarray | None],
    sequences: list[str],
    n_repeats: int = 8,
    seed: int = 0,
) -> dict:
    """Spread of the forward task's own output on unmodified input.

    The forward task samples rather than decoding greedily, so it has a
    non-zero noise floor. Every ablation effect must be compared against this;
    an effect smaller than the floor is not evidence of anything.

    Returns per-property and pooled statistics of |prediction - mean
    prediction| across repeats of the identical prompt.
    """
    devs: list[np.ndarray] = []
    per_seq = []
    for si, seq in enumerate(sequences):
        preds = []
        for r in range(n_repeats):
            p = predict(seq, seed + 1000 * si + r)
            if p is not None:
                preds.append(p)
        if len(preds) < 2:
            continue
        arr = np.vstack(preds)
        d = np.abs(arr - arr.mean(axis=0, keepdims=True))
        devs.append(d)
        per_seq.append(
            {
                "sequence_index": si,
                "length": len(seq),
                "n_valid": len(preds),
                "mean_abs_dev": float(d.mean()),
                "max_abs_dev": float(d.max()),
                "n_distinct_outputs": len({tuple(np.round(x, 6)) for x in preds}),
            }
        )

    if not devs:
        return {"error": "no sequence produced two parseable predictions"}

    stacked = np.vstack(devs)
    return {
        "n_sequences": len(per_seq),
        "n_repeats": n_repeats,
        "pooled_mean_abs_dev": float(stacked.mean()),
        "pooled_p95_abs_dev": float(np.percentile(stacked, 95)),
        "pooled_max_abs_dev": float(stacked.max()),
        "per_property_mean_abs_dev": stacked.mean(axis=0).tolist(),
        "per_sequence": per_seq,
    }
