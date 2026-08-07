"""Novelty of a generated sequence against the known silk sequence set.

The released notebook's `is_novel` is an exact string membership test against
ALL_SILK_SEQ.csv. We keep that test verbatim, because reproducing the paper
means reproducing its definition of novelty, but we also report two softer
measures alongside it, because exact-match novelty is close to free: a single
substituted residue in a 400-residue sequence passes it.

The paper's own novelty claim (Section 2.2) rests on BLAST query cover and
percent identity against the wider protein databases, not on this exact-match
test. That part needs network BLAST or a local database; see
scripts/05_novelty_blast.py, which prepares the FASTA and documents what could
and could not be run offline.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@functools.lru_cache(maxsize=1)
def load_known_sequences(path: Path | None = None) -> tuple[str, ...]:
    """The 10,449 silk sequences shipped in the authors' repo."""
    path = path or config.ALL_SILK_SEQ_CSV
    df = pd.read_csv(path)
    if "Sequence" not in df.columns:
        raise ValueError(f"{path} has no 'Sequence' column; got {list(df.columns)}")
    return tuple(df["Sequence"].astype(str).tolist())


@functools.lru_cache(maxsize=1)
def _known_set() -> frozenset[str]:
    return frozenset(load_known_sequences())


def is_novel_exact(sequence: str) -> bool:
    """The paper's / notebook's test: not a byte-identical member of the set."""
    return sequence not in _known_set()


# --------------------------------------------------------------------------
# Softer novelty measures (ours)
# --------------------------------------------------------------------------


def _kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def max_kmer_jaccard(sequence: str, k: int = 6, known: tuple[str, ...] | None = None) -> float:
    """Highest k-mer Jaccard similarity to any known sequence.

    A cheap, deterministic, offline stand-in for alignment identity. It is not
    BLAST and we do not present it as BLAST: Jaccard over k-mer sets ignores
    order and position, so it will differ from percent identity. Its value is
    that it runs without network access and gives a reproducible number, which
    lets the novelty claim be examined even where BLAST is unavailable.
    """
    known = known or load_known_sequences()
    q = _kmers(sequence, k)
    if not q:
        return float("nan")
    best = 0.0
    for ref in known:
        r = _kmers(ref, k)
        if not r:
            continue
        inter = len(q & r)
        if inter == 0:
            continue
        j = inter / len(q | r)
        if j > best:
            best = j
    return best


def max_kmer_containment(
    sequence: str, k: int = 6, known: tuple[str, ...] | None = None
) -> float:
    """Fraction of the query's k-mers that appear in the best-matching known
    sequence. Unlike Jaccard this is insensitive to large length differences,
    which matters because generated sequences are often much shorter than the
    natural spidroins they resemble.
    """
    known = known or load_known_sequences()
    q = _kmers(sequence, k)
    if not q:
        return float("nan")
    best = 0.0
    for ref in known:
        r = _kmers(ref, k)
        if not r:
            continue
        c = len(q & r) / len(q)
        if c > best:
            best = c
    return best


def nearest_neighbours(
    sequence: str, known: tuple[str, ...] | None = None, k: int = 6, n: int = 2
) -> list[tuple[float, str]]:
    """The n most similar reference sequences, by k-mer containment.

    Stands in for the paper's BLAST step when selecting natural sequences to
    compare a generation against. It is not BLAST: containment over k-mer sets
    ignores residue order, so the neighbours it selects need not be those BLAST
    would select, and the scores are not percent identity. Used only to pick
    comparison specimens; never quoted as an alignment statistic.
    """
    known = known or load_known_sequences()
    q = _kmers(sequence, k)
    if not q:
        return []
    scored: list[tuple[float, str]] = []
    for ref in known:
        r = _kmers(ref, k)
        if not r:
            continue
        inter = len(q & r)
        if inter:
            scored.append((inter / len(q), ref))
    scored.sort(key=lambda t: -t[0])
    return scored[:n]


def novelty_report(sequences: list[str], k: int = 6) -> pd.DataFrame:
    known = load_known_sequences()
    rows = []
    for s in sequences:
        rows.append(
            {
                "sequence": s,
                "length": len(s),
                "novel_exact": is_novel_exact(s),
                f"max_jaccard_k{k}": max_kmer_jaccard(s, k, known),
                f"max_containment_k{k}": max_kmer_containment(s, k, known),
            }
        )
    return pd.DataFrame(rows)


def write_fasta(sequences: dict[str, str], path: Path) -> None:
    """Write sequences for submission to BLAST."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
