"""Motif counting and positional analysis, reproducing Figure 7.

Figure 7 has two parts:
  (a) motif counts, generated sequence vs its two BLAST neighbours;
  (b,c) kernel density estimates of *where* a given motif sits along the
        sequence, as relative position in [0, 1].

The motif definitions come from Table S4 of the Supporting Information, which
in turn derives from Table 1 of the Silkome paper. Those definitions are not in
the main text we have. Two consequences, both recorded rather than worked
around:

  1. If data/raw/table_s4_motifs.csv is present, we use it and the analysis is
     a genuine reproduction.
  2. If it is absent, we fall back to the canonical spidroin motifs in
     ablations.CANONICAL_MOTIFS. That is NOT a reproduction of Figure 7 - the
     motif set differs - and any output produced this way is labelled
     `motif_source = canonical_fallback` and excluded from reproduction claims.

The paper also states an assumption we inherit: "the specific classification of
MaSp type (e.g. MaSp1) spidroin has minimal impact on the motif analysis".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ablations import CANONICAL_MOTIFS


@dataclass
class MotifSet:
    patterns: dict[str, str]
    source: str  # "table_s4" or "canonical_fallback"


def load_motif_set() -> MotifSet:
    from .dataio import MissingInput, load_table_s4_motifs

    try:
        df = load_table_s4_motifs()
    except MissingInput:
        return MotifSet(patterns=dict(CANONICAL_MOTIFS), source="canonical_fallback")

    patterns = {}
    for _, row in df.iterrows():
        patterns[str(row["motif_id"])] = str(row["pattern"])
    return MotifSet(patterns=patterns, source="table_s4")


def count_motifs(sequence: str, patterns: dict[str, str]) -> dict[str, int]:
    """Non-overlapping match count per motif.

    re.finditer is non-overlapping. For motifs like A{4,} that is the right
    behaviour (one run of nine alanines is one poly-A tract, not six). For
    short literal motifs it undercounts overlapping occurrences; where that
    matters, count_motifs_overlapping is available and the choice is stated in
    the script that uses it.
    """
    return {name: len(re.findall(pat, sequence)) for name, pat in patterns.items()}


def count_motifs_overlapping(sequence: str, patterns: dict[str, str]) -> dict[str, int]:
    return {
        name: len(re.findall(f"(?=({pat}))", sequence)) for name, pat in patterns.items()
    }


def motif_positions(sequence: str, pattern: str) -> list[float]:
    """Relative positions in [0, 1] of each match's midpoint.

    Relative rather than absolute so that sequences of different length can be
    compared on one axis, which is what Figure 7b,c does.
    """
    n = len(sequence)
    if n == 0:
        return []
    return [((m.start() + m.end()) / 2.0) / n for m in re.finditer(pattern, sequence)]


def motif_count_table(
    named_sequences: dict[str, str], motif_set: MotifSet, per_100: bool = True
) -> pd.DataFrame:
    """Counts for a set of named sequences.

    `per_100` additionally reports density per 100 residues. Raw counts scale
    with length, and generated sequences are frequently shorter than the
    natural sequences they are compared against, so a raw-count comparison
    confounds motif usage with length. Figure 7a plots raw counts; we report
    both and say which is which.
    """
    rows = []
    for name, seq in named_sequences.items():
        counts = count_motifs(seq, motif_set.patterns)
        row = {"name": name, "length": len(seq)}
        row.update({f"count_{k}": v for k, v in counts.items()})
        if per_100:
            row.update(
                {f"per100_{k}": 100.0 * v / len(seq) if seq else np.nan for k, v in counts.items()}
            )
        rows.append(row)
    return pd.DataFrame(rows)


def motif_position_table(
    named_sequences: dict[str, str], motif_set: MotifSet
) -> pd.DataFrame:
    rows = []
    for name, seq in named_sequences.items():
        for motif, pat in motif_set.patterns.items():
            for pos in motif_positions(seq, pat):
                rows.append({"name": name, "motif": motif, "relative_position": pos})
    return pd.DataFrame(rows)
