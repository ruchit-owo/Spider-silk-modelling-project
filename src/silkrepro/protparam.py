"""Sequence-level protein descriptors, reproducing Figure 5.

Molecular weight, instability index and isoelectric point come from Biopython's
ProtParam implementation, as in the paper.

The secondary-structure fractions need care, and the reason is worth setting
out in full because it changes how Figure 5c should be read.

The Figure 5 caption defines the three fractions as the proportion of residues
that "tend to be in"

    helix : V I Y F W L
    turn  : N P G S
    sheet : E M A L

Those are exactly the groupings used by Biopython's
`ProteinAnalysis.secondary_structure_fraction()` up to and including v1.81,
and the caption reproduces that function's own documentation of the time.

In v1.82 Biopython changed the method. Its current docstring records why:

    "Note that, prior to v1.82, this method wrongly returned (Sheet, Turn,
    Helix) while claiming to return (Helix, Turn, Sheet)."

That is, the set V I Y F W L is a *sheet* propensity set and E M A L is a
*helix* propensity set; the pre-1.82 code returned them under swapped labels.
v1.82 also widened the groups to match the literature it cites (Haimov &
Srebnik 2016; Hutchinson & Thornton 1994; Kim & Berg 1993), giving

    helix : E M A L K
    turn  : N P G S D
    sheet : V I Y F W L T

The consequence for the paper is that the bars labelled "helix" in Figure 5c
are, under the corrected assignment, sheet propensities, and those labelled
"sheet" are helix propensities. The paper used the library as documented at the
time; the labels were subsequently corrected upstream. This matters more here
than it would in most papers, because beta-sheet content is the central
structural quantity in spider silk.

This module therefore computes both, explicitly, without calling the
version-dependent Biopython method at all:

    secondary_structure_legacy()    - the paper's caption, verbatim. Use this
                                      to reproduce Figure 5c.
    secondary_structure_corrected() - Biopython >= 1.82 groupings. Use this if
                                      you want the fractions to mean what
                                      their names say.

Both are reported in every output table. See docs/discrepancies.md, entry D1.
"""

from __future__ import annotations

import pandas as pd

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# The paper's caption / Biopython <= 1.81.
LEGACY_HELIX_AA = set("VIYFWL")
LEGACY_TURN_AA = set("NPGS")
LEGACY_SHEET_AA = set("EMAL")

# Biopython >= 1.82.
CORRECTED_HELIX_AA = set("EMALK")
CORRECTED_TURN_AA = set("NPGSD")
CORRECTED_SHEET_AA = set("VIYFWLT")


def _analysis(sequence: str):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return ProteinAnalysis(sequence)


def amino_acid_fractions(sequence: str) -> dict[str, float]:
    """Fraction (0-1) of each of the twenty standard residues.

    Computed directly rather than via Biopython, because
    `get_amino_acids_percent()` (<= 1.81) returned fractions in 0-1 while the
    `amino_acids_percent` property (>= 1.82) returns percentages in 0-100.
    Counting here removes that ambiguity from every downstream number.
    """
    n = len(sequence)
    if n == 0:
        return dict.fromkeys(AA_ORDER, float("nan"))
    return {aa: sequence.count(aa) / n for aa in AA_ORDER}


def _fraction_in(sequence: str, group: set[str]) -> float:
    n = len(sequence)
    return sum(c in group for c in sequence) / n if n else float("nan")


def secondary_structure_legacy(sequence: str) -> dict[str, float]:
    """The Figure 5c definition as written in the paper's caption."""
    return {
        "frac_helix_legacy": _fraction_in(sequence, LEGACY_HELIX_AA),
        "frac_turn_legacy": _fraction_in(sequence, LEGACY_TURN_AA),
        "frac_sheet_legacy": _fraction_in(sequence, LEGACY_SHEET_AA),
    }


def secondary_structure_corrected(sequence: str) -> dict[str, float]:
    """Biopython >= 1.82 groupings, under which the labels are as named."""
    return {
        "frac_helix_corrected": _fraction_in(sequence, CORRECTED_HELIX_AA),
        "frac_turn_corrected": _fraction_in(sequence, CORRECTED_TURN_AA),
        "frac_sheet_corrected": _fraction_in(sequence, CORRECTED_SHEET_AA),
    }


def descriptors(sequence: str) -> dict:
    """The Figure 5a descriptors, composition, and both structure conventions.

    Raises on non-standard residues rather than substituting, since a silent
    substitution would change the molecular weight.
    """
    bad = set(sequence) - set(AA_ORDER)
    if bad:
        raise ValueError(f"non-standard residues in sequence: {sorted(bad)}")

    pa = _analysis(sequence)
    out = {
        "length": len(sequence),
        "molecular_weight": float(pa.molecular_weight()),
        "instability_index": float(pa.instability_index()),
        "isoelectric_point": float(pa.isoelectric_point()),
        "gravy": float(pa.gravy()),
        "aromaticity": float(pa.aromaticity()),
    }
    out.update(secondary_structure_legacy(sequence))
    out.update(secondary_structure_corrected(sequence))
    for aa, f in amino_acid_fractions(sequence).items():
        out[f"pct_{aa}"] = f * 100.0
    return out


def descriptor_table(named_sequences: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, seq in named_sequences.items():
        d = {"name": name, "sequence": seq}
        d.update(descriptors(seq))
        rows.append(d)
    return pd.DataFrame(rows)
