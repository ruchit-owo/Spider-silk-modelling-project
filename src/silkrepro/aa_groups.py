"""Amino-acid classification schemes.

Needed for the composition-difference analysis in
`scripts/14_composition_association.py`, which groups residues into
hydrophobic / polar / charged panels and normalises within each group.

The source being followed (Pandey, Chen & Keten, Commun. Mater. 2024,
Supplementary Note 8) says only "hydrophobic, polar, and charged groups" and
does not give the assignment table. THE SOURCE DOES NOT SPECIFY THIS. Since the
normalisation in their Supplementary Equation 3 divides by the maximum *within
a group*, the grouping directly determines the resulting bar heights, so the
choice cannot be left implicit.

We therefore implement two standard schemes and report both. If the selected
residues agree between them, the ambiguity did not matter; if they disagree,
that disagreement is the honest result and is reported as such rather than
resolved by picking the one that matches better.

The two schemes differ only in the residues that are genuinely contested:
glycine, cysteine, histidine and tyrosine. Everything else is assigned the same
way in any textbook treatment.
"""

from __future__ import annotations

# Scheme A - the classic textbook split (e.g. Lehninger, Nelson & Cox).
# Nonpolar/hydrophobic side chains, polar uncharged side chains, and those
# charged at physiological pH. Glycine is placed with the nonpolar residues
# (its side chain is a hydrogen), cysteine with the polar residues (thiol),
# histidine with the charged residues (positively charged at pH 7 in part).
SCHEME_A = {
    "hydrophobic": set("AVLIPFMWG"),
    "polar": set("STCYNQ"),
    "charged": set("DEKRH"),
}

# Scheme B - a common alternative that treats glycine as polar/special given
# its lack of a side chain, cysteine as hydrophobic given its behaviour in
# folded cores, histidine as polar given that it is only partly protonated at
# pH 7, and tyrosine as hydrophobic given its large aromatic ring.
SCHEME_B = {
    "hydrophobic": set("AVLIPFMWCY"),
    "polar": set("STNQGH"),
    "charged": set("DEKR"),
}

SCHEMES = {"A_lehninger_style": SCHEME_A, "B_alternative": SCHEME_B}

# Residues the source paper selected from its own version of this analysis.
# Used only for comparison; we do not tune anything to reproduce it.
KETEN_SELECTED = list("VIPLFTQYNSDEKR")


def validate(scheme: dict[str, set[str]]) -> None:
    """A scheme must partition the twenty standard residues exactly once."""
    all_aa = set("ACDEFGHIKLMNPQRSTVWY")
    union: set[str] = set()
    for group, members in scheme.items():
        overlap = union & members
        if overlap:
            raise ValueError(f"residues in more than one group: {sorted(overlap)}")
        union |= members
    if union != all_aa:
        missing = sorted(all_aa - union)
        extra = sorted(union - all_aa)
        raise ValueError(f"not a partition; missing {missing}, unexpected {extra}")


def group_of(residue: str, scheme: dict[str, set[str]]) -> str:
    for group, members in scheme.items():
        if residue in members:
            return group
    raise KeyError(residue)


def differences(scheme_a: dict[str, set[str]], scheme_b: dict[str, set[str]]) -> dict:
    """Which residues the two schemes assign differently. Reported in output."""
    out = {}
    for aa in sorted("ACDEFGHIKLMNPQRSTVWY"):
        ga, gb = group_of(aa, scheme_a), group_of(aa, scheme_b)
        if ga != gb:
            out[aa] = {"scheme_a": ga, "scheme_b": gb}
    return out
