"""Reading the Silkome v1 release.

Two files, both from Arakawa et al., Sci. Adv. 8, eabo6043 (2022):

    spider-silkome-database.v1.prot.fasta
        11,155 records. Headers are pipe-separated:
        seq_id|idv_id|family|genus|species|type|subtype|region
        `region` is NTD or CTD, so records are terminal-domain-anchored
        spidroin fragments rather than complete proteins.

    mechanical_properties.csv
        446 rows, one per tested individual, keyed by `idv_id`.

The join key is `idv_id` - the individual spider whose fibre was tested. Not
`ncbi_tax_id`: that field exists in both files but shares no values between
them, so joining on it returns nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FASTA_FIELDS = ["seq_id", "idv_id", "family", "genus", "species", "type", "subtype", "region"]

# Silkome's column names, in the order of config.PROPERTY_NAMES.
SILKOME_COLUMNS = [
    "toughness",
    "toughness_sd",
    "young's_modulus",
    "young's_modulus_sd",
    "tensile_strength",
    "tensile_strength_sd",
    "strain_at_break",
    "strain_at_break_sd",
]

SPIDROIN_FAMILIES = ("MaSp", "MiSp", "AcSp", "AgSp", "PySp", "CySp", "Flag", "CrSp")


def read_fasta(path: Path) -> pd.DataFrame:
    records: list[tuple[str, str]] = []
    name, buf = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))

    parts = [r[0].split("|") for r in records]
    bad = [i for i, p in enumerate(parts) if len(p) != len(FASTA_FIELDS)]
    if bad:
        raise ValueError(
            f"{len(bad)} FASTA headers lack {len(FASTA_FIELDS)} fields; "
            f"first offender: {records[bad[0]][0]!r}"
        )
    df = pd.DataFrame(parts, columns=FASTA_FIELDS)
    df["sequence"] = [r[1] for r in records]
    df["idv_id"] = df["idv_id"].astype(int)
    return df


def join_sequences_to_properties(fasta_path: Path, mech_path: Path) -> pd.DataFrame:
    """All spidroin types, joined on idv_id, complete properties only.

    This is the population the paper's normalisation constants are taken over
    ("across the entire dataset (not limited to MaSp sequences)"), so it is
    also the population `normalisation_from` uses.
    """
    fasta = read_fasta(fasta_path)
    mech = pd.read_csv(mech_path)

    missing = [c for c in SILKOME_COLUMNS if c not in mech.columns]
    if missing:
        raise ValueError(f"{mech_path} is missing columns {missing}")

    # family/genus/species exist in both. Keep the FASTA's (they describe the
    # sequence record) and prefix the CSV's so nothing is silently shadowed.
    overlap = [c for c in mech.columns if c in fasta.columns and c != "idv_id"]
    mech = mech.rename(columns={c: f"mech_{c}" for c in overlap})

    joined = fasta.merge(mech, on="idv_id", how="inner")
    complete = joined.dropna(subset=SILKOME_COLUMNS).copy()

    # Standardised property columns in the project's naming and order.
    for i, name in enumerate(config.PROPERTY_NAMES):
        complete[name] = complete[SILKOME_COLUMNS[i]].astype(float)

    complete["family_group"] = complete["type"].map(spidroin_family)
    return complete.reset_index(drop=True)


def spidroin_family(type_label: str) -> str:
    """Collapse fine-grained type labels (MaSp1, MaSp3B, ...) to a family."""
    for fam in SPIDROIN_FAMILIES:
        if str(type_label).startswith(fam):
            return fam
    return "other"


def normalisation_from(complete: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Min/max per property over the supplied rows.

    Passed the full all-types table, this reproduces the paper's Table S5; see
    ASSUMPTIONS.md A2 for the verification.
    """
    mins = complete[config.PROPERTY_NAMES].min().to_numpy(dtype=float)
    maxs = complete[config.PROPERTY_NAMES].max().to_numpy(dtype=float)
    return mins, maxs


def masp_subset(complete: pd.DataFrame) -> pd.DataFrame:
    """The 1,033 MaSp rows the paper fine-tunes on. See ASSUMPTIONS.md A1."""
    return complete[complete["type"].str.startswith("MaSp")].reset_index(drop=True)
