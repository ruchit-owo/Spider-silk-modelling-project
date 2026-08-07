"""Loading the Silkome sequence-property table and the normalisation constants.

State of the data, as of writing:

  ALL_SILK_SEQ.csv        shipped in the authors' GitHub repo. 10,449 silk
                          sequences, no properties. Used only for the
                          exact-match novelty test.

  1,033 MaSp pairs        NOT in the authors' repo. They come from the Silkome
                          dataset (Arakawa et al., Sci. Adv. 2022) as curated
                          by the paper's Experimental Section. Must be placed
                          in data/raw/ by hand - see data/raw/README.md.

  Table S5 normalisation  In the paper's Supporting Information. Gives the min
                          and max used to map each of the eight properties to
                          0-1. Must be placed in data/raw/ by hand.

Nothing in this module invents data. If a required file is absent, the loader
raises with the exact filename and expected columns, and the scripts that need
it record "not run: input missing" rather than substituting anything.

Important detail about the normalisation, from the Experimental Section: the
min and max were computed "across the entire dataset (not limited to MaSp
sequences)". So the MaSp subset does not span the full 0-1 range, and
recomputing min/max from the 1,033 MaSp rows alone would give different
constants. If Table S5 is unavailable, `derive_normalisation` can recover
constants from MaSp only, but it labels them as such and the discrepancy is
recorded rather than papered over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Canonical file names this project expects in data/raw/.
SILKOME_PAIRS_CSV = "silkome_masp_pairs.csv"
TABLE_S5_JSON = "table_s5_normalisation.json"
TABLE_S4_MOTIFS_CSV = "table_s4_motifs.csv"

# Columns the pairs table must have. Raw (un-normalised) units are preferred
# because they let us apply Table S5 ourselves and check the result.
REQUIRED_PAIR_COLUMNS = [
    "sequence",
    "toughness",
    "toughness_sd",
    "E",
    "E_sd",
    "strength",
    "strength_sd",
    "strain",
    "strain_sd",
]


class MissingInput(FileNotFoundError):
    """Raised when a hand-supplied input file is absent.

    Carries instructions so the failure is actionable rather than just loud.
    """


@dataclass
class Normalisation:
    """Min/max per property, plus where the constants came from."""

    mins: np.ndarray
    maxs: np.ndarray
    source: str  # "table_s5" or "derived_from_masp_subset"

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        rng = np.where(self.maxs - self.mins == 0, 1.0, self.maxs - self.mins)
        return (np.asarray(raw, dtype=float) - self.mins) / rng

    def denormalise(self, norm: np.ndarray) -> np.ndarray:
        return np.asarray(norm, dtype=float) * (self.maxs - self.mins) + self.mins

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "properties": config.PROPERTY_NAMES,
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
        }


def _raw(path_name: str) -> Path:
    return config.DATA_RAW / path_name


def load_pairs(path: Path | None = None) -> pd.DataFrame:
    """The 1,033 MaSp sequence-property pairs, in raw units."""
    path = path or _raw(SILKOME_PAIRS_CSV)
    if not path.exists():
        raise MissingInput(
            f"{path} not found.\n"
            f"This file is not in the authors' GitHub repo. Build it from the "
            f"Silkome dataset (Arakawa et al., Sci. Adv. 2022, supplementary) "
            f"as described in data/raw/README.md.\n"
            f"Required columns: {REQUIRED_PAIR_COLUMNS}"
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; has {list(df.columns)}")
    return df


def load_normalisation(path: Path | None = None) -> Normalisation:
    """Table S5 min/max constants."""
    path = path or _raw(TABLE_S5_JSON)
    if not path.exists():
        raise MissingInput(
            f"{path} not found.\n"
            f"These are the eight (min, max) pairs in Table S5 of the paper's "
            f"Supporting Information. Transcribe them into JSON as:\n"
            f'  {{"toughness": [min, max], "toughness_sd": [min, max], ...}}\n'
            f"using the property names {config.PROPERTY_NAMES}."
        )
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    missing = [p for p in config.PROPERTY_NAMES if p not in d]
    if missing:
        raise ValueError(f"{path} missing entries for {missing}")
    mins = np.array([float(d[p][0]) for p in config.PROPERTY_NAMES])
    maxs = np.array([float(d[p][1]) for p in config.PROPERTY_NAMES])
    return Normalisation(mins=mins, maxs=maxs, source="table_s5")


def derive_normalisation(df: pd.DataFrame) -> Normalisation:
    """Fallback: min/max from the supplied rows only.

    THE PAPER SPECIFIES that its constants come from the entire silkome
    dataset, not the MaSp subset. If this function is used, the resulting
    normalised values will not match the paper's, and every downstream number
    conditioned on them is affected. Scripts that fall back to this record
    `normalisation_source = derived_from_masp_subset` in their manifest, and
    docs/discrepancies.md carries the consequence.
    """
    mins = df[config.PROPERTY_NAMES].min().to_numpy(dtype=float)
    maxs = df[config.PROPERTY_NAMES].max().to_numpy(dtype=float)
    return Normalisation(mins=mins, maxs=maxs, source="derived_from_masp_subset")


def normalised_targets(df: pd.DataFrame, norm: Normalisation) -> np.ndarray:
    """(N, 8) normalised property matrix in config.PROPERTY_NAMES order."""
    raw = df[config.PROPERTY_NAMES].to_numpy(dtype=float)
    return norm.normalise(raw)


def check_normalisation_consistency(norm_values: np.ndarray) -> dict:
    """Sanity check on normalised data.

    Under the paper's scheme, values from the whole silkome lie in [0, 1] and
    the MaSp subset lies inside that. Values outside [0, 1] mean the constants
    do not belong to this data. Reported, not corrected.
    """
    v = np.asarray(norm_values, dtype=float)
    return {
        "min": v.min(axis=0).tolist(),
        "max": v.max(axis=0).tolist(),
        "n_below_zero": int((v < 0).sum()),
        "n_above_one": int((v > 1).sum()),
        "in_unit_interval": bool((v >= 0).all() and (v <= 1).all()),
    }


def load_table_s4_motifs(path: Path | None = None) -> pd.DataFrame:
    """The motif set used in the paper's Figure 7 analysis.

    Expected columns: motif_id (e.g. T_neg_1), pattern (regex or literal),
    property (toughness/E/strength/strain), direction (pos/neg).
    """
    path = path or _raw(TABLE_S4_MOTIFS_CSV)
    if not path.exists():
        raise MissingInput(
            f"{path} not found.\n"
            f"This is Table S4 of the Supporting Information, itself derived "
            f"from Table 1 of the Silkome paper. Transcribe it with columns: "
            f"motif_id, pattern, property, direction."
        )
    return pd.read_csv(path)
