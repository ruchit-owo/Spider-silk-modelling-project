"""Rebuild the paper's 1,033 MaSp sequence/property pairs from Silkome v1.

The paper says the dataset was "constructed and curated based on the silkome
dataset" and contains 1,033 pairs, but does not give the curation rule. We
searched for a rule that lands on exactly 1,033 and found one:

    1. Parse spider-silkome-database.v1.prot.fasta. Headers are
       seq_id|idv_id|family|genus|species|type|subtype|region.
    2. Inner-join to mechanical_properties.csv on idv_id (the individual
       spider the fibre was tested on), NOT on ncbi_tax_id.
    3. Keep rows with all eight mechanical values present.
    4. Keep rows whose `type` begins with "MaSp" - that is MaSp, MaSp1,
       MaSp2, MaSp3, MaSp2B, MaSp3B.

    -> 1,033 rows exactly.

Three things about this deserve to be said plainly.

  - Landing on 1,033 is strong evidence the rule is right, but it is still a
    reconstruction. The paper does not state it, and a different rule could in
    principle reach the same count.
  - The 1,033 rows contain 1,028 distinct sequences. Five sequences appear
    twice, paired with different property values, because the same protein was
    recovered from two individuals whose fibres tested differently. We keep
    both, since the paper's count requires it, but this means the dataset
    contains contradictory labels for five inputs. Reported, not removed.
  - The silkome protein FASTA is annotated by region, NTD or CTD. So these are
    terminal-domain-anchored records rather than complete spidroins. The
    joined MaSp set is 553 CTD and 480 NTD.

The normalisation constants are recovered the same way, and this one can be
verified rather than argued. The paper's Experimental Section prints a worked
example: a sequence together with the eight normalised values paired with it,
[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]. That sequence is
silkome record idv_id 7305 (Nephilingis livida, MaSp1, CTD). Applying min/max
taken over the joined table across ALL spidroin types - which is what the
paper means by "across the entire dataset (not limited to MaSp sequences)" -
reproduces all eight published values to within 0.0005. Taking min/max over
the raw mechanical_properties.csv instead, or over MaSp rows only, does not.

That check runs every time this script runs. If it fails, the dataset is not
the paper's and the script says so and exits non-zero.

Usage:
    python scripts/03_build_dataset.py \
        --fasta path/to/spider-silkome-database.v1.prot.fasta \
        --mech  path/to/mechanical_properties.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, runinfo  # noqa: E402

# Column names in Silkome's mechanical_properties.csv, in the order of
# config.PROPERTY_NAMES.
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

FASTA_FIELDS = ["seq_id", "idv_id", "family", "genus", "species", "type", "subtype", "region"]

# The paper's worked example, used as the verification anchor.
WORKED_EXAMPLE_IDV = 7305
WORKED_EXAMPLE_NORMALISED = [0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]

EXPECTED_ROWS = 1033


def read_fasta(path: Path) -> pd.DataFrame:
    records, name, buf = [], None, []
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
            f"{len(bad)} FASTA headers do not have {len(FASTA_FIELDS)} "
            f"pipe-separated fields; first offender: {records[bad[0]][0]!r}"
        )
    df = pd.DataFrame(parts, columns=FASTA_FIELDS)
    df["sequence"] = [r[1] for r in records]
    df["idv_id"] = df["idv_id"].astype(int)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--mech", type=Path, required=True)
    args = ap.parse_args()

    fasta = read_fasta(args.fasta)
    mech = pd.read_csv(args.mech)
    print(f"FASTA records                 {len(fasta):>6}")
    print(f"mechanical-property rows      {len(mech):>6}")

    missing = [c for c in SILKOME_COLUMNS if c not in mech.columns]
    if missing:
        raise SystemExit(f"mechanical_properties.csv is missing columns {missing}")

    # family/genus/species appear in both tables. Keep the FASTA's taxonomy
    # (it describes the sequence record) and prefix the CSV's so nothing is
    # silently shadowed.
    overlap = [c for c in mech.columns if c in fasta.columns and c != "idv_id"]
    mech = mech.rename(columns={c: f"mech_{c}" for c in overlap})
    joined = fasta.merge(mech, on="idv_id", how="inner")
    print(f"joined on idv_id              {len(joined):>6}")

    complete = joined.dropna(subset=SILKOME_COLUMNS).copy()
    print(f"with all eight properties     {len(complete):>6}")

    masp = complete[complete["type"].str.startswith("MaSp")].copy()
    print(f"MaSp types only               {len(masp):>6}   (paper states {EXPECTED_ROWS})")

    count_matches = len(masp) == EXPECTED_ROWS
    if not count_matches:
        print(f"\n  NOTE: reconstruction gives {len(masp)}, not {EXPECTED_ROWS}. "
              f"The curation rule is not stated in the paper; this difference "
              f"is a finding, not something to force. Continuing.")

    # ---- normalisation, recovered over ALL types -------------------------
    pop = complete[SILKOME_COLUMNS]
    mins = pop.min().to_numpy(dtype=float)
    maxs = pop.max().to_numpy(dtype=float)

    print("\nrecovered normalisation constants "
          f"(min/max over all {len(complete)} joined rows, all spidroin types):")
    for name, lo, hi in zip(config.PROPERTY_NAMES, mins, maxs):
        print(f"  {name:14s} [{lo:>8.4g}, {hi:>8.4g}]")

    # ---- verification against the paper's worked example -----------------
    # One individual contributes many sequence records of different spidroin
    # types, but the eight mechanical values are a property of the individual's
    # fibre, so any of its rows carries the same raw values. We select the
    # MaSp record for the printed label, since that is the one the paper's
    # worked-example sequence belongs to.
    anchor_all = complete[complete["idv_id"] == WORKED_EXAMPLE_IDV]
    anchor_masp = anchor_all[anchor_all["type"].str.startswith("MaSp")]
    anchor = anchor_masp if not anchor_masp.empty else anchor_all
    if anchor.empty:
        print(f"\n  cannot verify: idv_id {WORKED_EXAMPLE_IDV} not in the joined table")
        verified, max_err = False, float("nan")
    else:
        raw = anchor.iloc[0][SILKOME_COLUMNS].to_numpy(dtype=float)
        norm = (raw - mins) / (maxs - mins)
        expected = np.array(WORKED_EXAMPLE_NORMALISED)
        max_err = float(np.max(np.abs(norm - expected)))
        verified = max_err < 0.002  # the paper prints three decimals

        print(f"\nverification against the paper's worked example "
              f"(idv_id {WORKED_EXAMPLE_IDV}, "
              f"{anchor.iloc[0]['genus']} {anchor.iloc[0]['species']}, "
              f"{anchor.iloc[0]['type']}):")
        print(f"  raw        {np.round(raw, 4).tolist()}")
        print(f"  ours       {np.round(norm, 3).tolist()}")
        print(f"  paper      {WORKED_EXAMPLE_NORMALISED}")
        print(f"  max error  {max_err:.5f}   -> {'VERIFIED' if verified else 'MISMATCH'}")

    if not verified:
        print("\nThe recovered normalisation does not reproduce the paper's "
              "published values. Stopping rather than writing a dataset that "
              "is on a different scale from the paper's.")
        return 1

    # ---- write out -------------------------------------------------------
    out = pd.DataFrame(
        {
            "sequence": masp["sequence"].to_numpy(),
            **{
                config.PROPERTY_NAMES[i]: masp[SILKOME_COLUMNS[i]].to_numpy(dtype=float)
                for i in range(8)
            },
        }
    )
    for meta in ("idv_id", "family", "genus", "species", "type", "region"):
        out[meta] = masp[meta].to_numpy()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    pairs_path = config.DATA_RAW / "silkome_masp_pairs.csv"
    out.to_csv(pairs_path, index=False)

    norm_path = config.DATA_RAW / "table_s5_normalisation.json"
    with open(norm_path, "w", encoding="utf-8") as fh:
        json.dump(
            {n: [float(lo), float(hi)] for n, lo, hi in zip(config.PROPERTY_NAMES, mins, maxs)},
            fh,
            indent=2,
        )

    # ---- audit -----------------------------------------------------------
    dup = out["sequence"].duplicated(keep=False)
    n_unique = int(out["sequence"].nunique())
    normed = (out[config.PROPERTY_NAMES].to_numpy(dtype=float) - mins) / (maxs - mins)

    print(f"\nwrote {pairs_path}  ({len(out)} rows, {n_unique} distinct sequences)")
    print(f"wrote {norm_path}")
    print(f"\nsequences appearing more than once: {int(dup.sum())} rows "
          f"({len(out) - n_unique} duplicate labels)")
    print(f"region split: {dict(out['region'].value_counts())}")
    print(f"type split:   {dict(out['type'].value_counts())}")
    print(f"length range: {out['sequence'].str.len().min()}-"
          f"{out['sequence'].str.len().max()}, "
          f"median {int(out['sequence'].str.len().median())}")
    print(f"normalised values within [0,1]: "
          f"{bool((normed >= 0).all() and (normed <= 1).all())}")

    path = runinfo.write_result(
        "03_dataset_build",
        {
            "n_fasta": len(fasta),
            "n_mech_rows": len(mech),
            "n_joined": len(joined),
            "n_complete": len(complete),
            "n_masp": len(masp),
            "expected_rows": EXPECTED_ROWS,
            "row_count_matches_paper": count_matches,
            "n_distinct_sequences": n_unique,
            "n_duplicate_label_rows": int(dup.sum()),
            "region_counts": {k: int(v) for k, v in out["region"].value_counts().items()},
            "type_counts": {k: int(v) for k, v in out["type"].value_counts().items()},
            "normalisation": {
                "source": "recovered_from_silkome_all_types",
                "mins": mins.tolist(),
                "maxs": maxs.tolist(),
            },
            "worked_example_verification": {
                "idv_id": WORKED_EXAMPLE_IDV,
                "verified": bool(verified),
                "max_abs_error": max_err,
                "paper_values": WORKED_EXAMPLE_NORMALISED,
            },
            "normalised_within_unit_interval": bool(
                (normed >= 0).all() and (normed <= 1).all()
            ),
        },
        script="scripts/03_build_dataset.py",
        params={"fasta": str(args.fasta), "mech": str(args.mech)},
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
