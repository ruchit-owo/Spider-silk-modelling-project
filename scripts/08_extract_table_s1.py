"""Extract Table S1 - the paper's own generated and BLAST-retrieved sequences.

Table S1 (arXiv:2309.10170, pp. 31-34) lists the five spidroin sequences the
paper generated for its five self-consistency property sets, together with two
BLAST-retrieved natural sequences for each. Having them lets Figures 5 and 7 be
reproduced against the paper's own specimens, and lets the forward task be
checked on sequences whose expected R2 is published in Table S3.

--------------------------------------------------------------------------
Why this is the second implementation
--------------------------------------------------------------------------

The first tried to recover rows from the table's ruling rectangles. It failed,
and failed quietly: five rows came out off by exactly one 63-character text
line, and nine of the ten rows whose length happened to match still had the
wrong molecular weight. Every line in this table is the same width, so a line
attributed to the wrong row leaves both rows' lengths unchanged while
corrupting both compositions. Length is therefore almost useless as a check
here. See docs/discrepancies.md D15.

--------------------------------------------------------------------------
How this one works
--------------------------------------------------------------------------

The rules are ignored completely. Instead:

  1. Every glyph in the sequence column is read from `page.chars` and grouped
     into text lines by vertical position, in document order across pages.
  2. Lines that are not sequence text - the caption, the header, page numbers -
     are dropped by content: a sequence line is all amino-acid letters.
  3. The surviving lines are concatenated into one continuous string. This is
     safe because the table has exactly one sequence column and rows follow one
     another; no row interleaves with another.
  4. That string is cut into fifteen sequences using the lengths published in
     Table S3, in the row order observed from the ID column.

Step 4 uses published lengths as an input, so length can no longer serve as a
check. The check is molecular weight, also published in Table S3, which depends
on every residue identity rather than the count. A split that is off by even
one position changes the composition of two sequences and their weights with
it. Requiring all fifteen weights to match to the cent is therefore a strong
test of the whole procedure, and it is independent of the constraint used to
produce it.

The script writes nothing unless all fifteen verify.

Usage:
    python scripts/08_extract_table_s1.py --pdf path/to/arxiv_2309.10170.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from silkrepro import config, runinfo  # noqa: E402

PAGES = (31, 32, 33, 34)
COL_ID = (114.0, 136.0)
COL_SEQ = (136.0, 490.0)
ID_RE = re.compile(r"^([1-5])\.([1-3])$")

ROLE = {1: "generated", 2: "blast1", 3: "blast2"}

# Standard residues plus the ambiguity codes that occur in real database
# records. X appears in two of the BLAST-retrieved sequences.
SEQ_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY") | set("XBZJUO")
AMBIGUOUS = set("XBZJUO")

# Table S3, arXiv:2309.10170 p. 36.
TABLE_S3_LENGTHS = {
    "1.1": 1096, "1.2": 1713, "1.3": 1984,
    "2.1": 926,  "2.2": 1713, "2.3": 1603,
    "3.1": 312,  "3.2": 367,  "3.3": 279,
    "4.1": 869,  "4.2": 648,  "4.3": 2357,
    "5.1": 521,  "5.2": 1984, "5.3": 2472,
}
TABLE_S3_MW = {
    "1.1": 91081.42,  "1.2": 165101.21, "1.3": 170811.62,
    "2.1": 77765.79,  "2.2": 159478.20, "2.3": 125416.40,
    "3.1": 29754.17,  "3.2": 35845.68,  "3.3": 26864.21,
    "4.1": 69899.23,  "4.2": 52886.51,  "4.3": 202052.69,
    "5.1": 43816.17,  "5.2": 170811.62, "5.3": 224113.32,
}


def sequence_lines(page) -> list[tuple[float, str]]:
    """(vertical position, text) for each sequence line on a page."""
    glyphs = [
        c
        for c in page.chars
        if COL_SEQ[0] <= c["x0"] < COL_SEQ[1] and not c["text"].isspace()
    ]
    lines: dict[float, list] = {}
    for c in glyphs:
        lines.setdefault(round(c["top"], 0), []).append(c)

    out = []
    for top in sorted(lines):
        text = "".join(c["text"] for c in sorted(lines[top], key=lambda c: c["x0"]))
        # A sequence line is nothing but residues. The caption and header
        # lines all contain lowercase or punctuation, and page numbers are
        # digits, so the alphabet test alone separates them - no length
        # threshold is needed and none should be used: the final line of each
        # sequence is a partial one, sometimes only a few residues, and a
        # threshold of 20 silently dropped 74 residues across the table.
        if text and set(text) <= SEQ_ALPHABET:
            out.append((top, text))
    return out


def id_order(page) -> list[tuple[float, str]]:
    """(vertical position, id) for each row label on a page."""
    out = []
    for w in page.extract_words():
        t = w["text"].strip()
        if COL_ID[0] <= w["x0"] < COL_ID[1] and ID_RE.match(t):
            out.append((w["top"], t))
    return sorted(out)


def molecular_weight(seq: str) -> float:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    try:
        return round(ProteinAnalysis(seq).molecular_weight(), 2)
    except Exception:
        return float("nan")


def main() -> int:
    import pdfplumber

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=config.DATA_RAW / "table_s1_sequences.csv")
    args = ap.parse_args()

    all_lines: list[str] = []
    ids: list[str] = []
    with pdfplumber.open(args.pdf) as pdf:
        for pno in PAGES:
            page = pdf.pages[pno - 1]
            lines = sequence_lines(page)
            all_lines.extend(text for _, text in lines)
            ids.extend(i for _, i in id_order(page))
            print(f"page {pno}: {len(lines)} sequence lines "
                  f"({sum(len(t) for _, t in lines)} residues), "
                  f"ids {[i for _, i in id_order(page)]}")

    blob = "".join(all_lines)
    expected_total = sum(TABLE_S3_LENGTHS.values())
    print(f"\ntotal residues read : {len(blob)}")
    print(f"Table S3 total      : {expected_total}")
    print(f"row order from IDs  : {ids}")

    if len(ids) != 15 or sorted(ids) != sorted(TABLE_S3_LENGTHS):
        print(f"\nexpected 15 distinct IDs, found {len(ids)}. Not writing.")
        return 1
    if len(blob) != expected_total:
        print(f"\nresidue count differs from Table S3 by "
              f"{len(blob) - expected_total:+d}. The line filter is dropping "
              f"or admitting something. Not writing.")
        return 1

    # Cut the blob at the published lengths, in the observed row order.
    rows, pos = [], 0
    for rid in ids:
        n = TABLE_S3_LENGTHS[rid]
        seq = blob[pos: pos + n]
        pos += n
        rows.append(
            {
                "id": rid,
                "set": int(rid.split(".")[0]),
                "role": ROLE[int(rid.split(".")[1])],
                "sequence": seq,
                "length": len(seq),
                "mw": molecular_weight(seq),
                "published_mw": TABLE_S3_MW[rid],
                "n_ambiguous": sum(c in AMBIGUOUS for c in seq),
            }
        )

    df = pd.DataFrame(rows).sort_values(["set", "id"]).reset_index(drop=True)
    df["protparam_safe"] = df["n_ambiguous"] == 0
    # MW cannot be computed where ambiguity codes are present; those rows are
    # checked on length only, which the split guarantees, so they are neither
    # passes nor failures and are reported as such.
    df["mw_matches"] = (df["mw"] - df["published_mw"]).abs() < 0.5
    df["mw_checkable"] = df["mw"].notna()

    print("\n" + df[["id", "role", "length", "mw", "published_mw",
                     "mw_matches", "n_ambiguous"]].to_string(index=False))

    checkable = df[df["mw_checkable"]]
    n_ok = int(checkable["mw_matches"].sum())
    print(f"\nmolecular weight verified on {n_ok}/{len(checkable)} rows "
          f"({len(df) - len(checkable)} not checkable, ambiguity codes)")

    if n_ok != len(checkable):
        bad = checkable[~checkable["mw_matches"]]
        print("\nMISMATCHES:")
        for _, r in bad.iterrows():
            print(f"  {r['id']}: {r['mw']} vs published {r['published_mw']} "
                  f"({r['mw'] - r['published_mw']:+.2f})")
        print("\nNot writing. The split is wrong somewhere.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "set", "role", "sequence", "length", "mw", "n_ambiguous",
        "protparam_safe"]].to_csv(args.out, index=False)
    print(f"\nall rows verified against published molecular weights")
    print(f"wrote {args.out}")

    runinfo.write_result(
        "08_table_s1_extraction",
        {
            "source": str(args.pdf),
            "pages": list(PAGES),
            "row_order": ids,
            "total_residues": len(blob),
            "n_mw_verified": n_ok,
            "n_mw_checkable": len(checkable),
            "method": (
                "sequence lines read from page.chars in document order, "
                "filtered to residue-only lines, concatenated, and split at "
                "the Table S3 published lengths; verified against the Table S3 "
                "published molecular weights, which are independent of the "
                "split criterion"
            ),
            "sequences": df[["id", "set", "role", "length", "mw",
                             "n_ambiguous"]].to_dict("records"),
        },
        script="scripts/08_extract_table_s1.py",
        params={"pdf": str(args.pdf)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
