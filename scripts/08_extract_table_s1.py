"""Extract Table S1 - the paper's own generated and BLAST-retrieved sequences.

Table S1 lists the five spidroin sequences the paper generated for its five
self-consistency property sets, together with two BLAST-retrieved natural
sequences for each (BLAST 1 and BLAST 2), giving fifteen sequences in total.
Having them lets Figures 5 and 7 be reproduced against the paper's own
specimens instead of against nearest-neighbour substitutes.

The table is not in the Wiley Supporting Information we could reach, but it is
in the arXiv preprint of the same work, arXiv:2309.10170, on pages 31-34.

Parsing note. The obvious approach - extract the page text and split it - does
not work here, and the failure is silent rather than loud. The ID cell is
vertically centred within its row, so in linearised text the ID appears in the
*middle* of the sequence it labels, and sequences that span a page break run
together. We therefore parse geometrically:

  - the table's rules are drawn as thin filled rectangles, not lines, so row
    bands come from the rectangles' vertical positions;
  - columns come from the rectangles' x positions:
        Type 72-114 | ID 114-136 | Sequence 136-490 | NCBI 490-544
  - a row band carrying no ID is a continuation of the previous row, which is
    how page breaks are handled.

This same class of mistake - trusting linearised text for a multi-column
layout - produced a false finding earlier in this project (see
docs/discrepancies.md D13), so the extraction is verified before use: every
sequence must be non-empty, contain only standard amino-acid letters, and the
fifteen IDs must be exactly 1.1-5.3.

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
from silkrepro.tasks import VALID_AA  # noqa: E402

PAGES = (31, 32, 33, 34)          # 1-indexed pages holding Table S1
COL_ID = (114.0, 136.0)
COL_SEQ = (136.0, 490.0)
COL_NCBI = (490.0, 560.0)
ID_RE = re.compile(r"^([1-5])\.([1-3])$")

# Sequence lengths published in Table S3 (arXiv:2309.10170 p. 36). Used as an
# independent check on this extraction: the parse is only trusted if every
# length matches. This caught a 63-residue truncation on four page-spanning
# rows that produced no error and no visible corruption.
TABLE_S3_LENGTHS = {
    "1.1": 1096, "1.2": 1713, "1.3": 1984,
    "2.1": 926,  "2.2": 1713, "2.3": 1603,
    "3.1": 312,  "3.2": 367,  "3.3": 279,
    "4.1": 869,  "4.2": 648,  "4.3": 2357,
    "5.1": 521,  "5.2": 1984, "5.3": 2472,
}

# Molecular weights from the same table. A far stronger check than length:
# length only counts characters, whereas MW depends on every residue identity,
# so an exact match to two decimals effectively rules out a mis-parse. Used to
# settle which row an ambiguous page-spanning block belongs to.
TABLE_S3_MW = {
    "1.1": 91081.42,  "1.2": 165101.21, "1.3": 170811.62,
    "2.1": 77765.79,  "2.2": 159478.20, "2.3": 125416.40,
    "3.1": 29754.17,  "3.2": 35845.68,  "3.3": 26864.21,
    "4.1": 69899.23,  "4.2": 52886.51,  "4.3": 202052.69,
    "5.1": 43816.17,  "5.2": 170811.62, "5.3": 224113.32,
}

# A block of 63 residues sits above the first rule on page 34, between rows
# 4.3 and 5.3, and the geometry alone does not say which it belongs to.
# Molecular weight settles it: attached to 5.3 the MW is 224113.32, exactly the
# published value, whereas attaching it to 4.3 leaves both rows wrong. So a
# page's leading band starts the row below it, not the row above.
LEADING_BAND_STARTS_NEXT_ROW = True

# ID convention in the table: <property set>.<role>, role 1 = generated,
# 2 = BLAST 1, 3 = BLAST 2.
ROLE = {1: "generated", 2: "blast1", 3: "blast2"}


def row_bands(page) -> list[tuple[float, float]]:
    """Vertical bands between the table's horizontal rules.

    Bands must also be opened above the first rule and below the last one.
    A row that spans a page break has text sitting outside the rules on the
    continuation page, and dropping it silently truncates that sequence -
    which it did, by exactly 63 residues on four rows, until this was fixed.
    Table S3's published sequence lengths are what caught it.
    """
    ys = sorted(
        {
            round(r["top"], 1)
            for r in page.rects
            if (r["x1"] - r["x0"]) > 200 and (r["bottom"] - r["top"]) < 3
        }
    )
    # Collapse rule pairs (each rule is a rect with a top and a bottom edge).
    merged: list[float] = []
    for y in ys:
        if not merged or y - merged[-1] > 2:
            merged.append(y)
    if not merged:
        return []
    edges = [0.0] + merged + [float(page.height)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


SEQ_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def looks_like_protein(chunk: str) -> bool:
    """Is this band's text a sequence continuation, or page furniture?"""
    return len(chunk) >= 20 and set(chunk) <= SEQ_ALPHABET


def words_in(words, x_lo, x_hi, y_lo, y_hi):
    return [
        w
        for w in words
        if x_lo <= w["x0"] < x_hi and y_lo - 1 <= w["top"] < y_hi + 1
    ]


def chars_in(page, x_lo, x_hi, y_lo, y_hi) -> str:
    """Sequence text from raw glyphs, in reading order.

    `extract_words` groups glyphs into words using spacing heuristics, and on
    these densely-set sequence blocks it loses characters: rows came out with
    the right length but the wrong molecular weight, which is only possible if
    residue identities were altered. Reading `page.chars` directly and sorting
    by (line, x) reproduces the published molecular weight exactly, so nothing
    is inferred about spacing at all.
    """
    # Half-open in y, with no tolerance. A tolerance of even one point puts a
    # glyph sitting on a rule into two bands at once, which moved whole
    # 63-character lines between adjacent rows: lengths still looked right
    # because every line is the same width, but the molecular weights did not.
    glyphs = [
        c
        for c in page.chars
        if x_lo <= c["x0"] < x_hi
        and y_lo <= c["top"] < y_hi
        and not c["text"].isspace()
    ]
    # Group into lines by rounded vertical position, then order left to right.
    lines: dict[float, list] = {}
    for c in glyphs:
        lines.setdefault(round(c["top"], 0), []).append(c)
    out = []
    for top in sorted(lines):
        out.append("".join(c["text"] for c in sorted(lines[top], key=lambda c: c["x0"])))
    return "".join(out)


def main() -> int:
    import pdfplumber

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=config.DATA_RAW / "table_s1_sequences.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    pending = ""  # a leading band held over until the next row's ID appears
    with pdfplumber.open(args.pdf) as pdf:
        for pno in PAGES:
            page = pdf.pages[pno - 1]
            words = page.extract_words()
            bands = row_bands(page)
            for band_i, (y0, y1) in enumerate(bands):
                ids = [
                    w["text"].strip()
                    for w in words_in(words, *COL_ID, y0, y1)
                    if ID_RE.match(w["text"].strip())
                ]
                # Page numbers are centred, which puts them inside the
                # sequence column's x-range; strip digits rather than let them
                # fail the protein content gate and lose the whole band.
                chunk = "".join(
                    ch for ch in chars_in(page, *COL_SEQ, y0, y1) if not ch.isdigit()
                )
                if not chunk:
                    continue
                ncbi = " ".join(
                    w["text"] for w in
                    sorted(words_in(words, *COL_NCBI, y0, y1),
                           key=lambda w: (round(w["top"], 1), w["x0"]))
                ).strip()

                if ids:
                    rows.append({"id": ids[0], "sequence": pending + chunk,
                                 "ncbi": ncbi, "page": pno})
                    pending = ""
                elif (
                    band_i == 0
                    and LEADING_BAND_STARTS_NEXT_ROW
                    and looks_like_protein(chunk)
                ):
                    # Leading band on a page, no ID: it begins the row whose ID
                    # appears in the band below. See LEADING_BAND_STARTS_NEXT_ROW.
                    pending += chunk
                elif rows and looks_like_protein(chunk):
                    # No ID in this band: continuation of the previous row,
                    # which is how a sequence spanning a page break appears.
                    # Gated on content, because the bands above the first rule
                    # and below the last one also contain the table caption and
                    # the page number, which must not be appended to a sequence.
                    rows[-1]["sequence"] += chunk
                    if ncbi and not rows[-1]["ncbi"]:
                        rows[-1]["ncbi"] = ncbi

    if not rows:
        raise SystemExit("no rows parsed; check the page numbers and column bounds")

    df = pd.DataFrame(rows)
    df["set"] = df["id"].str.split(".").str[0].astype(int)
    df["role"] = df["id"].str.split(".").str[1].astype(int).map(ROLE)
    df["length"] = df["sequence"].str.len()
    df = df.sort_values(["set", "id"]).reset_index(drop=True)

    # The accession column extracts with stray spaces inside the token; it is
    # metadata only, never used in an analysis, but tidy it so it is readable.
    df["ncbi"] = (
        df["ncbi"].str.replace(r"\s+", "", regex=True).str.replace("from", " from ")
    )

    # ---- verification, before anything downstream uses this ----------
    # Ambiguity codes are legitimate here. The BLAST-retrieved sequences are
    # real database records, and X (unknown residue) genuinely occurs in them;
    # so do B, Z, J and U occasionally. Their presence is not a parse failure,
    # but it does matter downstream because ProtParam cannot compute a
    # molecular weight for them, so we count them and carry the count.
    AMBIGUOUS = set("XBZJUO")
    problems = []
    expected = {f"{s}.{r}" for s in range(1, 6) for r in range(1, 4)}
    got = set(df["id"])
    if got != expected:
        problems.append(f"IDs are {sorted(got)}, expected {sorted(expected)}")

    df["n_ambiguous"] = df["sequence"].apply(lambda s: sum(c in AMBIGUOUS for c in s))
    df["protparam_safe"] = df["n_ambiguous"] == 0
    df["published_length"] = df["id"].map(TABLE_S3_LENGTHS)
    df["length_matches_table_s3"] = df["length"] == df["published_length"]

    mismatched = df[~df["length_matches_table_s3"]]
    for _, r in mismatched.iterrows():
        problems.append(
            f"{r['id']}: extracted {r['length']} residues, Table S3 says "
            f"{r['published_length']} (difference {r['length'] - r['published_length']:+d})"
        )

    # Molecular weight check - stronger than length, since it depends on every
    # residue identity rather than just the count.
    def _mw(seq: str) -> float:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        try:
            return round(ProteinAnalysis(seq).molecular_weight(), 2)
        except Exception:
            return float("nan")

    df["mw"] = df["sequence"].apply(_mw)
    df["published_mw"] = df["id"].map(TABLE_S3_MW)
    df["mw_matches"] = (df["mw"] - df["published_mw"]).abs() < 0.5
    for _, r in df.iterrows():
        if pd.notna(r["mw"]) and not r["mw_matches"]:
            problems.append(
                f"{r['id']}: MW {r['mw']} vs published {r['published_mw']} "
                f"(difference {r['mw'] - r['published_mw']:+.2f})"
            )

    for _, r in df.iterrows():
        bad = set(r["sequence"]) - VALID_AA - AMBIGUOUS
        if bad:
            problems.append(f"{r['id']}: unexpected characters {sorted(bad)}")
        if r["length"] < 50:
            problems.append(f"{r['id']}: implausibly short ({r['length']})")
        # More than a few percent ambiguous would suggest a parse problem
        # rather than an ordinary database record.
        if r["length"] and r["n_ambiguous"] / r["length"] > 0.05:
            problems.append(
                f"{r['id']}: {r['n_ambiguous']}/{r['length']} ambiguous residues, "
                f"too many to be an ordinary record"
            )

    print(
        df[["id", "set", "role", "length", "published_length",
            "length_matches_table_s3", "mw", "published_mw", "mw_matches",
            "n_ambiguous"]]
        .to_string(index=False)
    )

    if problems:
        print("\nEXTRACTION PROBLEMS:")
        for p in problems:
            print("  -", p)
        print("\nNot writing the file. Fix the parse before using these "
              "sequences for anything.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[
        ["id", "set", "role", "sequence", "length", "n_ambiguous",
         "protparam_safe", "ncbi"]
    ].to_csv(args.out, index=False)
    print(f"\nall {len(df)} sequences parsed and validated")
    print(f"lengths {df['length'].min()}-{df['length'].max()}, "
          f"median {int(df['length'].median())}")
    n_amb = int((~df["protparam_safe"]).sum())
    if n_amb:
        print(f"{n_amb} sequences carry ambiguity codes and are excluded from "
              f"ProtParam descriptors: "
              f"{', '.join(df.loc[~df['protparam_safe'], 'id'])}")
    print(f"wrote {args.out}")

    runinfo.write_result(
        "08_table_s1_extraction",
        {
            "source": str(args.pdf),
            "pages": list(PAGES),
            "n_sequences": len(df),
            "sequences": df[["id", "set", "role", "length", "ncbi"]].to_dict("records"),
        },
        script="scripts/08_extract_table_s1.py",
        params={"pdf": str(args.pdf)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
