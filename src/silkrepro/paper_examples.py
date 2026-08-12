"""Worked examples printed in the paper, transcribed verbatim.

The Experimental Section shows one fine-tuning record in full: a sequence, and
the eight property values paired with it. Because the released checkpoint was
fine-tuned on that record, submitting the sequence to the forward task should
return that vector. That makes it the sharpest available check that we have
the right checkpoint and the right prompt format - it tests the whole path at
once, and it has a known correct answer, which nothing else in this project
does.

Transcribed from the paper's "CalculateSilkContent" and "GenerateSilkContent"
examples. The sequence is printed across several lines in the PDF with no
hyphenation, so the concatenation below is unambiguous.
"""

PAPER_EXAMPLE_SEQUENCE = (
    "AAAGGAGQGGYGGQGAGQGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGSGQGG"
    "YGGQGAGAAAAAAAAGGAGQGGYGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQGAGQGSG"
    "AAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGGQGAGQGAAAAAAGGAGQGGYGGQ"
    "GAGQGAGAAAAAAGGAGQGGYGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQGAGAAAAAA"
    "GGAGQGGYGGQGAGQGAAAAAAGGAGQGGYGGQGAGQGGYGGQGAGAAAAAAGGAGQGGYGGLGGQ"
    "GAGQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGLGGQGAGAAAAAAGGAGQG"
    "GYGGQGAGQGGYGGQGSGAAAAAAAAGGAGQGGYGGLGSQGAGQGAGAAAAAAGGAGQGGYGGQGA"
    "GQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGGYGGQGAGQGAGAAAAAAGGAGQGG"
    "YGGLGSGQGGYGGQGAGAAAAAAGGAGQGGYGGQGAGAAAASAAASRLSSPEASSGLSGCDVLVQA"
    "LLEVVSALIHILGSSSIGPVNYGSASQSTQIVGQSVYQALG"
)

# The eight values the paper pairs with the sequence above, in the
# config.PROPERTY_NAMES order.
PAPER_EXAMPLE_PROPERTIES = [0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]

# --------------------------------------------------------------------------
# The two printed variants
# --------------------------------------------------------------------------
#
# The paper prints two example sequences against this one property vector: one
# for the forward "CalculateSilkContent" task (above, 635 residues) and one for
# the inverse "GenerateSilkContent" task, which is longer.
#
# The forward example is not a database record. Aligned against silkome record
# idv_id 7305 (Nephilingis livida, MaSp1, CTD, 671 residues), which IS in the
# reconstructed 1,033-pair dataset, the 635-residue forward example matches on
# all 635 of its characters, with a single contiguous 36-residue deletion at
# record positions 583-619:
#
#     RVSSAVSNLVSSGPTNSAALSNTISSVVSQISASNP
#
# So the transcription here is byte-exact and the difference is in the paper,
# which prints two variants of one record. Most likely a slip while preparing
# the forward example. Stated neutrally; nothing depends on which it is.
#
# Both variants return the published property vector from the forward task, so
# both work as known-answer tests. See docs/discrepancies.md D8.

WORKED_EXAMPLE_IDV_ID = 7305
WORKED_EXAMPLE_RECORD_LENGTH = 671
WORKED_EXAMPLE_DELETED_BLOCK = "RVSSAVSNLVSSGPTNSAALSNTISSVVSQISASNP"


def silkome_record_variant() -> str:
    """The full 671-residue silkome record the forward example derives from.

    Loaded from the reconstructed dataset rather than pasted here, so it stays
    tied to the data actually in use. Raises if the dataset is absent.
    """
    from . import dataio

    df = dataio.load_pairs()
    hits = [
        s
        for s in df.loc[df["idv_id"] == WORKED_EXAMPLE_IDV_ID, "sequence"].astype(str)
        if len(s) == WORKED_EXAMPLE_RECORD_LENGTH
    ]
    if not hits:
        raise LookupError(
            f"no {WORKED_EXAMPLE_RECORD_LENGTH}-residue record for idv_id "
            f"{WORKED_EXAMPLE_IDV_ID} in the reconstructed dataset"
        )
    return hits[0]
