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

# The paper's "GenerateSilkContent" example uses the same property vector and
# shows a sequence that differs from the one above in its C-terminal region.
# We record it for completeness but make no claim about it: the paper does not
# say whether it is a training record or a generation.
PAPER_INVERSE_EXAMPLE_DIFFERS_IN_C_TERMINUS = True
