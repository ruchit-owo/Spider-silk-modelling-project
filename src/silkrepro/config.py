"""Constants taken from the paper, the released notebook, or chosen by us.

Everything in this file is tagged with where it came from. Three tags are used
throughout the project and they mean different things:

    PAPER     - stated explicitly in Lu, Kaplan & Buehler (2024), main text.
    CODE      - not in the paper; read off the authors' released notebook
                (SilkomeGPT_inference.ipynb).
    OURS      - neither; a choice we made. Every OURS entry has a matching row
                in ASSUMPTIONS.md giving the reasoning.

Reference: W. Lu, D. L. Kaplan, M. J. Buehler, Adv. Funct. Mater. 2024, 34,
2311324. DOI 10.1002/adfm.202311324
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# The authors' repo, unpacked. Contains ALL_SILK_SEQ.csv, which is the novelty
# reference set (sequences only, no properties).
REF_REPO = ROOT / "_ref" / "SilkomeGPT-main"
ALL_SILK_SEQ_CSV = REF_REPO / "ALL_SILK_SEQ.csv"

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

# PAPER, Experimental Section: the released fine-tuned checkpoint.
MODEL_NAME = "lamm-mit/SilkomeGPT"

# PAPER: 12 layers, 8 attention heads, hidden 1024, intermediate 4096,
# 253.6M parameters, rotary positional embedding, GPT-NeoX-inspired.
# scripts/01_smoke_test_model.py checks the loaded checkpoint against these.
EXPECTED_ARCH = {
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "hidden_size": 1024,
    "intermediate_size": 4096,
}
EXPECTED_PARAMS_M = 253.6

# --------------------------------------------------------------------------
# The 8D property vector
# --------------------------------------------------------------------------

# PAPER, Eq. (1). Order matters: the model was fine-tuned on this ordering and
# nothing in the prompt tells it which slot is which.
PROPERTY_NAMES = [
    "toughness",
    "toughness_sd",
    "E",
    "E_sd",
    "strength",
    "strength_sd",
    "strain",
    "strain_sd",
]

# Indices of the four mechanical properties, as opposed to their SDs. Several
# of our analyses treat these separately, because a model that predicts the SD
# slots well but the property slots badly would still score a respectable R2
# over all eight.
MECHANICAL_IDX = [0, 2, 4, 6]
SD_IDX = [1, 3, 5, 7]

# --------------------------------------------------------------------------
# Target property sets (PAPER, Table 1)
# --------------------------------------------------------------------------

# Three sets used for the forward-prediction assessment (Figure 3). The paper
# describes these as "randomly selected to encompass a range of property
# values".
FORWARD_SETS = {
    "F1": [0.600, 0.204, 0.279, 0.043, 0.329, 0.127, 0.502, 0.139],
    "F2": [0.300, 0.200, 0.200, 0.100, 0.300, 0.100, 0.200, 0.100],
    "F3": [0.100, 0.200, 0.200, 0.040, 0.300, 0.100, 0.700, 0.100],
}

# R2 reported by the paper for each of the above (Table 1, Figure 3).
FORWARD_R2_PAPER = {"F1": 0.8764, "F2": 0.8369, "F3": 0.6889}

# Five sets used for the self-consistency analysis (Figures 4-7).
SELFCONSISTENCY_SETS = {
    "S1": [0.250, 0.252, 0.200, 0.151, 0.310, 0.203, 0.293, 0.265],
    "S2": [0.550, 0.252, 0.750, 0.151, 0.310, 0.203, 0.293, 0.265],
    "S3": [1.000, 0.252, 0.450, 0.151, 0.200, 0.203, 0.293, 0.265],
    "S4": [0.900, 0.252, 0.800, 0.151, 0.310, 0.203, 0.293, 0.265],
    "S5": [0.240, 0.252, 0.206, 0.151, 0.270, 0.203, 0.293, 0.265],
}

SELFCONSISTENCY_R2_PAPER = {
    "S1": 0.8899,
    "S2": 0.5640,
    "S3": 0.7167,
    "S4": 0.7843,
    "S5": 0.7751,
}

# How the paper characterises each self-consistency set (Section 2, Figure 2).
# Carried here so figures and tables can be labelled without re-reading the text.
SELFCONSISTENCY_RATIONALE = {
    "S1": "common (E, toughness) pair = (0.20, 0.25)",
    "S2": "rare but existing pair, high E = (0.75, 0.55)",
    "S3": "rare but existing pair, high toughness = (0.45, 1.00)",
    "S4": "non-existent pair, both elevated = (0.80, 0.90)",
    "S5": "common (strength, toughness) pair = (0.27, 0.24)",
}

ALL_SETS = {**FORWARD_SETS, **SELFCONSISTENCY_SETS}
ALL_R2_PAPER = {**FORWARD_R2_PAPER, **SELFCONSISTENCY_R2_PAPER}

# --------------------------------------------------------------------------
# Sampling hyperparameters
# --------------------------------------------------------------------------

# CODE. The paper states only that results "with the highest R2 values are
# selected from a set of sampling attempts" and never gives the sampling
# settings or the size of that set. These are the values hard-coded in
# generate_new_and_find_best() and generate_new() in the released notebook.
GEN_TEMPERATURE = 1.25
GEN_TOP_K = 500
GEN_TOP_P = 0.8
GEN_MAX_NEW_TOKENS = 512

# CODE. Forward pass ("CalculateSilkContent") settings, from generate_new().
FWD_TEMPERATURE = 0.01
FWD_TOP_K = 500
FWD_TOP_P = 0.9
FWD_MAX_NEW_TOKENS = 64

# OURS (assumption A5). We decode the forward task greedily rather than
# sampling at temperature 0.01. Measured justification, from
# scripts/01b_forward_decoding_diagnostics.py over 60 real silk sequences:
#
#   - greedy and sampled agree exactly on 58/60 sequences (96.7%);
#   - greedy parses 60/60 at the notebook's 64-token budget, and never emits
#     stray residues before the answer, whereas sampling sometimes continues
#     the amino-acid sequence first and can then be cut off mid-answer;
#   - greedy is invariant to batch composition (verified 16/16), which
#     sampling is not, because all rows of a batch share one RNG stream.
#
# The last point matters for us and not for the authors: we batch the forward
# task for speed, and under sampling a candidate's predicted properties would
# depend on which other candidates happened to share its batch. Table 1 is
# additionally re-run under the notebook's sampled setting as a sensitivity
# check (scripts/02_reproduce_table1.py --faithful-sampling).
FWD_DO_SAMPLE = False

# CODE. Budget in the released notebook: 32 samples per step x 64 repeats.
# We treat N as a variable rather than a constant - quantifying its effect is
# the point of the best-of-N ablation - but this is the reference budget.
PAPER_BUDGET_PER_STEP = 32
PAPER_BUDGET_REPEATS = 64
PAPER_BUDGET_TOTAL = PAPER_BUDGET_PER_STEP * PAPER_BUDGET_REPEATS  # 2048

# OURS. Default seed for every script. Each script also writes the seed it
# actually used into its output manifest.
SEED = 20260807

# --------------------------------------------------------------------------
# Task prompt formats
# --------------------------------------------------------------------------

# PAPER, Experimental Section, and confirmed against the released notebook.
# The forward task puts the sequence in <> and the model emits [] values;
# the inverse task puts the values in <> and the model emits [] sequence.
FORWARD_TASK = "CalculateSilkContent"
INVERSE_TASK = "GenerateSilkContent"

# CODE. Present in the released notebook and in the checkpoint's task
# vocabulary, but not described anywhere in the paper. We record predictions
# from it but make no claims about it.
SOLUBILITY_TASK = "CalculateSolubility"
