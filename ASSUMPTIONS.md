# Assumptions

Every choice this project made that the paper does not state. Each row gives
the assumption, why it was necessary, what we chose, the reasoning, and - where
possible - the evidence that the choice is right.

The rule followed throughout: where the paper does not specify something, say
so explicitly rather than quietly picking a value. Nothing here was tuned to
make a result match the paper.

Tags used in the code: `PAPER` (stated in the paper), `CODE` (from the authors'
released notebook, not the paper), `OURS` (our choice, listed here).

---

## A1 — Reconstruction rule for the 1,033-pair dataset

**The paper does not specify this.** It says the dataset was "constructed and
curated based on the silkome dataset" and contains 1,033 pairs, but gives no
curation rule, and the pairs are not in the authors' GitHub repository.

**Assumed rule.** Parse `spider-silkome-database.v1.prot.fasta`; inner-join to
`mechanical_properties.csv` on `idv_id` (the individual spider whose fibre was
tested, *not* `ncbi_tax_id`); keep rows with all eight mechanical values
present; keep rows whose `type` field begins with `MaSp`.

**Reasoning and evidence.** Joining on `ncbi_tax_id` yields zero rows, so it is
not the key. Joining on `idv_id` gives 3,778 rows, 3,563 with complete
properties. Restricting to MaSp types gives **exactly 1,033**, matching the
paper's stated count. Reaching the published number on the nose is strong
evidence, but it remains a reconstruction: a different rule could in principle
reach the same count. Implemented in `scripts/03_build_dataset.py`.

## A2 — Normalisation constants (the paper's Table S5)

**We could not obtain the Supporting Information**, so Table S5 was recovered
rather than read.

**Assumed.** Min and max per property taken over the sequence-joined table
across *all* spidroin types (3,563 rows), not over the raw 446-row
`mechanical_properties.csv` and not over the MaSp subset.

**Reasoning and evidence.** The Experimental Section says the extremes were
identified "across the entire dataset (not limited to MaSp sequences)". Three
candidate populations were tested against the worked example printed in the
Experimental Section, which pairs a sequence with the normalised vector
`[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]`:

| population | max abs. error vs published |
|---|---|
| raw `mechanical_properties.csv` (446 rows) | 0.139 |
| **sequence-joined, all types (3,563 rows)** | **0.00047** |
| sequence-joined, MaSp only (1,033 rows) | 0.054 |

The middle row reproduces all eight published values to within the paper's
own rounding. This is a verification, not a fit: the constants were derived
from the data and then checked against a number we did not use in deriving
them. `scripts/03_build_dataset.py` re-runs the check on every invocation and
exits non-zero if it fails.

Recovered constants (raw units as published in Silkome):

| property | min | max |
|---|---|---|
| toughness | 0.005 | 0.39 |
| toughness SD | 0.001 | 0.136 |
| E | 0.38 | 37.0 |
| E SD | 0.03 | 9.76 |
| strength | 0.17 | 3.33 |
| strength SD | 0.01 | 0.80 |
| strain | 5.1 | 53.2 |
| strain SD | 0.1 | 13.7 |

## A3 — Validity filter on generated sequences

**The paper does not specify this.** It does not say whether generated strings
were checked for being proteins at all.

**Assumed.** A generation counts as a sequence only if the first bracketed
group is non-empty and contains only the twenty standard amino-acid letters.

**Reasoning.** A string containing digits or brackets is not a protein and
would corrupt the ProtParam and motif analyses downstream. The released
notebook has no such check; it wraps parsing in a bare `except:` so malformed
output silently disappears. We reject explicitly and **report the rejection
rate** in every run manifest, so the effective sampling budget is visible
rather than assumed.

## A4 — Definition of novelty

**Followed the paper/notebook, and supplemented.** The released `is_novel` is
exact string membership in `ALL_SILK_SEQ.csv`. We keep that verbatim, because
reproducing the paper means reproducing its definition.

**Added, clearly labelled as ours.** Maximum 6-mer Jaccard similarity and
containment against the same reference set. Exact-match novelty is close to
free - one substituted residue in a 400-residue sequence passes it - so a
softer measure is needed to say anything about how novel a sequence is. These
are *not* BLAST and are never presented as BLAST; they are order-insensitive
and will not equal percent identity. Their merit is that they run offline and
are deterministic.

## A5 — Greedy decoding for the forward task

**Deviation from the released notebook, with measured justification.** The
notebook decodes `CalculateSilkContent` with `do_sample=True` at temperature
0.01, top_k 500, top_p 0.9, `max_new_tokens=64`. We decode greedily, keeping
the 64-token budget.

**Reasoning and evidence** (`scripts/01b_forward_decoding_diagnostics.py`, 60
real silk sequences):

- greedy and sampled produce **identical** vectors on 58/60 sequences (96.7%);
  the two that differ do so by at most 0.38 in one slot;
- greedy parses **60/60** at a 64-token budget and never emits stray residues
  before the answer. Sampling sometimes continues the amino-acid sequence
  first, and can then be truncated mid-answer, which the notebook counts as
  nothing;
- greedy is invariant to batch composition (verified 16/16 between batch sizes
  1 and 8); sampling is not, because all rows of a batch draw from one RNG
  stream.

The last point is ours to solve and not the authors': we batch the forward
task for speed, and under sampling a candidate's predicted properties would
depend on which other candidates happened to share its batch. Table 1 is
additionally re-run with the notebook's sampled setting
(`--faithful-sampling`) as a sensitivity check.

## A6 — Inference precision

**The paper does not specify this.** We use float16 on GPU (float32 on CPU).
The checkpoint's own weights are the reference; float16 is standard for
inference at this scale and halves the memory, which matters on the 6 GB card
this was run on. The forward task's determinism under greedy decoding was
verified at this precision, so precision-induced tie-breaking is not silently
changing results.

## A7 — Terminal-domain boundaries for the domain ablation

**The paper does not specify this**, and does not perform this ablation; it is
ours. We take the N-terminal domain as the first 130 residues and the
C-terminal domain as the last 100, nominal values from the spidroin
literature.

**Reasoning.** No domain annotator is run in this project. The two operators
`keep_termini` and `keep_core` use the same cut points and partition the
sequence exactly, so the comparison between them is internally consistent even
though the boundary is coarse. Conclusions are correspondingly coarse: this
ablation can say "the model reads the terminal region more than the core" or
the reverse, not where the domain ends.

## A8 — Motif set when Table S4 is unavailable

**We could not obtain the Supporting Information.** Figure 7's motifs come
from Table S4, itself derived from Table 1 of the Silkome paper.

**Assumed.** If `data/raw/table_s4_motifs.csv` is absent, the motif analysis
falls back to canonical spidroin motifs (poly-A, GGX, GPGXX, GA, QQ) and every
output is labelled `motif_source = canonical_fallback`.

**This is not a reproduction of Figure 7** and is never described as one. The
fallback exists so the ablation work, which only needs motifs that are real
and frequent, can proceed.

## A9 — Sampling budget N

**The paper does not specify this.** It says results "with the highest R2
values are selected from a set of sampling attempts" without giving the size of
the set. The released notebook uses 32 samples per step x 64 repeats = 2,048.

**Assumed.** We treat N as a variable rather than a constant, generate 2,048
candidates per property set to match the notebook, and report the entire
best-of-N curve. Quantifying the effect of N is one of this project's
extensions rather than a nuisance parameter.

## A10 — Batched forward inference

**Ours, for speed.** The forward pass dominates runtime. Batching requires
left-padding, which is only safe if padding does not perturb the result.

**Evidence.** `tests/test_batching_equivalence.py` checks batched against
unbatched predictions on real sequences of deliberately unequal length, and
checks that batch size does not change results. It is marked `slow` and should
be re-run after any transformers or torch upgrade.

## A11 — Checkpoint revision

**Ours.** `scripts/00_check_env.py` resolves and records the Hugging Face
commit of `lamm-mit/SilkomeGPT` into `results/model_revision.txt`. Set
`SILKOME_REVISION` to pin it. Without pinning, a future re-run could silently
use different weights.

## A12 — Keeping duplicate sequence labels

**Ours.** The reconstructed dataset has 1,033 rows but 1,028 distinct
sequences: five sequences appear twice with different property values, because
the same protein was recovered from two individuals whose fibres tested
differently.

**Assumed.** Keep both, since the paper's count of 1,033 requires it. The
consequence - the dataset contains contradictory labels for five inputs - is
reported rather than removed.

## A13 — Amino-acid composition accessor

**Ours, forced by an upstream change.** Biopython's
`get_amino_acids_percent()` (<= 1.81) returned fractions in 0-1; the
`amino_acids_percent` property (>= 1.82) returns percentages in 0-100. We
count residues directly instead of calling either, so no downstream number
depends on which Biopython is installed.

## A14 — Secondary-structure convention

See `docs/discrepancies.md` entry **D1**, which this assumption exists to
handle. Both the paper's caption convention and Biopython's corrected one are
computed and reported side by side; neither is silently substituted for the
other.
