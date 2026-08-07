# Results

Every number here was produced by a script in `scripts/` and is backed by a
JSON manifest in `results/` recording the git commit, model revision, seed and
package versions that produced it.

Each claim is tagged:

- **measured** — came out of a run on this machine
- **derived** — computed from measured quantities
- **assumed** — a choice we made, cross-referenced to `ASSUMPTIONS.md`

Hardware: NVIDIA RTX 4050 Laptop (6 GB), float16 inference. Checkpoint
`lamm-mit/SilkomeGPT` at revision `d5360a134ee319051fa947e53a63f6fda3488cba`.

---

## 1. The checkpoint is the model the paper describes

**measured** (`scripts/00_check_env.py`)

| | paper | checkpoint |
|---|---|---|
| layers | 12 | 12 |
| attention heads | 8 | 8 |
| hidden size | 1024 | 1024 |
| intermediate size | 4096 | 4096 |
| parameters | 253.6 M | **253,556,736** (253.56 M) |
| architecture | GPT-NeoX-style, rotary | `gpt_neox` |

Vocabulary 50,000, max position embeddings 2,048. All four stated
hyperparameters match and the parameter count agrees to the paper's rounding.

## 2. The forward task reproduces the paper's own worked example

**measured** (`scripts/01_smoke_test.py`, `scripts/01b_forward_decoding_diagnostics.py`)

The Experimental Section prints one fine-tuning record in full — a sequence and
the eight property values paired with it. Submitting that sequence to
`CalculateSilkContent` returns

```
[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]
```

which is exactly the published vector, on all eight entries.

This is the strongest single check available in the project: it exercises the
tokenizer, the prompt format, the checkpoint and the output parser at once, and
it has a known correct answer. Nothing else here does.

Note (`docs/discrepancies.md` D8): our sequence was transcribed from the PDF
and is 635 residues, a 94.9% 6-mer-containment match to Silkome record
`idv_id 7305` rather than a byte-identical match — PDF extraction dropped
characters across line wraps. The model returns the published vector anyway.

## 3. The 1,033-pair dataset, reconstructed

**measured + assumed** (`scripts/03_build_dataset.py`, assumption A1)

The paper does not give its curation rule and the pairs are not in the authors'
repository. Reconstructing from Silkome v1 primary data:

| step | rows |
|---|---|
| protein FASTA records | 11,155 |
| mechanical-property rows | 446 |
| joined on `idv_id` | 3,778 |
| with all eight properties present | 3,563 |
| `type` beginning with MaSp | **1,033** |

The paper states 1,033. Joining on `ncbi_tax_id` instead returns **zero** rows —
that field exists in both files but shares no values, so it is not the key.

Composition of the reconstructed set:

- 1,033 rows, **1,028 distinct sequences** — five sequences carry two different
  label vectors, from individuals whose fibres tested differently (A12)
- 553 CTD, 480 NTD — these are terminal-domain-anchored records, not complete
  spidroins
- types: MaSp1 349, MaSp2 331, MaSp 223, MaSp3B 47, MaSp3 43, MaSp2B 40
- lengths 115–1,854, median 378

## 4. Table S5 normalisation constants, recovered and verified

**derived, then verified against a published value** (assumption A2)

We could not obtain the Supporting Information, so the constants were recovered
rather than read. Three candidate populations were tested against the worked
example of §2, whose normalised vector the paper publishes:

| population for min/max | max abs. error vs published |
|---|---|
| raw `mechanical_properties.csv` (446 rows) | 0.139 |
| **sequence-joined, all spidroin types (3,563 rows)** | **0.00047** |
| sequence-joined, MaSp only (1,033 rows) | 0.054 |

The middle row reproduces all eight published values to within the paper's own
three-decimal rounding. This matches the Experimental Section's statement that
extremes were taken "across the entire dataset (not limited to MaSp
sequences)".

This is a verification, not a fit: the constants come from the data, and were
then checked against a number that played no part in deriving them.

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

`scripts/03_build_dataset.py` re-runs this check on every invocation and exits
non-zero if it fails.

## 5. Forward-task decoding

**measured** (`scripts/01b_forward_decoding_diagnostics.py`, 60 real silk
sequences, assumption A5)

| token budget | parse rate (greedy) |
|---|---|
| 32 | 0/60 |
| 64 | **60/60** |
| 96–192 | 60/60 |

Greedy vs the notebook's sampled setting (T=0.01, top_k 500, top_p 0.9):

- identical vectors on **58/60** (96.7%)
- differing on 2/60, by at most 0.38 in one slot
- neither failed to parse

Batch-size invariance under greedy: **16/16** identical between batch sizes 1
and 8. Under sampling it is *not* invariant, because all rows of a batch draw
from one RNG stream — which is why we decode greedily, since we batch the
forward pass for speed.

An incidental finding: under sampling, the model sometimes continues the
amino-acid sequence before emitting the property vector, and can then be cut
off mid-answer by the 64-token budget. The released notebook catches that in a
bare `except:`, so those cases silently vanish rather than being counted. Under
greedy decoding it never happens.

## 6. Extension — sequence-content ablations

**measured** (`scripts/11_ablation_sequence_content.py`, 30 MaSp probe
sequences, 8 repeats each for the noise floor)

**Noise floor: zero.** Under greedy decoding the forward task returned one
distinct output across 8 repeats of every probe. Every effect below is
therefore above the floor by construction.

**Reference scale: 0.0874.** This is the mean absolute deviation of predictions
across the 30 unmodified probes — how much the model's output moves between
different natural silk sequences. Effects are given as a fraction of it, since
an absolute shift of 0.13 means nothing without knowing the range.

| ablation | mean \|Δ\| | × reference scale |
|---|---|---|
| reverse | 0.1322 | 1.51 |
| core only (termini removed) | 0.1268 | 1.45 |
| shuffle | 0.1253 | 1.43 |
| block shuffle | 0.1245 | 1.43 |
| scattered control (polyA) | 0.1217 | 1.39 |
| **random sequence, matched length** | **0.1211** | **1.39** |
| scattered control (GGX) | 0.1129 | 1.29 |
| knockout GPGXX | 0.0954 | 1.09 |
| scattered control (GPGXX) | 0.0855 | 0.98 |
| knockout polyA | 0.0809 | 0.93 |
| termini only (core removed) | 0.0773 | 0.89 |
| substitute GGX | 0.0761 | 0.87 |
| block control (polyA) | 0.0731 | 0.84 |
| block control (GPGXX) | 0.0720 | 0.82 |
| knockout GGX | 0.0715 | 0.82 |
| substitute GPGXX | 0.0690 | 0.79 |
| substitute polyA | 0.0614 | 0.70 |
| block control (GGX) | 0.0401 | 0.46 |

Two findings.

**(a) No motif-specific effect.** Every knockout lands *between* its two
controls:

| motif | knockout | scattered control | block control |
|---|---|---|---|
| polyA | 0.0809 | 0.1217 | 0.0731 |
| GGX | 0.0715 | 0.1129 | 0.0401 |
| GPGXX | 0.0954 | 0.0855 | 0.0720 |

A motif knockout removes several contiguous runs, so sitting between a
scattered removal and a single block removal of the same size is exactly what
removing that many residues *without* any motif-specific sensitivity predicts.
Removing poly-A tracts — the β-sheet nanocrystal formers, the most
mechanically consequential motif in spider silk — perturbs the prediction
*less* than deleting the same number of residues at random positions. We find
no evidence that the forward task treats these motifs as special.

**(b) Destroying the sequence entirely is not much worse than perturbing it.**
Replacing a spidroin with uniform random residues of the same length
(`random_matched`, 0.1211) moves the prediction about as far as shuffling it
(0.1253) or reversing it (0.1322) — and all three sit near 1.4× the spread
observed across genuinely different natural sequences. The forward task's
output range appears to saturate: past a certain amount of disruption, more
disruption does not move it further.

**Interpretation limits.** A shuffled or random sequence is far outside the
fine-tuning distribution, so a large shift shows the model is sensitive to
order without showing it uses order meaningfully. These measure the model, not
spider silk, and no claim about real mechanical behaviour follows from them.

## 7. Extension — composition-only baseline

**measured** (`scripts/12_baseline_composition.py`, 1,033 pairs)

Mean R² over the four mechanical properties (SD slots excluded, since they vary
little and inflate the average):

| features | model | in-sample | 5-fold CV |
|---|---|---|---|
| composition | ridge | 0.026 | 0.001 |
| composition | GBM | **0.799** | **−0.155** |
| composition + motifs | ridge | 0.025 | 0.004 |
| composition + motifs | GBM | 0.798 | −0.155 |
| dipeptide | ridge | 0.100 | 0.014 |
| dipeptide | GBM | 0.870 | −0.130 |

Species-grouped CV (ridge on composition), so no species appears in both train
and test: mean R² over mechanical properties **−0.049**.

The gradient-boosted models fit the training set well (0.80–0.87) and
generalise worse than predicting the mean. The sequence→property mapping is
genuinely hard from composition, and hand-made features do not solve it.

This is a control, not a competitor: the baseline cannot generate sequences,
which is the paper's actual contribution. What it establishes is a floor, and a
warning — on this dataset, in-sample accuracy and generalisation are close to
unrelated. Since the paper fine-tunes on all 1,033 pairs and holds nothing out
(`docs/discrepancies.md` D5), that warning applies to any forward-task accuracy
measured on them, including ours.

## 8. Extension — best-of-N selection

**pending.** The 2,048-candidate × 8-set run is in progress; 3 of 8 property
sets complete at the time of writing. This section will carry the best-of-N
curves, the single-sample distribution, and the number of samples needed to
reach each published value.

Early signal from the completed pilot (set S1, N=64): pool maximum 0.712 against
a paper value of 0.8899, with mean −8.15 and median −1.45 over 20 scoreable
candidates. Single samples are mostly far below the reported figure.

## 9. Not reproduced

| item | reason |
|---|---|
| Figure 7 (motif analysis) | needs Table S4's motif definitions, in the Supporting Information we could not obtain (A8, D7). The fallback analysis uses canonical spidroin motifs and is a different analysis, labelled as such |
| Figure 6 (AlphaFold2 structures) | not attempted; needs AF2 compute, and the paper limits the claim to visual comparison at pLDDT 40–60 |
| Section 2.2 BLAST novelty | needs Table S2, or network BLAST. Substituted with offline k-mer measures, never described as alignment statistics (A4) |
| Figure 5 against the paper's sequences | needs Table S1. Reproduced instead on our own generations vs their nearest silkome neighbours, stated wherever used |
