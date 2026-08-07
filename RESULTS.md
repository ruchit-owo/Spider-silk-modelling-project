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

## 8. Table 1 — partial reproduction

**measured** (`scripts/02_reproduce_table1.py`, 2,048 candidates per set,
8 sets, ~4.5 h)

| set | paper R² | our max | our mean | our median | scoreable pool |
|---|---|---|---|---|---|
| F1 | 0.8764 | 0.7221 | −0.36 | −0.03 | 815 |
| F2 | 0.8369 | 0.6964 | −3.57 | −1.41 | 739 |
| F3 | 0.6889 | **0.6780** | −0.36 | +0.09 | 735 |
| S1 | 0.8899 | 0.7120 | −6.54 | −2.25 | 712 |
| S2 | 0.5640 | 0.2234 | −0.71 | −0.67 | 738 |
| S3 | 0.7167 | 0.6494 | −0.37 | −0.31 | 752 |
| S4 | 0.7843 | 0.3487 | −0.49 | −0.48 | 710 |
| S5 | 0.7751 | 0.6233 | −11.59 | −3.61 | 641 |

**We do not reach the paper's value in any of the eight sets.** The gap ranges
from 0.011 (F3, essentially reproduced) to 0.436 (S4). Mean gap 0.27. In no set
did any of ~700–800 candidates match or exceed the published figure.

The direction and the mechanism reproduce; the magnitude does not. That is a
partial reproduction, and it is stated as partial.

## 8a. Best-of-N — where the reported number comes from

**derived** (`scripts/10_ablation_bestofn.py`)

Expected best R² as a function of the sampling budget (bootstrap over the pool):

| set | N=1 | N=8 | N=64 | N=512 |
|---|---|---|---|---|
| F1 | −0.36 | 0.34 | 0.56 | 0.68 |
| F2 | −3.59 | 0.41 | 0.59 | 0.67 |
| F3 | −0.37 | 0.36 | 0.54 | 0.67 |
| S1 | −6.20 | −0.06 | 0.55 | 0.71 |
| S2 | −0.71 | −0.09 | 0.10 | 0.20 |
| S3 | −0.37 | 0.13 | 0.41 | 0.54 |
| S4 | −0.49 | 0.02 | 0.28 | 0.35 |
| S5 | −11.03 | −0.46 | 0.33 | 0.57 |

A single sample from the inverse task is, typically, worse than useless — the
median candidate has negative R² in 6 of 8 sets. Essentially all of the
apparent performance is produced by drawing many candidates and keeping the
best. This is a legitimate design procedure; the point is that the reported
number is a property of the search budget as much as of the model, and the
budget is not stated (`docs/discrepancies.md` D3).

## 8b. Why our maximum is lower — one hypothesis tested and rejected

**measured** (`scripts/10b_scoring_stochasticity.py`)

The notebook scores candidates with a *sampled* forward pass, so its R² values
are themselves noisy, and a maximum over noisy scores is inflated. We decode
greedily (A5). Could that alone explain the gap?

Re-scoring the *same* candidate sequences with the notebook's sampled decoder:

| set | max, greedy | max, sampled | inflation | gap to paper remaining |
|---|---|---|---|---|
| S1 | 0.7120 | 0.7379 | +0.026 | 0.152 |
| F1 | 0.7221 | 0.7221 | 0.000 | 0.154 |
| S4 | 0.3487 | 0.3487 | 0.000 | 0.436 |
| S2 | 0.2234 | 0.2234 | 0.000 | 0.341 |

Mean inflation **+0.0065**; scores identical on 97–99% of candidates. **The
hypothesis is rejected** — our decoding choice is not responsible for the
shortfall.

What remains as candidate explanations, none of which we can distinguish from
the information available: a larger effective candidate pool in the original
run, different generation seeds, or additional selection steps not described in
the paper or the notebook. We record the gap as unexplained rather than
attributing it.

## 8c. Most generations are verbatim training sequences

**measured** (`scripts/07_novelty_audit.py`)

| set | parsed | verbatim copies | copy rate | novel |
|---|---|---|---|---|
| F1 | 1946 | 1122 | 57.7% | 824 |
| F2 | 1961 | 1219 | 62.2% | 742 |
| F3 | 1949 | 1208 | 62.0% | 741 |
| S1 | 1953 | 1237 | 63.3% | 716 |
| S2 | 1948 | 1206 | 61.9% | 742 |
| S3 | 1955 | 1194 | 61.1% | 761 |
| S4 | 1958 | 1243 | 63.5% | 715 |
| S5 | 1974 | 1324 | 67.1% | 650 |

**Overall 62.3%.** A nominal budget of 2,048 yields about 771 scoreable
candidates. The copies are not concentrated on a few sequences — 200–290
distinct known sequences are reproduced per set, the most frequent accounting
for under 6%.

Among sequences that *do* pass the exact-match test, mean best 6-mer
containment against the reference set is **0.746**. That is compatible with
novel arrangements of familiar parts, which is what one expects of a repetitive
protein family; it is not evidence of copying, and it is not percent identity
(A4).

## 8d. The forward task largely reproduces memorised labels

**measured** (`scripts/04_forward_eval_dataset.py`,
`scripts/04b_memorisation_audit.py`)

Over the 1,033 fine-tuning pairs (1,026 parsed, 99.3%), in-sample R²:

| property | R², all rows | R², rows whose label was *not* reproduced exactly |
|---|---|---|
| toughness | +0.807 | +0.342 |
| E | +0.578 | −0.596 |
| strength | +0.721 | +0.007 |
| strain | +0.681 | −0.204 |
| **mean (mechanical)** | **+0.696** | **−0.113** |

**732 of 1,026 rows (71.3%)** have all eight values reproduced to within
0.0005 — that is, the model returns the training label exactly. Mean absolute
error on those rows is 0.00025; on the remaining 294 rows it is 0.109.

So the aggregate in-sample R² of 0.696 rests mostly on exact label retrieval.
On rows where retrieval did not occur, the model is worse than predicting the
dataset mean.

**What this does not show.** The non-memorised rows are selected on the outcome
and are not a fair held-out sample — their median length is 456 against 361 for
the memorised rows, so they differ systematically. Their R² is a diagnostic,
not a generalisation estimate. Obtaining one would require retraining with a
held-out fold, which this project does not do.

**What it is not.** This contradicts no claim in the paper. The paper fine-tunes
on all known pairs, states so, reports self-consistency rather than held-out
accuracy, and does not claim otherwise. It does change how the forward task's
apparent accuracy should be read.

## 8e. Extension — out-of-distribution spidroins

**measured** (`scripts/13_extension_nonmasp.py`, ~60 sequences per family)

| family | in training distribution | mean R² (mechanical) | mean abs error |
|---|---|---|---|
| MaSp | **yes** | **+0.826** | 0.024 |
| MiSp | no | −0.270 | 0.141 |
| AcSp | no | −0.317 | 0.170 |
| AgSp | no | −0.430 | 0.158 |
| CySp | no | −0.472 | 0.151 |
| Flag | no | −0.514 | 0.161 |
| PySp | no | −0.540 | 0.152 |

Accuracy falls off a cliff outside MaSp — from +0.83 to between −0.27 and
−0.54, with error rising six-fold.

And the predictions barely move with family at all: the between-family spread
of the mean prediction is **0.019** against a within-family spread of **0.117**,
a ratio of **0.162**. Given a spidroin it has not memorised, the model emits
something close to its training marginal regardless of which silk family the
sequence came from.

*Caveat, stated before the numbers are used:* Silkome's mechanical properties
are measured on dragline fibre, which is MaSp. Non-MaSp sequences here are
paired with the same individual's dragline properties, not with their own silk
type's. So this is not a test of whether the model can predict flagelliform
silk — no analysis of this dataset could be. The between/within ratio needs no
such caveat.

## 8f. Composition differences between high- and low-performing silks

**measured** (`scripts/14_composition_association.py`)

A model-free companion to the ablations, following Supplementary Note 8 of
Pandey, Chen & Keten (*Commun. Mater.* 2024) — **not** Figure 7 of Lu et al.,
which is a different analysis (see §9).

Ranking the 175 MaSp1-bearing individuals by each mechanical property and
comparing top-10 against bottom-10 composition, then normalising within
residue group:

| | residues selected in ≥2 properties |
|---|---|
| scheme A (Lehninger-style) | A D E G H I K L M N Q R S T V Y |
| scheme B (alternative) | A D E F G I K L M N Q R S T V |
| **agreed by both** | **A D E G I K L M N Q R S T V** |
| Pandey/Chen/Keten reported | D E F I K L N P Q R S T V Y |

Both schemes recover **12 of their 14** residues. The two schemes disagree only
on H, Y and F — precisely the residues whose classification is contested.

The grouping matters because the normalisation is within-group, and **the source
does not specify it** (assumption A15). We report both schemes rather than
choosing the one that agrees better.

## 9. Not reproduced

| item | reason |
|---|---|
| Figure 7 (motif analysis) | needs Table S4's motif definitions, in the Supporting Information we could not obtain (A8, D7). The fallback analysis uses canonical spidroin motifs and is a different analysis, labelled as such |
| Figure 6 (AlphaFold2 structures) | not attempted; needs AF2 compute, and the paper limits the claim to visual comparison at pLDDT 40–60 |
| Section 2.2 BLAST novelty | needs Table S2, or network BLAST. Substituted with offline k-mer measures, never described as alignment statistics (A4) |
| Figure 5 against the paper's sequences | needs Table S1. Reproduced instead on our own generations vs their nearest silkome neighbours, stated wherever used |
