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

Sharper than it first appears (`docs/discrepancies.md` D8). The paper prints
two variants of this record: the 671-residue Silkome record `idv_id 7305`, and
the 635-residue forward example, which is that record with one contiguous
36-residue block deleted. Our transcription is byte-exact against the record on
all 635 of its characters. **Both** return the published vector. So the model
returns the published answer for a sequence that is not the database record and
is missing 36 residues from it. Two known-answer tests, not one.

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

**Two references, one of which is vacuous and one of which is not.**

Resubmitting the identical prompt gives a floor of exactly zero — the forward
task returned one distinct output across every repeat of every probe. Under
greedy decoding (A5) that is true by construction, so this floor can never
reject anything. It is reported to demonstrate determinism, not as a safeguard;
an earlier version of the documentation oversold it as a live check.

The reference that works is a **single conservative point mutation** — one
residue swapped for a chemically similar one, the smallest edit that changes
the sequence at all. Measured at **0.0148**, 0.17× the reference scale, with a
median of exactly zero: most single substitutions change nothing. Every
ablation below is well above it, so all of them clear the bar that can actually
be failed.

**Reference scale: 0.0874.** This is the mean absolute deviation of predictions
across the 30 unmodified probes — how much the model's output moves between
different natural silk sequences. Effects are given as a fraction of it, since
an absolute shift of 0.13 means nothing without knowing the range.

| ablation | mean \|Δ\| | × reference scale |
|---|---|---|
| shuffle | 0.1467 | 1.68 |
| reverse | 0.1322 | 1.51 |
| block shuffle | 0.1301 | 1.49 |
| core only (termini removed) | 0.1268 | 1.45 |
| **random sequence, matched length** | **0.1230** | **1.41** |
| scattered control (GPGXX) | 0.1063 | 1.22 |
| scattered control (polyA) | 0.1043 | 1.19 |
| knockout GPGXX | 0.0954 | 1.09 |
| scattered control (GGX) | 0.0914 | 1.05 |
| knockout polyA | 0.0809 | 0.93 |
| termini only (core removed) | 0.0773 | 0.89 |
| knockout GGX | 0.0715 | 0.82 |
| substitute GGX | 0.0697 | 0.80 |
| substitute polyA | 0.0674 | 0.77 |
| substitute GPGXX | 0.0672 | 0.77 |
| block control (GGX) | 0.0551 | 0.63 |
| block control (polyA) | 0.0493 | 0.56 |
| block control (GPGXX) | 0.0429 | 0.49 |
| *point mutation (floor)* | *0.0148* | *0.17* |

Two findings.

**(a) No motif-specific effect.** Every knockout lands *between* its two
controls:

| motif | knockout | scattered control | block control |
|---|---|---|---|
| polyA | 0.0809 | 0.1043 | 0.0493 |
| GGX | 0.0715 | 0.0914 | 0.0551 |
| GPGXX | 0.0954 | 0.1063 | 0.0429 |

A motif knockout removes several contiguous runs, so sitting between a
scattered removal and a single block removal of the same size is exactly what
removing that many residues *without* any motif-specific sensitivity predicts.
Removing poly-A tracts — the β-sheet nanocrystal formers, the most
mechanically consequential motif in spider silk — perturbs the prediction
*less* than deleting the same number of residues at random positions. We find
no evidence that the forward task treats these motifs as special.

**(b) Destroying the sequence entirely is not much worse than perturbing it.**
Replacing a spidroin with uniform random residues of the same length
(`random_matched`, 0.1230) moves the prediction about as far as shuffling it
(0.1467) or reversing it (0.1322) — all three sit near 1.4–1.7× the spread
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

### 7a. The baseline on Table 1's own axis

The same baseline scored the paper's way — within-vector R², one value per
sequence — puts it on the only axis directly comparable to Table 1:

| | within-vector R² |
|---|---|
| composition ridge on real pairs | mean −1.70, median −0.22, **max 0.697**, fraction > 0: 0.38 |
| paper, eight property sets | 0.564 – 0.890 |
| ours, eight property sets | 0.223 – 0.722 |

A ridge regression on amino-acid counts reaches a within-vector R² of 0.697 on
the best of 1,033 real sequences — inside the range the paper reports and above
five of our own eight maxima.

**This is not a like-for-like comparison and should not be read as one.** The
baseline figure is the maximum over 1,033 in-sample fits to sequences whose
labels it was trained on; the paper's is the maximum over roughly 2,000
*generated* sequences scored against a target that was never in any training
set. Different procedures, different denominators. What the number does show is
that a value in the 0.6–0.7 band, on this metric, is reachable without a
transformer and without generation — which is worth knowing before reading any
single such value as evidence of sequence understanding.

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

## 8g. The forward task reproduces the published R² exactly

**measured** (`scripts/08_extract_table_s1.py`, `scripts/09_verify_published_sequences.py`)

This is the decisive result, and it reframes §8.

Table S1 gives the five sequences the paper actually generated; Table S3 gives
the property vector its forward task predicted for each, and the resulting R².
So both the input and the expected output are known, and the forward task can
be checked with no sampling and no search.

Submitting the paper's own sequences to the released checkpoint:

| set | paper R² | ours | difference | predicted vector vs Table S3 |
|---|---|---|---|---|
| S1 | 0.8899 | **0.8899** | +0.0000 | **8/8 slots identical** |
| S2 | 0.5640 | 0.5176 | −0.0464 | 6/8 identical |
| S3 | 0.7167 | **0.7167** | −0.0000 | **8/8 identical** |
| S4 | 0.7843 | **0.7843** | +0.0000 | **8/8 identical** |
| S5 | 0.7751 | **0.7751** | +0.0000 | **8/8 identical** |

Four of the five reproduce to four decimal places, and not merely in R² — the
entire eight-value predicted vector is identical to the published one, slot for
slot. S2 matches on its first six slots and differs on the last two
(strain 0.245 vs 0.247, strain SD 0.151 vs 0.257), which is consistent with the
paper's sampled forward decoding differing from our greedy decoding on a
low-confidence tail.

**What this settles.** The forward task reproduces exactly. Therefore the
shortfall reported in §8 — where our maximum fell below the published value on
all eight sets — is **entirely attributable to the inverse task's search**, not
to the scorer, the checkpoint, the prompt format, or our decoding choice. Our
2,048 candidates simply did not include sequences as good as the ones the
authors found. Their generated sequences are also markedly longer than ours
(1096, 926, 312, 869, 521 residues against our 530, 493, 471, 185, 256), so the
two searches explored different regions.

This is a stronger reproduction result than §8 alone suggested, and it should
be read as correcting the impression that section gives on its own.

**One observation in passing.** Across the five published sequences the forward
task returns only three distinct vectors: S1 and S5 both begin
[0.278, 0.274, 0.182, …] and S2 and S4 both begin [0.745, 0.141, 0.511, …].
Five different sequences, three distinct answers. That is consistent with the
retrieval behaviour documented in §8d.

## 8h. Figure 7 against the paper's own specimens

**measured** (`scripts/09_verify_published_sequences.py`, Table S4 motif set)

Motif-density profile correlation, each generated sequence against its own two
BLAST neighbours from Table S1:

| set | Pearson r | paper's characterisation |
|---|---|---|
| S1 | 0.658 | "particularly evident" |
| S2 | **0.448** | "congruence is diminished in set 2" |
| S3 | **0.941** | "particularly evident" |
| S4 | **0.944** | "particularly evident" |
| S5 | 0.660 | — |

This supports the paper's qualitative claim. Set 2 is the weakest by a clear
margin, exactly as the paper reports, and sets 3 and 4 are the strongest. Set 1,
which the paper groups with 3 and 4, comes out middling here rather than high.

Note that our earlier run using nearest database neighbours as substitutes gave
a different and misleading picture (S2 at 0.99, S4 at −0.02). Using the paper's
actual comparison sequences changes the answer, which is worth remembering
whenever a substitute specimen stands in for a published one.

## 8i. Our descriptors match the paper's published ones

**measured** (`scripts/09b_verify_against_tables_s2_s3.py`)

Table S3 publishes molecular weight, instability index and isoelectric point
for all fifteen Table S1 sequences, computed with ProtParam. Recomputing them
from the extracted sequences:

| descriptor | agreement |
|---|---|
| molecular weight | **15/15** |
| isoelectric point | **15/15** |
| instability index | 13/15 |

This validates the Figure 5 reproduction against the authors' own output rather
than against our expectations, and it would have caught a Biopython version
difference, a units error, or a bad sequence.

Two details resolved along the way, both concerning the `X` ambiguity codes in
sequences 1.3 and 5.2 (the same accession):

- Biopython refuses to weigh a sequence containing `X`, yet the paper reports a
  weight. Recovering the implied mass — (published MW − MW with `X` dropped) /
  number of `X` — gives **111.985 Da**, the conventional average residue mass,
  for both sequences independently. So the paper's tool weighed unknown
  residues as average ones rather than discarding them. Applying that
  convention reproduces both published weights exactly.
- The instability index still differs for those two (28.88 against a published
  28.55). That index is computed from dipeptides, and removing an `X` joins the
  residues on either side into a dipeptide that was never in the sequence. The
  residual is a consequence of how the unknown residue is handled, not a
  disagreement about the sequence. We report it rather than tuning it away.

## 8j. What the k-mer novelty proxy actually tracks

**measured** (`scripts/09b_verify_against_tables_s2_s3.py`)

Our offline k-mer measures stand in for BLAST (assumption A4) and had never
been calibrated against anything. Table S2 publishes, per property set, the
highest BLAST query cover and percent identity, which allows a first check:

| | vs published QC | vs published id% |
|---|---|---|
| max 6-mer containment | **+0.65** | −0.55 |
| max 6-mer Jaccard | +0.62 | −0.50 |

Over five points, so nothing here is established. What the sign pattern
suggests is worth recording anyway: the k-mer measures track **query cover**,
how much of the sequence aligns, and if anything run *opposite* to percent
identity. That is the expected behaviour for an order-insensitive set overlap,
and it means the proxy is not a stand-in for identity even loosely. A4 already
said it is not percent identity; this puts a number on the distinction.

**A note on the paper's own novelty criterion**, applied to its own numbers.
The paper cites 50–60% similarity as the threshold below which a sequence
counts as novel. Highest percent identity by set: 38, **72**, 46, 44.5, 44.
Set 2 sits above the threshold — but at a query cover of 11%, an alignment
covering a ninth of the query. The paper's own note for set 2 reads that the
sequences "display limited alignment, yet reasonable composition similarities".
Identity read without cover would overstate the similarity, so the pair belongs
together; we record both rather than either alone.

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
of the mean prediction is **0.019** against a within-family spread of
**0.095**, a ratio of **0.199** — both measured over the same four mechanical
properties. Given a spidroin it has not memorised, the model emits something
close to its training marginal regardless of which silk family the sequence
came from.

*(An earlier version reported 0.162, taking the numerator over the four
mechanical properties and the denominator over all eight. The SD slots vary
more, so that inflated the denominator. The corrected figure is 0.199; the
conclusion is unchanged, both being far below 1.)*

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
| scheme A (Lehninger-style) | A E G K Q S T |
| scheme B (alternative) | A E G K Q S V |
| **agreed by both** | **A E G K Q S** |
| Pandey/Chen/Keten reported | D E F I K L N P Q R S T V Y |

**The overlap with the published list is not distinguishable from chance.**

| scheme | observed overlap | permutation null | p |
|---|---|---|---|
| A | 5/14 | 5.3/14 (p5–p95 3–7) | 0.75 |
| B | 5/14 | 5.5/14 (p5–p95 4–7) | 0.79 |

The null applies the identical selection rule to uniform random `C_diff`
values, 2,000 replicates. Our real data does no better than noise.

**An earlier version of this section reported "both schemes recover 12 of their
14 residues" as convergent evidence. That claim was worthless and is
retracted.** It used `--select-per-group 5`, and the groups are small — the
charged group has five members in scheme A and four in scheme B — so every
charged residue was selected automatically regardless of the data, and the rule
picked 17.8 of 20 residues on average. Measured against the same null, random
noise recovers **12.7 of 14**, i.e. *better* than the real data's 12. The
analysis could not fail.

The rule now selects two per group, and the null is reported alongside the
observed value every time. The honest result is negative: at this dataset size
and with this method, we cannot distinguish the composition signal from chance.

The grouping ambiguity remains real and is documented (assumption A15), but it
is no longer the interesting part — the selection rule was the problem, not the
grouping.

## 9. Not reproduced

| item | reason |
|---|---|
| Figure 6 (AlphaFold2 structures) | not attempted; needs AF2 compute, and the paper limits the claim to visual comparison at pLDDT 40–60 |
| Section 2.2 BLAST novelty | needs network BLAST or a local database, neither of which this project runs. Table S2 is now available but reports the authors' BLAST output rather than letting us reproduce it. Substituted with offline k-mer measures, never described as alignment statistics (A4) |
| A held-out estimate of forward-task accuracy | would require retraining with a fold held out (D5) |

Reproduced after the Supporting Information was located in the arXiv preprint
(D7), having previously been listed here as not reproduced:

| item | status |
|---|---|
| Figure 7 (motif analysis) | **reproduced** with the Table S4 motif set, against the paper's own BLAST specimens (§8h) |
| Figure 5 (protein descriptors) | **reproduced** against the paper's Table S1 sequences |
| Table 1 forward predictions | **reproduced exactly** on four of five sets (§8g) |
| Table S5 normalisation | **recovered independently, then confirmed** against the published table (D14) |
