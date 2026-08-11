# Discrepancies and open points

Differences between what the paper reports and what we measured, plus points
where the paper leaves something unstated that materially affects how a number
should be read.

These are recorded neutrally. A short communication cannot state every detail,
and several entries below are upstream tooling changes that postdate
publication rather than anything the authors did. Where we could verify
something, the verification is given; where we could not, that is said plainly
instead of guessed at.

Status key: **confirmed** (we measured it), **unresolved** (blocked on an input
we do not have), **clarification** (not a disagreement; a reading that is easy
to get wrong).

---

## D1 — Figure 5c: helix and sheet labels are exchanged relative to current Biopython

**Status: confirmed.**

The Figure 5 caption defines the secondary-structure fractions as the
proportion of residues tending to be in

- helix: V, I, Y, F, W, L
- turn: N, P, G, S
- sheet: E, M, A, L

Those are exactly the groupings used by Biopython's
`ProteinAnalysis.secondary_structure_fraction()` up to and including v1.81, and
the caption reproduces that function's own documentation of the time.

Biopython v1.82 changed the method. Its current docstring records why:

> prior to v1.82, this method wrongly returned (Sheet, Turn, Helix) while
> claiming to return (Helix, Turn, Sheet)

That is, V I Y F W L is a *sheet* propensity set and E M A L is a *helix*
propensity set; the older code returned them under swapped labels. v1.82 also
widened the groups to match the literature it cites, giving helix = E M A L K,
turn = N P G S D, sheet = V I Y F W L T.

**Consequence.** Under the corrected assignment, the bars labelled "helix" in
Figure 5c are sheet propensities and those labelled "sheet" are helix
propensities. The paper used the library as documented at the time; the labels
were corrected upstream afterwards. This matters more here than it would in
most papers, because beta-sheet content is the central structural quantity in
spider silk, and a reader comparing Figure 5c against silk structural
literature would draw the wrong conclusion about which residues drive sheet
content.

**What we do.** `silkrepro.protparam` computes both conventions explicitly,
without calling the version-dependent Biopython method, and reports both in
every table (`frac_*_legacy`, `frac_*_corrected`). Figure 5 is reproduced using
the legacy convention, so it matches the paper, and is annotated with the
corrected values.
`tests/test_protparam.py::test_legacy_helix_equals_current_biopython_sheet`
pins the relationship so this entry cannot go stale silently.

## D2 — Property set 3's strength value departs from the stated median

**Status: confirmed (internal to the paper).**

The paper says that for each self-consistency set, the property values other
than the two being varied are set to the dataset medians. Sets S1, S2 and S4
all carry strength = 0.310. Set S3 varies toughness and E, so its strength
should also be 0.310, but Table 1 and the Figure 2 caption both give **0.200**.

Both locations agree with each other, so this is not a typesetting slip in one
place. We keep the published value, since using 0.310 instead would change
S3's target spread and therefore its R2, and changing a published input to get
a better match would not be reproduction.

Asserted in
`tests/test_config_against_paper.py::test_set3_strength_departs_from_the_stated_median`.

## D3 — The sampling budget behind the headline R2 values is not stated

**Status: clarification, quantified by our extension.**

The paper reports R2 values of 0.8764, 0.8369, 0.6889 (Figure 3) and 0.8899,
0.5640, 0.7167, 0.7843, 0.7751 (Figure 4), describing them as "the results
with the highest R2 values ... selected from a set of sampling attempts". The
size of that set is not given anywhere in the text.

The released notebook gives it: `mum_samples_perstep=32`, `num_repeat=64`,
followed by `np.argmax` over the pooled R2 values - a maximum over up to
**2,048** candidates per property set.

A maximum over N draws grows with N with no change to the model. Without N, a
reader cannot compare these values against a single-shot number from another
method, and cannot estimate the compute needed to reach them. This is not a
flaw in the procedure - screening many generated candidates is exactly what one
should do when generation is cheap - but it does change what the number means.

`scripts/10_ablation_bestofn.py` reports the full distribution and the
best-of-N curve. See `RESULTS.md` for the measured numbers.

## D4 — "R2" is computed within one 8-vector, not across a test set

**Status: clarification.**

The R2 values in Table 1 are `r2_score(target_8vector, predicted_8vector)`:
one designed sequence, its eight predicted property values against the eight
values requested. The released notebook makes this explicit inside
`generate_new()`.

This is a self-consistency measure - does the model's forward pass agree with
the target its inverse pass was asked to hit - and the paper describes it as
such. It is not a measure of generalisation to unseen sequences and should not
be read as one, which is easy to do given that "R2" in a property-prediction
context usually means the latter.

Two further consequences:

- R2 here is measured against the variance of the eight target values. Sets
  whose eight entries cluster tightly have a small denominator and a brittle
  R2. We report `target_spread` alongside every within-vector R2.
- Four of the eight entries are standard deviations, not properties. A model
  that hits the SD slots and misses the properties still scores well. We report
  the four mechanical slots separately.

`silkrepro.metrics` implements both definitions under distinct names and never
mixes them.

## D5 — There is no held-out evaluation

**Status: clarification.**

The Experimental Section states that "the fine-tuning training set includes all
known pairs of sequence and properties". So every one of the 1,033 pairs was
seen in training, and any forward-task accuracy measured on them - including
ours in `scripts/04_forward_eval_dataset.py` - is a training-set number.

This bounds what can be claimed about predictive accuracy from above and gives
no estimate of it. Producing one would require retraining with a held-out fold.
We do not do that, and we do not present any number in this project as a
generalisation estimate for SilkomeGPT. Our composition baselines *are*
cross-validated, and are reported with both in-sample and cross-validated
scores so the comparison against the transformer is like-for-like in the first
case and honest in the second.

## D6 — A large fraction of generations reproduce training sequences verbatim

**Status: confirmed.**

Novelty in the paper's pipeline is exact string membership in
`ALL_SILK_SEQ.csv`, and non-novel candidates are discarded before scoring. The
paper reports the surviving sequences but not the discard rate.

In our runs a substantial fraction of inverse-task generations are byte-identical
to sequences in the reference set. The measured rate is in `RESULTS.md` and in
`results/table1_reproduction.csv` (`n_generated` vs `n_novel`).

This does not contradict any claim in the paper, which analyses only the novel
outputs. It does mean the effective sampling budget is smaller than the nominal
one, and it is worth knowing when reading the novelty argument in Section 2.2:
the model's default behaviour includes a good deal of retrieval.

## D7 — Supporting Information: initially unavailable, later located

**Status: resolved.**

The Wiley Supporting Information could not be retrieved (the publisher returns
402/403 to automated requests), and a PDF initially supplied as the SI turned
out to be the supplementary file for a different paper (Pandey, Chen & Keten,
*Commun. Mater.* 2024; see D12).

The supplementary tables were subsequently located in the **arXiv preprint of
the same work**, [arXiv:2309.10170](https://arxiv.org/abs/2309.10170), which
carries them as appended pages: Table S1 on p. 31, S2 on p. 35, S3 on p. 36,
S4 on p. 37, S5 on p. 38.

| SI item | contents | status now |
|---|---|---|
| Table S1 | the five published generated sequences and ten BLAST neighbours | **available**, not yet incorporated; would let Figures 5 and 7 be compared against the paper's own specimens rather than nearest database neighbours |
| Table S2 | BLAST query cover and percent identity | **available**, not yet incorporated |
| Table S3 | ProtParam values and pLDDT scores | **available**, not yet incorporated |
| Table S4 | the motif definitions | **obtained and in use.** `data/raw/table_s4_motifs.csv`; `scripts/06_motif_analysis.py` now reports `motif_source = table_s4`, so Figure 7 is a genuine reproduction. All 14 entries verified against the Silkome Table 1 they derive from (D13) |
| Table S5 | normalisation constants | **obtained, and it confirms our independent recovery exactly** on all eight properties - see A2 and D14 |

Table S4 as transcribed (14 motifs):

| ID | motif | property | effect |
|---|---|---|---|
| T_pos_1 | GYGQGG | toughness | + |
| T_pos_2 | GGGQ | toughness | + |
| T_neg_1 | SQGP | toughness | − |
| T_neg_2 | SY | toughness | − |
| T_neg_3 | SV | toughness | − |
| TS_pos_1 | GYGQGG | tensile strength | + |
| TS_pos_2 | QGGS | tensile strength | + |
| TS_neg_1 | PQ | tensile strength | − |
| SB_pos_1 | GYGQGG | strain at break | + |
| SB_pos_2 | QGP | strain at break | + |
| SB_pos_3 | PGA | strain at break | + |
| E_pos_1 | PA | elastic modulus | + |
| E_pos_2 | GQ | elastic modulus | + |
| E_neg_1 | GGQ | elastic modulus | − |

## D13 — WITHDRAWN. Table S4 matches its cited source on all 14 entries

**Status: withdrawn. There is no discrepancy.**

An earlier version of this entry reported that three motifs — GQ (`E_pos_2`),
QGP (`SB_pos_2`) and PGA (`SB_pos_3`) — carried the opposite sign in Table S4
to the Silkome Table 1 it cites. **That claim was wrong and is retracted.**

Checked against the published table read directly from the *Science Advances*
PDF, Silkome Table 1 lists:

- **Strain at break, positive effect:** MaSp1-GYGQGG; **QGP, PGA in MaSp1**
- **Young's modulus, positive effect:** PA in MaSp2; GL in MaSp1 and MaSp2;
  **MaSp1-GQ**
- **Young's modulus, negative effect:** Q in MaSp2; **MaSp1-GGQ**

All fourteen Table S4 entries agree with this. Lu et al.'s transcription is
faithful.

**Where the error came from, since it is instructive.** The withdrawn claim
rested on two automated retrievals of the PMC version of the Silkome paper.
Table 1 sits in the left column of a two-column page, and its "positive
effect" and "negative effect" columns are separated by whitespace rather than
rules. Linearised text extraction interleaves the two columns and the body
text, and both retrievals assigned several entries to the wrong column. Reading
the same table as a rendered image resolves it immediately and unambiguously.

The original entry did flag that it rested on a web fetch rather than a local
computation, and recommended checking the source directly. Doing so is what
overturned it. Two lessons kept for the rest of this project: table structure
in multi-column PDFs must be read visually, not from extracted text; and a
claim that a published paper contradicts its own source deserves a higher
evidence bar than a claim about our own measurements, because the cost of
getting it wrong falls on someone else.

`data/raw/table_s4_motifs.csv` was unaffected — it was transcribed from a
rendered image of Table S4 from the outset, and needs no correction.

## D15 — Table S1 extraction: resolved on the second implementation

**Status: resolved.** All rows now verify against the published molecular
weights. The account below of the first, failed attempt is kept because the
failure mode is worth knowing.

**The rewrite.** The rules are ignored entirely. Every glyph in the sequence
column is read from `page.chars`, grouped into lines, filtered to residue-only
lines by content, concatenated in document order, and cut at the lengths
published in Table S3 in the row order read from the ID column. Because
published lengths are now an *input*, length cannot verify the result;
molecular weight does, and it is independent of the split criterion — a split
off by one position changes the composition of two sequences and both weights.

Result: total 18,844 residues, exactly the Table S3 sum, and **13 of 13
checkable rows match their published molecular weight to the cent**. The other
two contain `X` ambiguity codes so ProtParam cannot weigh them; they are
length-exact and bracketed by verified rows in the same contiguous blob.

One further detail cost 74 residues before it was found: the last line of each
sequence is a partial one, sometimes only a few residues, and the original
20-character minimum silently discarded them. No length threshold is needed —
the residue-only alphabet test alone separates sequence lines from the caption,
header and page numbers, all of which contain lowercase or digits.

**What it unlocked.** With the sequences verified, the forward task was checked
against the paper's own specimens and reproduces the published R² exactly on
four of five sets, with the full eight-value predicted vector identical slot
for slot. See `RESULTS.md` §8g. That in turn shows the shortfall in §8 is
entirely in the inverse task's search, not the scorer.

### The first attempt, and why it failed

**Do not use rule-rectangle row detection on this table.**

Table S1 (arXiv:2309.10170, pp. 31-34) holds the five sequences the paper
generated and their ten BLAST neighbours. Extracting it would allow Figures 5
and 7 to be reproduced against the paper's own specimens, and would allow the
forward task to be checked on a sequence whose expected R2 is published.

The extraction is not correct yet, and the outputs have been deleted rather
than kept with a caveat.

**What the checks show.** Table S3 publishes both the length and the molecular
weight of all fifteen sequences. Against those:

- five rows are off by exactly ±63 residues, one text line, so whole lines are
  being attributed to the adjacent row;
- of the ten rows whose length *does* match, nine still have the wrong
  molecular weight. Length can match while composition does not, because every
  line in the table is the same width, so a line swapped between two adjacent
  rows leaves both lengths unchanged;
- exactly one row, 5.3, matches its published MW to the cent (224113.32),
  which shows the machinery can be exact and the remaining errors are in row
  attribution rather than in glyph reading.

Three fixes were tried and none resolved it: reading raw `page.chars` instead
of `extract_words` (identical output, so glyphs are being read faithfully);
tightening the band boundaries to half-open with no tolerance; and changing
whether a page's leading band belongs to the row above or below.

**Why this is recorded rather than quietly retried.** An interim version of
this work ran the forward task on the corrupted sequences and obtained R2
values far *below* the published ones — which, had it been believed, would have
been reported as the paper failing on its own specimens. It was retracted
before reaching `RESULTS.md`. The corrected extraction gives the opposite
answer: exact agreement.

The molecular-weight check is what caught it. Length alone passed ten of
fifteen rows, because every line in the table is the same width and a line
misattributed between adjacent rows leaves both lengths unchanged. Any future
work on this table should verify against MW, not length.

## D14 — Recovered normalisation constants match the published Table S5 exactly

**Status: confirmed.**

Before the Supporting Information was located, the Table S5 normalisation
constants were recovered from the Silkome data and verified against a single
published worked example (assumption A2). Table S5 has since been obtained from
the arXiv preprint, and it matches our recovered values on all eight
properties:

| property | unit | Table S5 | ours |
|---|---|---|---|
| toughness | GJ/m³ | 0.005 – 0.39 | 0.005 – 0.39 |
| SD toughness | GJ/m³ | 0.001 – 0.136 | 0.001 – 0.136 |
| E | GPa | 0.38 – 37.0 | 0.38 – 37.0 |
| SD E | GPa | 0.03 – 9.76 | 0.03 – 9.76 |
| strength | GPa | 0.17 – 3.33 | 0.17 – 3.33 |
| SD strength | GPa | 0.01 – 0.8 | 0.01 – 0.8 |
| strain | % | 5.1 – 53.2 | 5.1 – 53.2 |
| SD strain | % | 0.1 – 13.7 | 0.1 – 13.7 |

This also confirms assumption A1, the dataset reconstruction rule: the
constants are a property of the population they were computed over, and
recovering them exactly means we assembled the same population the authors did.

The units, which we did not have before, are recorded here for reference.

## D8 — The worked example sequence, as printed, differs from the Silkome record

**Status: confirmed, ours not theirs.**

The Experimental Section prints a sequence for the `CalculateSilkContent`
example. Extracting it from the PDF yields 635 residues, which is a 94.9%
6-mer-containment match to Silkome record `idv_id 7305` (*Nephilingis livida*,
MaSp, 671 residues) and not a byte-identical match to anything.

The difference is almost certainly our PDF text extraction losing characters
across line wraps, not an error in the paper. We record it because the
transcription is used in a test: submitting our 635-residue transcription to
the forward task nonetheless returns exactly the published vector
`[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]`, which is the
strongest single confirmation that we have the right checkpoint and the right
prompt format.

## D10 — We do not reach the reported R² on any of the eight property sets

**Status: confirmed. This is the reproduction's main shortfall.**

Generating 2,048 candidates per set with the released checkpoint and the
notebook's decoding settings, our maximum falls below the published value on
all eight sets, by between 0.011 (F3) and 0.436 (S4), mean 0.27. In no set did
any of ~700–800 scoreable candidates reach the published figure.

One candidate explanation was tested and **rejected**: our greedy forward
decoding (A5) removes scoring noise that the notebook's sampled decoding
carries, and a maximum over noisy scores is inflated. Re-scoring the identical
candidates with the notebook's sampled decoder moved the maximum by +0.0065 on
average (`scripts/10b_scoring_stochasticity.py`), nowhere near enough.

Remaining candidate explanations, which we cannot distinguish from the
information available: a larger effective candidate pool in the original run
(ours is ~771 scoreable per set after the 62.3% verbatim-copy rate, see D6), a
different random seed, or a selection step present in the original work but not
described in the paper or the released notebook.

We record this as an unexplained gap rather than attributing it to any of these.
The mechanism the paper describes reproduces; the magnitude does not.

## D11 — The forward task reproduces 71% of training labels exactly

**Status: confirmed.**

Of the 1,026 fine-tuning pairs the model successfully scored, **732 (71.3%)**
come back with all eight property values identical to the training label, to
within 0.0005 — the model's own output resolution. Mean absolute error on those
rows is 0.00025. On the remaining 294 rows it is 0.109, and the mean R² over
the four mechanical properties is **−0.113**, worse than predicting the dataset
mean.

So the in-sample R² of 0.696 (§8d of `RESULTS.md`) is composed mostly of exact
label retrieval rather than prediction from sequence.

Three things this is not:

- **Not a contradiction of any claim.** The paper fine-tunes on all known pairs,
  says so plainly, reports self-consistency rather than held-out accuracy, and
  makes no generalisation claim. Memorisation is the expected consequence of
  training on everything and evaluating on the same data, which is what the
  design implies.
- **Not a generalisation estimate.** The non-memorised rows are selected on the
  outcome and differ systematically from the rest (median length 456 vs 361),
  so −0.113 is a diagnostic, not a held-out score.
- **Not isolated.** It is consistent with three other measurements: 62.3% of
  inverse-task generations are verbatim training sequences (D6); the model's
  accuracy collapses on non-MaSp spidroins it never saw (+0.83 → −0.27…−0.54);
  and its predictions barely vary with spidroin family at all (between/within
  spread ratio 0.162). The same behaviour shows up in both task directions.

## D12 — A reviewer comment described a different paper's Figure 7

**Status: clarification, recorded for provenance.**

During this work a reviewer comment recommended reproducing "Figure 7" by
slicing the dataset into top-10/bottom-10 by property, computing per-residue
composition via "Eq. S2", forming `Cdiff` via "Eq. S3", grouping residues as
hydrophobic/polar/charged, and comparing against the residues
V I P L F T Q Y N S D E K R.

That describes **Supplementary Note 8 and Supplementary Figure 7 of Pandey,
Chen & Keten, *Commun. Mater.* 2024**, not Figure 7 of Lu, Kaplan & Buehler.
The equations, the top-10/bottom-10 slicing, the three-group normalisation and
the residue list all appear verbatim in that paper's supplementary file. Lu et
al.'s Figure 7 is a motif count and positional-KDE analysis using the motif
definitions of its own Table S4.

The confusion is easy to make: the supplementary PDF supplied for this project
was that paper's, not Lu et al.'s (see D7).

We implemented the analysis anyway, as
`scripts/14_composition_association.py`, because it is a useful model-free
companion to the ablations — but labelled as reproducing the Keten
supplementary analysis, not Lu et al.'s Figure 7. The reviewer's methodological
point, that the residue grouping is unspecified and must be chosen and stated
explicitly, was correct and is handled as assumption A15.

## D9 — Argument order of `r2_score` differs between two places in the released code

**Status: clarification, no effect on the published numbers.**

In `generate_new()` the notebook calls `r2_score(req, prop)` - target first,
prediction second, which is the convention sklearn documents and which produces
the values reported in Table 1. In the separate `validate()` helper it calls
`r2_score(c_res, c_GT)` - predictions first. R2 is not symmetric in its
arguments, so the two are not the same quantity.

`validate()` does not appear to produce any number in the paper, so this does
not affect the published results. We note it only because someone re-using that
helper would get a different quantity than Table 1's. This project always uses
(target, prediction) and has a test asserting the two orders differ, so the
convention cannot drift.
