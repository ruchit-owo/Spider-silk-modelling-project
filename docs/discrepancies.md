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

## D7 — Supporting Information not available to this reproduction

**Status: unresolved.**

We were not able to obtain the paper's Supporting Information. The PDF supplied
as SI turned out to be the supplementary file for a different paper (Pandey,
Chen & Keten, *Commun. Mater.* 2024). What this blocks:

| SI item | contents | consequence |
|---|---|---|
| Table S1 | the five published generated sequences and ten BLAST neighbours | cannot compare our generations against the published ones directly |
| Table S2 | BLAST query cover and percent identity | the novelty analysis of Section 2.2 cannot be reproduced; we substitute offline k-mer measures, clearly labelled |
| Table S3 | ProtParam values and pLDDT scores | Figure 5 is reproduced on our own generations, not against the published ones |
| Table S4 | the motif definitions | **Figure 7 cannot be reproduced.** We fall back to canonical spidroin motifs, labelled `canonical_fallback`, which is a different analysis |
| Table S5 | normalisation constants | **recovered independently and verified** - see assumption A2. Not blocking. |

Of these, only Table S4 blocks a headline figure. If it becomes available, drop
it in `data/raw/table_s4_motifs.csv` and `scripts/05_motif_analysis.py`
switches from fallback to reproduction with no other change.

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
