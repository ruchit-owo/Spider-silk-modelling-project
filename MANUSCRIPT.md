# What SilkomeGPT's reported accuracy measures: a reproduction and four probes

**A reproduction of** W. Lu, D. L. Kaplan, M. J. Buehler, "Generative Modeling,
Design, and Analysis of Spider Silk Protein Sequences for Enhanced Mechanical
Properties", *Adv. Funct. Mater.* **34**, 2311324 (2024).

---

## Abstract

We reproduce the SilkomeGPT pipeline against the authors' released checkpoint
and reconstruct its training data from primary sources. The checkpoint matches
the architecture the paper describes to the parameter, and returns the paper's
own published property vector for the worked example printed in its methods.
We recover the dataset (1,033 pairs, exactly) and the normalisation constants
of Table S5, which we could not obtain, verifying the latter against a
published value to within 0.0005.

**The forward task reproduces the published R² exactly.** Given the paper's own
generated sequences, four of the five self-consistency sets return the reported
R² to four decimal places, with the entire eight-value predicted vector
identical slot for slot. Our own end-to-end reproduction nonetheless falls
short on all eight property sets, by between 0.011 and 0.436 — and because the
scorer reproduces exactly, that shortfall is located precisely: it lies in the
inverse task's search, which did not find candidates as good as the authors'.

Four extensions characterise what the pipeline is doing. First, the reported R²
is largely a product of the sampling budget: the median single candidate has
negative R² in six of eight sets, and the expected best-of-N climbs from
roughly −0.4 at N=1 to between 0.09 and 0.59 at N=64 with no change to the
model. Second, motif knockouts (poly-A, GGX, GPGXX) land between two
length-matched controls, giving no evidence of motif-specific sensitivity.
Third, a composition-only baseline reaches R² 0.80 in-sample and −0.16
cross-validated, showing that on this dataset in-sample fit and generalisation
are nearly unrelated. Fourth, and connecting the rest: the forward task returns
the exact training label for 71.3% of the fine-tuning pairs, 62.3% of
inverse-task generations are verbatim training sequences, and accuracy
collapses from +0.83 to between −0.27 and −0.54 on spidroin families outside
MaSp, while the prediction barely varies with family at all.

None of this contradicts a claim the paper makes. The paper fine-tunes on all
known pairs, says so, and reports self-consistency rather than held-out
accuracy. What our measurements change is how those numbers should be read.

---

## 1. Why reproduce this paper

Generative protein design has moved quickly enough that the field's reporting
conventions have not entirely caught up with it. SilkomeGPT is a good case to
examine because it is unusually open: the checkpoint is public, an inference
notebook is public, and the underlying Silkome data is public. Almost
everything needed to check the work is available, which is not true of most
papers in this area.

The paper trains a 253.6M-parameter decoder-only transformer on 1,033 major
ampullate spidroin sequences paired with fibre-level mechanical properties, and
runs it in both directions. A forward task predicts eight property values from
a sequence. An inverse task designs a sequence to hit eight target values. The
two are chained: ask for a target, get a sequence, feed the sequence back, and
see whether the predicted properties match what was asked. Agreement is
reported as R², and those R² values — 0.5640 to 0.8899 across eight property
sets — are the paper's quantitative core.

We set out to reproduce those numbers and, where the paper leaves something
unstated, to measure rather than guess what the unstated thing does.

## 2. What we could verify exactly

Three checks came out clean, and they matter because they establish that we are
running the same model on the same data.

**The checkpoint is the model described.** Twelve layers, eight attention
heads, hidden size 1024, intermediate size 4096, and 253,556,736 parameters
against a stated 253.6M. All four hyperparameters match.

**The forward task returns the paper's own answer.** The Experimental Section
prints one fine-tuning record in full — a sequence, and the eight values paired
with it. Submitting that sequence returns

```
[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]
```

exactly the published vector, on all eight entries. This is the only test in
the project with a known correct answer, and it exercises the tokenizer, prompt
format, checkpoint and parser at once.

**The dataset reconstructs to 1,033.** The paper says the dataset was
"constructed and curated based on the silkome dataset" and gives no curation
rule; the pairs are not in the authors' repository. Working from the Silkome v1
release, we found that joining the protein FASTA to the mechanical-property
table on `idv_id` — the individual spider whose fibre was tested — keeping rows
with all eight properties present, and keeping types beginning with `MaSp`,
gives exactly 1,033 rows. Joining on `ncbi_tax_id` instead, which is the more
obvious key and is present in both files, returns zero rows.

Two properties of that set are worth stating because they are easy to miss:
1,033 rows contain 1,028 distinct sequences, because five sequences carry two
different label vectors from individuals whose fibres tested differently; and
the records are terminal-domain-anchored fragments (553 CTD, 480 NTD), not
complete spidroins.

### 2.1 Recovering Table S5 without the Supporting Information

We were unable to obtain the paper's Supporting Information. Table S5, which
holds the min–max constants used to normalise all eight properties to [0,1],
was therefore recovered rather than read.

The Experimental Section says the extremes were taken "across the entire
dataset (not limited to MaSp sequences)". We tested three readings of that
sentence against the worked example of §2, whose *normalised* vector the paper
publishes and which played no part in deriving the constants:

| population used for min/max | max abs. error vs published |
|---|---|
| raw `mechanical_properties.csv`, 446 rows | 0.139 |
| **sequence-joined table, all spidroin types, 3,563 rows** | **0.00047** |
| sequence-joined table, MaSp only, 1,033 rows | 0.054 |

The middle reading reproduces all eight published values to within the paper's
own three-decimal rounding. We treat this as a verification rather than a fit,
and the dataset builder re-runs the check on every invocation.

## 3. Table 1 does not fully reproduce

We generated 2,048 candidate sequences for each of the eight property sets
using the released notebook's decoding settings, discarded those that failed to
parse as protein or matched a known sequence exactly, scored the survivors with
the forward task, and took the maximum — the procedure the notebook implements.

| set | paper | our max | our median | scoreable pool |
|---|---|---|---|---|
| F1 | 0.8764 | 0.7221 | −0.03 | 815 |
| F2 | 0.8369 | 0.6964 | −1.41 | 739 |
| F3 | 0.6889 | 0.6780 | +0.09 | 735 |
| S1 | 0.8899 | 0.7120 | −2.25 | 712 |
| S2 | 0.5640 | 0.2234 | −0.67 | 738 |
| S3 | 0.7167 | 0.6494 | −0.31 | 752 |
| S4 | 0.7843 | 0.3487 | −0.48 | 710 |
| S5 | 0.7751 | 0.6233 | −3.61 | 641 |

We do not reach the published value on any set. F3 is essentially reproduced
(0.678 against 0.689). S4 and S2 fall short by 0.44 and 0.34. Across all eight,
none of roughly 700–800 candidates matched or exceeded the reported figure.

This is a partial reproduction and we describe it as one. The mechanism works —
the pipeline does produce sequences whose predicted properties approach the
requested targets — but the magnitude we obtain is lower.

### 3.1 One explanation tested, and rejected

The released notebook decodes the forward task with sampling enabled. That
makes each candidate's R² a random variable, and a maximum taken over N noisy
scores is inflated relative to a maximum over N noiseless ones. We decode
greedily, for reasons given in §6. Could that difference alone account for the
gap?

We re-scored the identical candidate sequences with the notebook's sampled
decoder — same sequences, same targets, only the scorer changed. The maximum
moved by +0.0259 on one set and by exactly zero on three others, mean +0.0065.
Scores were identical on 97–99% of candidates. The mechanism is real but two
orders of magnitude too small to explain a 0.27 gap.

### 3.2 The gap is in the search, not the scorer

Locating the Supporting Information in the arXiv preprint settled this. Table S1
gives the sequences the paper actually generated and Table S3 gives the property
vector its forward task predicted for each, so both the input and the expected
output are known and the scorer can be tested with no sampling at all.

| set | paper R² | ours | predicted vector vs Table S3 |
|---|---|---|---|
| S1 | 0.8899 | **0.8899** | **8/8 slots identical** |
| S2 | 0.5640 | 0.5176 | 6/8 |
| S3 | 0.7167 | **0.7167** | **8/8** |
| S4 | 0.7843 | **0.7843** | **8/8** |
| S5 | 0.7751 | **0.7751** | **8/8** |

Four of five reproduce to four decimals, and not merely in R² — the whole
eight-value vector matches. S2 agrees on its first six slots and differs on the
last two, consistent with the paper's sampled decoding diverging from our greedy
decoding on a low-confidence tail.

So the checkpoint, the prompt format, the parser and our decoding choice are all
exonerated. What our reproduction failed to match is the *search*: 2,048
candidates did not contain sequences as good as the ones the authors found.
Theirs are also markedly longer — 1096, 926, 312, 869 and 521 residues against
our 530, 493, 471, 185 and 256 — so the two searches explored different regions
of the space. Why theirs found longer and better candidates we cannot say from
the information available.

## 4. Where the reported number actually comes from

The paper says results "with the highest R² values are selected from a set of
sampling attempts" and never states the size of that set. The released notebook
does: 32 samples per step, 64 repeats, then `argmax` — up to 2,048 candidates
per property set.

That single unstated number governs how the result should be read, because a
maximum over N draws grows with N whether or not anything about the model
changes. Sweeping N over our pools:

| set | N=1 | N=8 | N=64 | N=512 |
|---|---|---|---|---|
| F1 | −0.36 | 0.34 | 0.56 | 0.68 |
| F2 | −3.59 | 0.41 | 0.59 | 0.67 |
| S1 | −6.20 | −0.06 | 0.55 | 0.71 |
| S5 | −11.03 | −0.46 | 0.33 | 0.57 |

A single draw from the inverse task is typically worse than useless: the median
candidate has negative R² in six of the eight sets. Essentially all of the
apparent performance is produced by generating many candidates and keeping the
best one.

We want to be careful about what follows from this. Screening thousands of
generated candidates is a perfectly sound design procedure — arguably the right
one when generation is cheap and evaluation is cheaper. The objection is not to
the method. It is that a reader given only the maximum cannot tell how much
search produced it, cannot compare it against a single-shot number from another
method, and cannot estimate the compute needed to reach it. Reporting N, or the
best-of-N curve, costs nothing and removes the ambiguity entirely.

## 5. What the forward task is doing

Four measurements, taken independently, converge on the same picture.

**It reproduces training labels.** Over the 1,033 fine-tuning pairs, the
forward task's in-sample R² averages 0.696 across the four mechanical
properties. But 732 of the 1,026 scored rows — **71.3%** — come back with all
eight values identical to the training label, to within 0.0005, which is the
resolution the model can express. Mean absolute error on those rows is 0.00025.
On the remaining 294 rows it is 0.109, and the mean R² is **−0.113**: worse
than predicting the dataset mean.

**It generates training sequences.** In the inverse direction, **62.3%** of
parsed generations across the eight property sets are byte-identical to
sequences in the reference set. These are not a handful of repeats — 200 to 290
distinct known sequences appear per set. A nominal budget of 2,048 candidates
therefore yields about 771 scoreable ones.

**It fails outside its training family.** The model is fine-tuned on MaSp only.
Given spidroins from the seven other families in the database, accuracy drops
from +0.826 to between −0.270 and −0.540, with mean absolute error rising from
0.024 to about 0.15.

**Its output barely depends on the family at all.** Across those families, the
between-family spread of the mean prediction is 0.019 against a within-family
spread of 0.095 — a ratio of 0.199, both over the same four mechanical
properties. Handed a spidroin it has not memorised, the
model emits something close to its training marginal regardless of which silk
the sequence came from.

Taken together these say the pipeline behaves substantially like a retrieval
system over its fine-tuning set, in both directions.

Three qualifications, because this result invites overreading.

It contradicts nothing the paper claims. The paper fine-tunes on all known
pairs, states this plainly in its Experimental Section, reports self-consistency
between its two tasks rather than held-out predictive accuracy, and does not
claim generalisation. Memorisation is the expected consequence of training on
everything and evaluating on the same data.

The −0.113 figure is not a generalisation estimate. Those rows are selected on
the outcome, and they differ systematically from the rest — median length 456
against 361. Selecting on the outcome biases the estimate downward by an
unknown amount. A real held-out number requires retraining with a fold held
out, which we did not do.

And the task itself is hard. Our composition-only baselines — ridge and
gradient boosting on amino-acid fractions, dipeptide fractions and motif
densities — reach R² 0.80 to 0.87 in-sample and −0.13 to −0.16 under 5-fold
cross-validation, or −0.049 when folds are grouped by species so that no
species appears in both train and test. On this dataset, with these features,
in-sample fit and generalisation are close to unrelated. That is a property of
the problem, not of any particular model.

## 6. What the sequence ablations show

If the forward task computes properties from sequence features, editing those
features should move the prediction in interpretable ways. We tested this on 30
MaSp sequences with a set of controlled edits, each destroying one kind of
information.

The first thing to establish was a noise floor. The forward task decodes with
sampling in the released notebook, so the same input need not give the same
output. Under greedy decoding we measured one distinct output across eight
repeats of every probe — the floor is zero, and every effect below is above it
by construction. We also needed a reference scale, since an absolute shift of
0.13 means nothing on its own: across the 30 unmodified probes, the model's
output varies by 0.0874, and we express effects as fractions of that.

**Motifs.** We knocked out poly-A tracts, GGX repeats and GPGXX repeats — the
β-sheet formers, the amorphous-region motif, and the MaSp2 elastic motif. Each
knockout was run against *two* length-matched controls: removing the same
number of residues scattered at random positions, and removing them as one
contiguous block.

| motif | knockout | scattered control | block control |
|---|---|---|---|
| poly-A | 0.0809 | 0.1217 | 0.0731 |
| GGX | 0.0715 | 0.1129 | 0.0401 |
| GPGXX | 0.0954 | 0.0855 | 0.0720 |

Every knockout lands between its two controls. Since a motif knockout removes
several contiguous runs, sitting between "one block" and "scattered" is exactly
what removing that many residues *without* any motif-specific sensitivity
predicts. Removing poly-A tracts — the most mechanically consequential motif in
spider silk — perturbs the prediction *less* than deleting the same number of
residues at random positions.

The second control turned out to be essential. Against the scattered control
alone, poly-A knockout looked like a large effect in the wrong direction; the
block control shows it is not an effect at all, only a difference in how the
deletion was distributed.

**Global structure.** Reversing the sequence (0.132), shuffling it (0.125),
block-shuffling it (0.125) and replacing it with uniform random residues of the
same length (0.121) all move the prediction by about the same amount — roughly
1.4× the spread the model shows across genuinely different natural sequences.
The output appears to saturate: past a certain amount of disruption, more
disruption does not move it further.

These measure the model, not spider silk. A shuffled spidroin is far outside
the fine-tuning distribution, so a large shift shows sensitivity to order
without showing that order is used meaningfully. No claim about real mechanical
behaviour follows.

## 7. A composition analysis, and an amino-acid grouping that had to be chosen

Separately from the model, we asked which residues distinguish high- from
low-performing silks in the data itself. Ranking the 175 MaSp1-bearing
individuals by each mechanical property, comparing the top ten against the
bottom ten on mean per-residue composition, and normalising the differences
within residue groups gives:

This follows Supplementary Note 8 of Pandey, Chen & Keten (*Commun. Mater.*
2024) rather than anything in Lu et al. The result is negative, and the way we
arrived at it is worth reporting.

Our first pass selected the top five residues per group and found that both
schemes recovered 12 of the 14 residues that paper reports. That looked like
convergent evidence. It was not. The groups are small — the charged group has
five members in one scheme and four in the other — so at k=5 every charged
residue is selected automatically whatever the data says, and the rule picks
17.8 of 20 residues on average. Applying the identical rule to uniform random
values recovers **12.7 of 14**, better than our real data managed. The analysis
could not fail, and an overlap statistic that cannot fail carries no
information.

With the rule tightened to two per group and a permutation null reported
alongside:

| scheme | observed | null | p |
|---|---|---|---|
| A | 5/14 | 5.3/14 | 0.75 |
| B | 5/14 | 5.5/14 | 0.79 |

At this dataset size and with this method, the composition signal is not
distinguishable from chance. That is the finding.

The grouping ambiguity we set out to handle — the source does not give the
hydrophobic/polar/charged assignment table, and the within-group normalisation
means the choice changes the output — is real and remains documented. But it
was not the thing that mattered. We had built a careful sensitivity analysis
around a secondary ambiguity while the primary statistic was vacuous, which is
its own kind of lesson: check that an analysis can fail before checking how
sensitive it is.

## 8. A labelling issue in Figure 5c

The Figure 5 caption defines its secondary-structure fractions as the
proportion of residues tending to be in helix (V I Y F W L), turn (N P G S) and
sheet (E M A L). Those are the groupings used by Biopython's
`secondary_structure_fraction()` up to v1.81, and the caption reproduces that
function's documentation of the time.

Biopython v1.82 changed the method, and its current docstring records why:
prior to that release it "wrongly returned (Sheet, Turn, Helix) while claiming
to return (Helix, Turn, Sheet)". The set V I Y F W L is a sheet-propensity set;
E M A L is a helix-propensity set.

Under the corrected assignment, the bars labelled "helix" in Figure 5c are
sheet propensities and vice versa. The paper used the library as documented
when it was written; the correction is upstream and postdates publication. It
is worth flagging because β-sheet content is the central structural quantity in
spider silk, so a reader comparing Figure 5c against the silk structural
literature would reach the wrong conclusion about which residues drive it. We
compute both conventions explicitly and report both.

## 9. What we could not do

For much of this work the Supporting Information was the binding constraint.
Wiley returns 402 to automated requests, and a file supplied to us as the SI
turned out to belong to a different paper. Figures 5 and 7 ran against
substitute specimens, and the motif analysis ran on canonical motifs rather
than the paper's own.

That constraint lifted when we found the supplementary tables appended to the
arXiv preprint of the same work, arXiv:2309.10170, pages 31–38. Figure 7 now
runs on the Table S4 motif set against the paper's own BLAST specimens, Figure
5 on its Table S1 sequences, and §3.2's exact check became possible. It is
worth saying plainly that the substitute-specimen versions of those analyses
gave materially different answers — the motif congruence ordering across
property sets was close to reversed — so a reproduction that had stopped at the
substitutes would have reported something wrong.

What remains genuinely out of reach: the BLAST novelty analysis of Section 2.2,
which needs network BLAST or a local database that this project does not run
(Table S2 records the authors' output but does not let us regenerate it); and
the AlphaFold2 structure comparison of Figure 6, which needs compute we did not
spend and which the paper itself limits to visual comparison at pLDDT 40–60.

There is also one thing no amount of supplementary material would fix: a
held-out estimate of forward-task accuracy. That needs retraining with a fold
held out, which we did not do.

## 10. Conclusions

The parts of this paper that can be checked exactly, check out: the released
checkpoint is the model described, it returns the published answer for the
published example, and the dataset and its normalisation reconstruct from
primary sources — the latter verified against a number we did not use in
deriving it.

The headline R² values reproduce in the part that tests the model and not in
the part that tests the search. Given the paper's own designed sequences, the
forward task returns its published R² to four decimals on four of five sets,
matching the entire predicted vector slot for slot. Run end to end, our own
search falls short on all eight property sets by a mean of 0.27, because it did
not find candidates as good as the authors' — not because anything about the
scorer differs.

The more useful finding is about what those values measure. They are maxima
over a sampling budget that is not stated, applied to a model that returns the
exact training label for 71% of its fine-tuning pairs and emits verbatim
training sequences 62% of the time. None of this contradicts the paper, which
is explicit that it trains on all available pairs and reports self-consistency.
But self-consistency between two directions of a model that has largely
memorised its training set is a weaker signal than the same number would be
from a model evaluated on held-out data, and the two are not distinguishable
from the reported figures alone.

Two changes would remove most of the ambiguity, and neither requires new
experiments: state N alongside any best-of-N result, and hold out a fold. On a
dataset of 1,033 pairs, a held-out fold is cheap, and our baselines suggest it
would be informative — the gap between in-sample and cross-validated
performance on this data is the difference between 0.80 and −0.16.

A word on this reproduction's own reliability, since it bears on how much
weight the paragraphs above deserve. Three claims we made during this work were
wrong and were retracted: that three of Table S4's motifs contradicted the
source it cites (an artifact of reading a two-column table as linear text);
that the paper's worked example differed from the database record because of
our own PDF extraction (the transcription was byte-exact — the paper prints two
variants); and that a composition analysis recovered 12 of 14 published
residues (a selection rule that could not fail, where random noise scored
higher). Each was caught by checking against a published quantity that had
played no part in producing the claim — Silkome's own Table 1, a sequence
alignment, a permutation null. Each would have survived internal consistency
checks alone. We report them because a reproduction that lists only its
successes is not much of a reproduction, and because the pattern is the useful
part: the failures were all cases where a number looked right and no external
quantity had been asked to confirm it.

The wider point is not specific to this paper. Generative design pipelines that
chain an inverse task into a forward task and report the agreement are
measuring something real, but what they measure depends on the search budget
and on how much of the training set the model has retained. Both are easy to
report and are usually not.

## Reproducing this work

Every number above traces to a JSON manifest under `results/` recording the git
commit, checkpoint revision, seed, platform and package versions that produced
it. `README.md` gives the run order. `ASSUMPTIONS.md` lists all fifteen choices
the source papers do not specify, with the reasoning and, where available, the
evidence. `docs/discrepancies.md` lists twelve discrepancies and open points.

Total compute: about six GPU-hours on a 6 GB laptop card.
