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

The headline R² values do not fully reproduce. Generating 2,048 candidate
sequences per property set, our maximum falls below the reported value on all
eight sets, by between 0.011 and 0.436 (mean 0.27). We test one candidate
explanation — that the paper's stochastic scoring inflates a
maximum-over-N statistic relative to our deterministic scoring — and reject it:
the effect is +0.0065. The gap remains unexplained.

Four extensions characterise what the pipeline is doing. First, the reported R²
is almost entirely a product of the sampling budget: the median single
candidate has negative R² in six of eight sets, and the expected best-of-N
climbs from roughly −0.4 at N=1 to 0.68 at N=512 with no change to the model.
Second, motif knockouts (poly-A, GGX, GPGXX) shift the forward prediction less
than a scattered deletion of the same size and land between two length-matched
controls, giving no evidence of motif-specific sensitivity. Third, a
composition-only baseline reaches R² 0.80 in-sample and −0.16 cross-validated,
showing that on this dataset in-sample fit and generalisation are nearly
unrelated. Fourth, and connecting the rest: the forward task returns the exact
training label for 71.3% of the fine-tuning pairs, 62.3% of inverse-task
generations are verbatim training sequences, and accuracy collapses from +0.83
to between −0.27 and −0.54 on spidroin families outside MaSp.

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

What remains: a larger effective candidate pool in the original run, a
different random seed, or a selection step not described in the paper or the
notebook. We cannot distinguish these from the information available, so we
record the gap as unexplained rather than attributing it to any of them.

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
spread of 0.117 — a ratio of 0.162. Handed a spidroin it has not memorised, the
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

| | residues selected in ≥2 properties |
|---|---|
| scheme A | A D E G H I K L M N Q R S T V Y |
| scheme B | A D E F G I K L M N Q R S T V |
| **both** | **A D E G I K L M N Q R S T V** |

This follows Supplementary Note 8 of Pandey, Chen & Keten (*Commun. Mater.*
2024) rather than anything in Lu et al., and recovers 12 of the 14 residues
that paper reports under either scheme.

The two schemes exist because the normalisation is *within group*, so the
hydrophobic/polar/charged assignment determines the output — and the source
does not give the assignment table. Rather than pick one and present it as the
result, we ran two standard schemes differing only on the genuinely contested
residues (C, G, H, Y). They disagree on H, Y and F. Neither was chosen to
improve agreement with the published list.

We note this at length because it is the general shape of the problem this
reproduction kept running into: an unstated methodological choice that changes
the numbers, where the honest move is to measure the sensitivity rather than
resolve it by fiat.

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

Figure 7's motif analysis depends on the motif definitions in Table S4 of the
Supporting Information, which we could not obtain. Our motif script falls back
to canonical spidroin motifs and is labelled as a different analysis, not as a
reproduction of Figure 7. Table S1's generated and BLAST-retrieved sequences
are likewise unavailable, so Figure 5 is reproduced on our own generations
against their nearest database neighbours rather than on the paper's specimens.
The BLAST novelty analysis of Section 2.2 needs Table S2 or network BLAST; we
substitute offline k-mer measures and never describe them as alignment
statistics. AlphaFold2 structure comparison was not attempted.

Of these, only Table S4 blocks a headline figure, and supplying it would switch
the analysis from fallback to reproduction with no other change.

## 10. Conclusions

The parts of this paper that can be checked exactly, check out: the released
checkpoint is the model described, it returns the published answer for the
published example, and the dataset and its normalisation reconstruct from
primary sources — the latter verified against a number we did not use in
deriving it.

The headline R² values reproduce only partially. We fall short on all eight
property sets by a mean of 0.27, and we could not explain the gap; the one
mechanism we could test contributes 0.0065 of it.

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
