# Reproduction of a generative transformer for spider silk protein design

## Summary

This project reproduces and analyses the model of Lu, Kaplan and Buehler
(*Advanced Functional Materials*, 2024), which fine-tunes a 253.6 million
parameter language model (SilkomeGPT) on major ampullate spidroin sequences to
predict fibre-level mechanical properties and to design sequences for target
properties. The work uses the authors' released checkpoint. It rebuilds the
training data and the normalisation constants from primary sources, repeats the
reported design procedure, and adds several analyses to characterise what the
model does.

## What was reproduced

- **Model.** The released checkpoint matches the reported architecture and has
  253,556,736 parameters, agreeing with the stated 253.6 million. Submitting the
  worked-example sequence printed in the paper returns the published property
  vector on all eight entries.
- **Dataset.** The paper does not give its construction rule. Joining the two
  Silkome files on the individual-spider identifier, keeping complete records
  whose type is MaSp, yields exactly the published count of 1,033 sequences.
- **Normalisation.** The min/max constants, initially unavailable, were recovered
  from the data to within 0.0005 of a published value, and later matched the
  supplementary table exactly on all eight properties.
- **Forward task.** Given the authors' own generated sequences, the forward task
  returns the published coefficient of determination to four decimal places on
  four of the five self-consistency targets, with the full predicted vector
  matching entry for entry.

## Where the reproduction is partial

Running the full design procedure with the reported sampling budget, the best
selected candidate falls below the published value on all eight targets, with a
mean shortfall of 0.27. Because the forward task reproduces exactly on the
authors' sequences, this difference lies in the inverse-task search, which did
not locate candidates as strong as the authors', rather than in the model or its
scoring. One likely explanation, that our deterministic scoring differs from the
authors' sampled scoring, was tested and accounts for only 0.007 of the gap.

## Additional findings

Four analyses that were not in the original paper describe the fine-tuned model:

- The reported value is a maximum over sampled candidates. The typical single
  candidate has a negative coefficient of determination on six of eight targets,
  so the reported figure depends strongly on the number of samples drawn, which
  the paper does not state.
- Removing mechanically relevant sequence motifs (poly-alanine, GGX, GPGXX)
  changes the forward prediction no more than removing the same number of
  residues at random positions, so the model shows no motif-specific response in
  this test.
- The forward task returns the exact training label for 71 percent of the
  fine-tuning sequences, and the inverse task copies a training sequence in 62
  percent of generations.
- Prediction accuracy is high on the training family (MaSp) and negative on the
  seven other spidroin families, and the prediction varies little with family.

These observations are consistent with the model behaving largely as a retrieval
system over its fine-tuning set. This does not contradict the paper, which states
that it trains on all available pairs and reports self-consistency rather than
held-out accuracy. It does indicate that the reported number should be read
narrowly. Stating the sampling budget and holding out a fraction of the data for
evaluation would resolve most of the ambiguity, and neither requires new
experiments.

## Deliverables

- A code repository with one script per analysis, a test suite, and a
  per-run record of software versions, random seed and model revision.
- A reproduction manuscript with all figures and tables.
- Documented lists of every assumption made where the paper is unspecified, and
  every point where results differ from the paper or the paper is internally
  inconsistent, each stated neutrally.

The model checkpoint and the Silkome primary data are available from their
original sources and are not redistributed.
