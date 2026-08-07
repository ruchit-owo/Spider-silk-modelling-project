# Methods

What the paper does, and what this repository does about it. Written so the
code can be followed without going back to the PDF.

## 1. The model

SilkomeGPT is an autoregressive decoder-only transformer in the GPT-NeoX family,
with rotary positional embeddings. The paper states 12 layers, 8 attention
heads, hidden size 1024, intermediate size 4096, and 253.6M parameters. Its
tokenizer is a custom byte-pair encoder with up to 50,000 tokens, trained on
about 800,000 protein sequences.

Training happened in two stages:

**Pretraining.** Next-token prediction over roughly 600,000 protein sequences up
to 600 residues, drawn from AlphaFold2 proteome predictions across many
organisms plus Swiss-Prot, including the silkome sequences. Batch size 48 with
4 gradient accumulation steps, linear LR schedule with warmup, Adam, LR 2e-4.

**Fine-tuning.** Three tasks, for sequences up to 768 residues:

| task | prompt | completion |
|---|---|---|
| `Sequence` | — | a silk sequence |
| `CalculateSilkContent` | `<SEQUENCE>` | `[v0,...,v7]` |
| `GenerateSilkContent` | `<v0,...,v7>` | `[SEQUENCE]` |

The paper states that fine-tuning used *all* known sequence/property pairs, so
there is no held-out split.

We do not retrain anything. All results here use the released checkpoint,
`lamm-mit/SilkomeGPT`, whose architecture `scripts/00_check_env.py` verifies
against the four stated hyperparameters and the parameter count.

## 2. The 8D property vector

Equation (1) of the paper:

```
[toughness, SD(toughness), E, SD(E), strength, SD(strength), strain, SD(strain)]
```

Order matters — nothing in the prompt identifies which slot is which, so the
model learned the positions. Four of the eight entries are standard deviations
across repeated fibre measurements, included because each fibre property is
associated with a set of spidroin sequences rather than one, and the SD carries
information about that spread.

All eight are min-max normalised to [0, 1]. The paper takes the extremes
"across the entire dataset (not limited to MaSp sequences)" and tabulates them
in Table S5.

Half the vector being standard deviations has a consequence for reading R²:
the SD slots vary less than the properties, so a model that fits them well and
the properties poorly still scores respectably over all eight. We therefore
report the four mechanical slots separately throughout
(`config.MECHANICAL_IDX`).

## 3. The dataset

1,033 MaSp sequence/property pairs, "constructed and curated based on the
silkome dataset" — the curation rule is not given, and the pairs are not in the
authors' repository.

We reconstruct it from the Silkome v1 primary release (Arakawa et al.,
*Sci. Adv.* 2022). `src/silkrepro/silkome.py` parses the protein FASTA, whose
headers are `seq_id|idv_id|family|genus|species|type|subtype|region`, and joins
to `mechanical_properties.csv` on `idv_id` — the individual spider whose fibre
was tested. Keeping rows with all eight properties present and a `type`
beginning with `MaSp` gives exactly 1,033 rows. See assumption A1.

Three properties of the reconstructed set worth knowing:

- 1,033 rows but 1,028 distinct sequences; five sequences carry two different
  label vectors, from two individuals whose fibres tested differently (A12).
- Records are terminal-domain-anchored (553 CTD, 480 NTD), not complete
  spidroins.
- Sequence lengths run 115–1,854, median 378. The authors' `ALL_SILK_SEQ.csv`
  is the same FASTA filtered to ≤768 residues, matching the fine-tuning limit.

## 4. The cycle-consistency procedure

This is what Table 1 and Figures 3–4 measure:

```
for each 8D target t:
    sample N sequences from GenerateSilkContent<t>
    discard those that do not parse as protein
    discard those that exactly match a known silk sequence
    for each survivor s:
        p = CalculateSilkContent<s>
        score R2(t, p) across the eight entries
    report max(R2)
```

Two features of this procedure govern how the reported numbers should be read,
and neither is stated in the paper's main text.

**The score is within-vector.** `r2_score(t, p)` with both arguments of length
8. The denominator is the variance of the eight target values, so targets whose
entries cluster tightly give a brittle R². We report `target_spread` alongside
every such R².

**The reported value is a maximum over N.** The paper says results "with the
highest R² values are selected from a set of sampling attempts" without giving
N. The released notebook's `generate_new_and_find_best` uses 32 samples per step
× 64 repeats = 2,048, then `np.argmax`. A maximum over N draws grows with N
without any change to the model.

## 5. Decoding settings

Not in the paper; read from the released notebook.

| | temperature | top_k | top_p | max_new_tokens |
|---|---|---|---|---|
| inverse (`GenerateSilkContent`) | 1.25 | 500 | 0.8 | 512 |
| forward (`CalculateSilkContent`) | 0.01 | 500 | 0.9 | 64 |

We keep the inverse settings verbatim. For the forward task we decode greedily
instead, for reasons measured in
`scripts/01b_forward_decoding_diagnostics.py` and recorded as assumption A5:
greedy agrees with the sampled setting on 58/60 sequences, parses 60/60 at the
same token budget, and — unlike sampling — is invariant to batch composition,
which matters because we batch the forward pass for speed. Table 1 is also
re-run under the notebook's sampled setting as a sensitivity check
(`--faithful-sampling`).

## 6. Downstream analyses

**Novelty** (§2.2 of the paper). Exact string membership in `ALL_SILK_SEQ.csv`,
then BLAST query cover and percent identity against wider protein databases.
We reproduce the exact-match test verbatim. The BLAST step needs Table S2,
which we could not obtain, so we substitute offline 6-mer Jaccard and
containment measures — clearly labelled, and never described as alignment
statistics (A4, D7).

**Protein descriptors** (Figure 5). Molecular weight, instability index and
isoelectric point via Biopython ProtParam, plus amino-acid composition and
secondary-structure fractions. The structure fractions need care: Biopython
v1.82 corrected a documented label swap in the method the paper used, so the
"helix" and "sheet" labels in Figure 5c are exchanged relative to current
Biopython. We compute both conventions explicitly and report both. See D1.

**Motifs** (Figure 7). Counts and positional distributions for motifs defined
in Table S4. We do not have Table S4, so this analysis runs on canonical
spidroin motifs and is labelled `canonical_fallback` — a different analysis,
not a reproduction. See A8 and D7.

**Molecular structure** (Figure 6). AlphaFold2 folding of generated sequences.
Not attempted here; it needs AF2 compute and the paper itself limits the claim
to visual comparison at pLDDT 40–60.

## 7. Extensions

| extension | script | question |
|---|---|---|
| best-of-N | `10_ablation_bestofn.py` | how much of the reported R² is the search rather than the model? |
| sequence content | `11_ablation_sequence_content.py` | what does the forward task read from the sequence? |
| composition baseline | `12_baseline_composition.py` | how far do amino-acid counts alone get you? |
| out-of-distribution | `13_extension_nonmasp.py` | does the prediction depend on the spidroin family at all? |

The sequence-content ablations are always reported against a measured noise
floor: the forward task's own spread when the identical prompt is resubmitted.
An effect smaller than that floor is not evidence of anything, and the code
computes the floor before it computes any effect.

## 8. Validation

Every metric and operator is tested against something external rather than
against itself:

- within-vector R² against `sklearn.metrics.r2_score` on random vectors;
- prompt strings character-by-character against the examples printed in the
  paper's Experimental Section;
- the secondary-structure groupings against the Figure 5 caption *and* against
  the installed Biopython, so an upstream change fails a test rather than
  silently changing a figure;
- batched inference against unbatched, on sequences of unequal length;
- the whole forward path against the paper's worked example, which has a known
  correct answer;
- ablation operators against their own preservation claims (shuffle preserves
  composition exactly, termini and core partition the sequence, and so on);
- the transcribed Table 1 constants against the paper's prose descriptions of
  the same sets, which caught the S3 inconsistency recorded as D2.
