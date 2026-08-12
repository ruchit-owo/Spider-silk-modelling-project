# Reproducing SilkomeGPT, and asking what its numbers mean

A reproduction of

> W. Lu, D. L. Kaplan, M. J. Buehler, "Generative Modeling, Design, and
> Analysis of Spider Silk Protein Sequences for Enhanced Mechanical
> Properties", *Adv. Funct. Mater.* **34**, 2311324 (2024).
> [doi:10.1002/adfm.202311324](https://doi.org/10.1002/adfm.202311324)

together with four extensions that test what the reported numbers depend on.

The paper fine-tunes a 253.6M-parameter GPT-NeoX-style transformer on 1,033
major ampullate spidroin (MaSp) sequences paired with fibre-level mechanical
properties, and uses it in both directions: a forward task that predicts eight
property values from a sequence, and an inverse task that designs a sequence to
hit eight target values. The two are chained into a cycle-consistency check,
and the agreement between requested and predicted properties is reported as
R².

This repository re-runs that pipeline against the authors' released checkpoint,
reconstructs the training dataset from primary sources, and then asks four
questions the paper leaves open.

---

## What was reproduced, and what it cost to get there

**Confirmed exactly.**

- The released checkpoint matches the architecture the paper states: 12 layers,
  8 attention heads, hidden size 1024, intermediate size 4096, and
  **253,556,736 parameters** against the paper's stated 253.6M.
- The forward task, given the worked example printed in the Experimental
  Section, returns `[0.327, 0.356, 0.261, 0.287, 0.437, 0.190, 0.220, 0.301]` —
  exactly the vector the paper pairs with that sequence.
- The 1,033-pair dataset was **reconstructed from primary sources** and lands
  on 1,033 rows on the nose (assumption A1).
- The normalisation constants of Table S5, which we could not obtain, were
  **recovered independently and verified** against the paper's own published
  values to within 0.0005 on all eight properties (assumption A2).

- **The forward task reproduces the published R² exactly.** Given the paper's
  own generated sequences (Table S1), four of the five self-consistency sets
  return the published R² to four decimals, with the full eight-value
  predicted vector identical slot for slot.
- **Figure 7 is reproduced** using the paper's Table S4 motif set, against its
  own BLAST comparison sequences.

**Reproduced with the number restated.** The headline R² values are maxima over
a pool of sampled candidates whose size the paper does not give. The released
notebook's pool is 2,048 per property set. Our own search does not find
candidates as good as theirs — see `RESULTS.md` §8 and §8g.

Everything above is set out in detail in **[`RESULTS.md`](RESULTS.md)**, with
each claim tagged as *measured*, *derived* or *assumed*.

## The four extensions

1. **Best-of-N selection** (`scripts/10_ablation_bestofn.py`). The reported R²
   is a maximum over N sampled candidates. We sweep N and report the whole
   curve, the single-sample distribution, and the number of samples needed to
   reach the published value.
2. **Sequence-content ablations** (`scripts/11_ablation_sequence_content.py`).
   Shuffle, block-shuffle, reverse, motif knockouts with length-matched
   controls, terminal-vs-core splits, and a random-sequence control — each
   measured against the forward task's own repeat noise floor.
3. **Composition-only baseline** (`scripts/12_baseline_composition.py`). Ridge
   and gradient boosting on amino-acid, dipeptide and motif features, scored
   both in-sample (comparable to the transformer, which was trained on
   everything) and cross-validated (honest).
4. **Out-of-distribution spidroins** (`scripts/13_extension_nonmasp.py`). The
   model is MaSp-only; the database has eight other families. Does the
   prediction actually depend on which family it is given?

## Two things to understand before reading any R² in this repository

**The paper's R² is computed within one 8-vector, not across a test set.** It
is `r2_score(target_8vector, predicted_8vector)` — one designed sequence, eight
predicted values against the eight requested. That is a self-consistency
measure, and the paper describes it as such. It is not a generalisation
estimate, and "R²" in a property-prediction context usually means the latter,
so it is easy to misread. `silkrepro.metrics` implements both definitions under
distinct names and never mixes them.

**There is no held-out set.** The Experimental Section states that fine-tuning
used "all known pairs of sequence and properties". Every forward-task accuracy
number measurable against the 1,033 pairs — including ours — is therefore a
training-set number and an upper bound, not an estimate of performance on a new
spidroin.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
```

Then fetch the authors' repository for the novelty reference set:

```bash
git clone https://github.com/lamm-mit/SilkomeGPT _ref/SilkomeGPT-main
```

And the Silkome primary data (Arakawa et al., *Sci. Adv.* **8**, eabo6043,
2022): `spider-silkome-database.v1.prot.fasta` and
`mechanical_properties.csv`.

> **Note on `trust_remote_code`.** The checkpoint ships custom modelling code
> and must be loaded with `trust_remote_code=True`, which executes Python from
> the Hugging Face Hub. `scripts/00_check_env.py` records the resolved commit
> into `results/model_revision.txt`; set `SILKOME_REVISION` to pin it.

## Running it

```bash
python scripts/00_check_env.py                     # download, verify architecture
python scripts/01_smoke_test.py                    # both tasks, end to end
python scripts/01b_forward_decoding_diagnostics.py # decides assumption A5
python scripts/03_build_dataset.py --fasta <fasta> --mech <csv>
python scripts/02_reproduce_table1.py --n 2048     # the long one, ~3 h on a 6 GB GPU
python scripts/04_forward_eval_dataset.py
python scripts/05_protein_properties.py
python scripts/06_motif_analysis.py
python scripts/10_ablation_bestofn.py              # CPU only
python scripts/11_ablation_sequence_content.py
python scripts/12_baseline_composition.py          # CPU only
python scripts/13_extension_nonmasp.py --fasta <fasta> --mech <csv>
python scripts/99_make_figures.py                  # CPU only
```

`scripts/02_reproduce_table1.py` resumes at property-set granularity: each set
is written when it finishes, so an interrupt loses only the set in progress,
not the whole run. `--n` counts *generation attempts*, matching the notebook's
32 × 64, and the attempt count is persisted separately from the parsed rows —
about 5% of attempts never parse, so resuming on row count alone would grow the
pool past the budget on every re-run and inflate the maximum. Pass `--n 128`
for a quick pass.

## Layout

```
src/silkrepro/      library code; every constant tagged PAPER / CODE / OURS
  config.py           the paper's numbers, transcribed and tested
  tasks.py            prompt formats and output parsing
  model.py            checkpoint wrapper, batched inference
  metrics.py          both R² definitions, best-of-N, parse accounting
  silkome.py          reading the Silkome v1 release
  dataio.py           dataset and normalisation loading
  ablations.py        the ablation operators and the noise floor
  baseline.py         composition-feature regressors
  novelty.py          exact-match novelty, plus k-mer measures
  protparam.py        Figure 5 descriptors, both structure conventions
  motifs.py           motif counting and positions
  runinfo.py          run manifests
scripts/            one script per analysis; all write manifests
tests/              66 tests (63 without `-m slow`, which needs the checkpoint)
docs/               methods and discrepancies
data/raw/           hand-supplied inputs (see its README)
results/            JSON + CSV outputs, each with provenance
figures/            generated figures
```

## Documents

| file | contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every measured number, with measured/derived/assumed tags |
| [`ASSUMPTIONS.md`](ASSUMPTIONS.md) | all 15 choices the paper does not specify, with reasoning and evidence |
| [`docs/discrepancies.md`](docs/discrepancies.md) | 15 discrepancies and open points, stated neutrally |
| [`docs/methods.md`](docs/methods.md) | what the paper does, in enough detail to follow the code |
| [`MANUSCRIPT.md`](MANUSCRIPT.md) | the write-up |

## Reproducibility

Every script writes a JSON manifest recording the git commit, working-tree
cleanliness, model revision, seed, platform, GPU, and the versions of torch,
transformers, numpy, pandas, scikit-learn and Biopython. Any number quoted in
the write-up traces back to one of these.

Two things are deliberately *not* reproducible from this repository alone: the
Silkome primary data and the authors' checkpoint, neither of which is ours to
redistribute. Both are freely available and `data/raw/README.md` says exactly
what to fetch.

## Note on scope

This project evaluates a published model against the data it was trained on. It
makes no claim about the mechanical behaviour of real spider silk, and none of
the generated sequences here have been synthesised or tested. The paper is
equally clear about this in its own limitations section.
