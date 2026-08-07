# Inputs that must be supplied by hand

Three inputs this project needs are not redistributable and are not in the
authors' GitHub repository. Nothing in the code substitutes or approximates
them: if a file is missing, the scripts that need it stop and record
`not run: input missing`, and the affected claims are marked as not reproduced
in `docs/discrepancies.md`.

## 1. `silkome_masp_pairs.csv` — the 1,033 sequence/property pairs

The paper fine-tunes on 1,033 MaSp sequences paired with four fiber-level
mechanical properties and their standard deviations. These come from the
Silkome dataset:

> K. Arakawa et al., "1000 spider silkomes: Linking sequences to silk physical
> properties", *Sci. Adv.* **8**, eabo6043 (2022).

Download the supplementary data from that paper and build a CSV with exactly
these columns:

| column | meaning | units |
|---|---|---|
| `sequence` | amino-acid sequence, one-letter code, no gaps | — |
| `toughness` | fiber toughness | raw, as published |
| `toughness_sd` | standard deviation across measurements | raw |
| `E` | elastic modulus | raw |
| `E_sd` | standard deviation | raw |
| `strength` | tensile strength | raw |
| `strength_sd` | standard deviation | raw |
| `strain` | strain at break | raw |
| `strain_sd` | standard deviation | raw |

Keep raw units. The normalisation is applied by `silkrepro.dataio` using the
constants below, so that we can check the result rather than inherit it.

**Known difficulty.** The paper says the dataset was "constructed and curated
based on the silkome dataset" but does not give the curation rule that takes
the published silkome tables to exactly 1,033 rows. If your assembled table
has a different row count, record it — `scripts/03_dataset_audit.py` reports
the count and the difference is a legitimate finding, not something to force
into agreement.

## 2. `table_s5_normalisation.json` — the min/max scaling constants

Table S5 of the paper's Supporting Information. Transcribe as:

```json
{
  "toughness":    [min, max],
  "toughness_sd": [min, max],
  "E":            [min, max],
  "E_sd":         [min, max],
  "strength":     [min, max],
  "strength_sd":  [min, max],
  "strain":       [min, max],
  "strain_sd":    [min, max]
}
```

These matter more than they look. The Experimental Section states the min and
max were taken "across the entire dataset (not limited to MaSp sequences)", so
they cannot be recovered from the 1,033 MaSp rows alone. Without this file,
`dataio.derive_normalisation` can produce MaSp-only constants, but every
downstream number then sits on a different scale from the paper's and is
labelled `normalisation_source = derived_from_masp_subset`.

## 3. `table_s4_motifs.csv` — the motif definitions for Figure 7

Table S4 of the Supporting Information, itself derived from Table 1 of the
Silkome paper. Columns:

| column | meaning |
|---|---|
| `motif_id` | the paper's label, e.g. `T_neg_1`, `E_pos_2`, `SB_pos_2` |
| `pattern` | the motif as a literal string or a Python regex |
| `property` | `toughness`, `E`, `strength` or `strain` |
| `direction` | `pos` or `neg` |

Without this file the motif analysis falls back to canonical spidroin motifs
(poly-A, GGX, GPGXX, ...), which is a different analysis and is labelled
`motif_source = canonical_fallback`. It is not a reproduction of Figure 7.

## Optional: `table_s1_sequences.csv`

Table S1 lists the five generated sequences the paper analysed and the ten
BLAST-retrieved natural sequences it compared them against. Supplying it lets
`scripts/06_compare_to_published_sequences.py` compare our generations against
the published ones directly. Columns: `name`, `set`, `role`
(`generated` / `blast1` / `blast2`), `sequence`, `accession` (optional).
