# Study notes (LaTeX)

Self-contained notes explaining the SilkomeGPT paper and this reproduction from
scratch, assuming no prior background in either machine learning or protein
biochemistry.

## Building

There is no LaTeX toolchain on the machine these were written on, so the source
has **not been compiled**. It passes a structural check (balanced braces and
environments, no dangling `\ref`, no duplicate labels, every `\input` and every
image present), but a first compile may still surface a stray warning.

### Overleaf (easiest)

1. Zip `main.tex`, `parts/` and `figures/` together.
2. New Project → Upload Project.
3. Set the compiler to **pdfLaTeX**, and compile twice so the table of
   contents and cross-references resolve.

### Locally

Install MiKTeX or TeX Live, then:

```bash
pdflatex main.tex && pdflatex main.tex
```

Only standard packages are used: `geometry`, `amsmath`, `graphicx`, `booktabs`,
`longtable`, `enumitem`, `xcolor`, `caption`, `fancyhdr`, `listings`, `tikz`,
`pgfplots`, `hyperref`, `microtype`, `lmodern`.

## Layout

| file | contents |
|---|---|
| `main.tex` | preamble, callout-box definitions, document skeleton |
| `parts/00_preface.tex` | what the project was, how to read the notes, the roadmap figure |
| `parts/01_biology.tex` | proteins, silk mechanics, spidroins, the Silkome dataset |
| `parts/02_transformers.tex` | language models, tokenisation, attention, decoding |
| `parts/03_metrics.tex` | R², the within-vector trap, best-of-N, in-sample vs held-out |
| `parts/04_thepaper.tex` | what Lu et al. did: tasks, targets, figures, claims |
| `parts/05_reproduction.tex` | rebuilding it, and the exact-match result |
| `parts/06_extensions.tex` | the four extensions, and every generated figure explained |
| `parts/07_mistakes.tex` | the retractions, and the pattern across them |
| `parts/08_reference.tex` | conclusions, assumption and discrepancy tables, glossary |

Figures are of two kinds: TikZ/pgfplots diagrams drawn in the source (editable,
resolution-independent) and the six result figures from `../figures/`, copied
into `figures/` so this directory builds on its own.
