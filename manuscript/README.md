# Manuscript

A reproduction report of the SilkomeGPT paper, written as a standalone
manuscript. Not compiled here (no LaTeX toolchain on this machine); it passes a
structural check for balanced environments, resolved references and citations,
present figures and balanced braces.

## Build

Upload `manuscript.tex` and `figures/` to Overleaf, set the compiler to
pdfLaTeX, and compile twice. Or locally:

    pdflatex manuscript.tex && pdflatex manuscript.tex

Standard packages only: geometry, amsmath, graphicx, booktabs, array, caption,
enumitem, hyperref, microtype, lmodern.

## Figures

The six result figures live in `figures/`, copied from the repository
`figures/` directory so this folder builds on its own.
