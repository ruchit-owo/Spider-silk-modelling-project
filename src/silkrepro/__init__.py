"""Reproduction and extension of Lu, Kaplan & Buehler (2024), SilkomeGPT.

W. Lu, D. L. Kaplan, M. J. Buehler, "Generative Modeling, Design, and Analysis
of Spider Silk Protein Sequences for Enhanced Mechanical Properties",
Adv. Funct. Mater. 2024, 34, 2311324. DOI 10.1002/adfm.202311324
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    ablations,
    baseline,
    config,
    dataio,
    metrics,
    motifs,
    novelty,
    paper_examples,
    protparam,
    runinfo,
    silkome,
    tasks,
)
