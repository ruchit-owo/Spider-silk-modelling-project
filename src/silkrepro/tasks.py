"""Prompt construction and output parsing for the three SilkomeGPT tasks.

The model was fine-tuned on a strict text format. Getting a character wrong
here does not raise an error, it just produces a worse-conditioned generation,
so the formats are written once, here, and used everywhere else.

Format (paper, Experimental Section):

    forward   CalculateSilkContent<SEQUENCE>[v0,v1,...,v7]
    inverse   GenerateSilkContent<v0,v1,...,v7>[SEQUENCE]

The released notebook formats the eight values with `:1.3f`, i.e. three decimal
places, and that is what we do too.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from . import config

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def format_property_vector(props: Sequence[float]) -> str:
    """Render an 8D property vector the way the fine-tuning data rendered it."""
    if len(props) != 8:
        raise ValueError(f"expected 8 properties, got {len(props)}")
    return ",".join(f"{v:1.3f}" for v in props)


def forward_prompt(sequence: str) -> str:
    """Prompt asking the model to predict properties for a given sequence."""
    return f"{config.FORWARD_TASK}<{sequence}>"


def inverse_prompt(props: Sequence[float]) -> str:
    """Prompt asking the model to design a sequence hitting a property target."""
    return f"{config.INVERSE_TASK}<{format_property_vector(props)}>"


def solubility_prompt(sequence: str) -> str:
    return f"{config.SOLUBILITY_TASK}<{sequence}>"


def _between_brackets(text: str) -> str | None:
    """Return the contents of the first [...] group, or None if absent.

    The released notebook uses str.find for this and lets a malformed
    generation raise inside a bare `except:`. We return None instead so that
    callers can count parse failures rather than silently dropping them - the
    failure rate is itself a result worth reporting.
    """
    i = text.find("[")
    if i == -1:
        return None
    j = text.find("]", i + 1)
    if j == -1:
        return None
    return text[i + 1 : j]


def parse_property_output(text: str) -> np.ndarray | None:
    """Pull an 8D float vector out of a forward-task generation.

    Returns None if the model produced something unparseable or the wrong
    number of values. Callers must handle None; see ParseStats in metrics.py.
    """
    body = _between_brackets(text)
    if body is None:
        return None
    parts = [p.strip() for p in body.split(",") if p.strip()]
    if len(parts) != 8:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return np.asarray(values, dtype=float)


def parse_sequence_output(text: str) -> str | None:
    """Pull an amino-acid sequence out of an inverse-task generation.

    We additionally require the result to be non-empty and to contain only the
    twenty standard amino-acid letters. The paper does not mention validity
    filtering; we apply it because a string containing digits or brackets is
    not a protein and would silently poison the downstream ProtParam and motif
    analyses. The rejection rate is reported rather than hidden - see
    ASSUMPTIONS.md row A3.
    """
    body = _between_brackets(text)
    if body is None:
        return None
    seq = body.strip()
    if not seq:
        return None
    if not set(seq) <= VALID_AA:
        return None
    return seq


def parse_solubility_output(text: str) -> float | None:
    body = _between_brackets(text)
    if body is None:
        return None
    try:
        return float(body.strip().split(",")[0])
    except ValueError:
        return None


_TASK_RE = re.compile(r"^([A-Za-z]+)<")


def task_of(prompt: str) -> str | None:
    """Which task a prompt string encodes. Used in tests and logging."""
    m = _TASK_RE.match(prompt)
    return m.group(1) if m else None
