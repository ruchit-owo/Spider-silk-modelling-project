"""Loading and calling the released SilkomeGPT checkpoint.

Security note: the checkpoint is published with custom modelling code and must
be loaded with trust_remote_code=True, which executes Python from the Hugging
Face Hub. We pin the revision (see MODEL_REVISION below) so that what we ran is
what a reader gets. If you would rather not execute remote code, there is no
way to run this reproduction - the architecture is not one of the built-in
transformers classes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from . import config, tasks

# Pinned at first download by scripts/00_check_env.py, which writes the
# resolved commit hash here-adjacent (results/model_revision.txt). Set to None
# to track the branch head.
MODEL_REVISION = os.environ.get("SILKOME_REVISION") or None


@dataclass
class GenerationSettings:
    """Decoding settings for one call. Defaults are the released notebook's."""

    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    do_sample: bool = True

    @classmethod
    def forward_default(cls) -> "GenerationSettings":
        """This project's forward-task setting: greedy. See config.FWD_DO_SAMPLE
        for the measured justification and assumption A5."""
        return cls(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=config.FWD_MAX_NEW_TOKENS,
            do_sample=False,
        )

    @classmethod
    def forward_notebook(cls) -> "GenerationSettings":
        """The released notebook's forward-task setting, sampled at T=0.01.
        Used for the sensitivity check, not as the default."""
        return cls(
            temperature=config.FWD_TEMPERATURE,
            top_k=config.FWD_TOP_K,
            top_p=config.FWD_TOP_P,
            max_new_tokens=config.FWD_MAX_NEW_TOKENS,
            do_sample=True,
        )

    @classmethod
    def inverse_default(cls) -> "GenerationSettings":
        return cls(
            temperature=config.GEN_TEMPERATURE,
            top_k=config.GEN_TOP_K,
            top_p=config.GEN_TOP_P,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
        )

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }


class SilkomeGPT:
    """Thin wrapper around the released checkpoint.

    Deliberately thin: the point of this project is to reproduce what the
    authors did, so this class adds batching, seeding and parse accounting but
    changes nothing about how the model is prompted or decoded.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        revision: str | None = MODEL_REVISION,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # float16 on GPU, float32 on CPU. The paper does not state inference
        # precision; ASSUMPTIONS.md row A6 covers this. The forward task was
        # verified deterministic at this precision
        # (scripts/01b_forward_decoding_diagnostics.py), so precision-induced
        # tie-breaking is not silently changing results.
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype

        kw = {"trust_remote_code": True}
        if revision:
            kw["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_name = model_name
        self.revision = revision

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def arch_summary(self) -> dict:
        c = self.model.config
        return {
            k: getattr(c, k, None)
            for k in (
                "num_hidden_layers",
                "num_attention_heads",
                "hidden_size",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "model_type",
            )
        }

    # ------------------------------------------------------------------
    # Raw generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        num_return_sequences: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Sample `num_return_sequences` continuations of `prompt`.

        Returns the full decoded strings, prompt included, exactly as the
        released notebook does. Parsing happens in tasks.py.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor(ids, device=self.device).unsqueeze(0)

        out = self.model.generate(
            inputs=inputs,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            max_new_tokens=settings.max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]

    # ------------------------------------------------------------------
    # Task-level calls
    # ------------------------------------------------------------------

    def predict_properties(
        self,
        sequence: str,
        settings: GenerationSettings | None = None,
        seed: int | None = None,
    ) -> np.ndarray | None:
        """Forward task. Returns an 8D vector, or None if unparseable."""
        settings = settings or GenerationSettings.forward_default()
        text = self.generate(tasks.forward_prompt(sequence), settings, 1, seed)[0]
        # The prompt itself contains no brackets, so the first [...] in the
        # decoded string is the model's answer.
        return tasks.parse_property_output(text[len(tasks.forward_prompt(sequence)) :])

    @torch.no_grad()
    def predict_properties_batch(
        self,
        sequences: list[str],
        settings: GenerationSettings | None = None,
        seed: int | None = None,
        batch_size: int = 16,
    ) -> list[np.ndarray | None]:
        """Forward task over many sequences at once.

        The forward pass dominates the runtime of a full reproduction (one
        64-token autoregressive decode per candidate, and there are thousands
        of candidates), so it is batched.

        Batching requires left-padding, since generation continues from the
        right-hand end of each row. The model uses rotary position embeddings
        and is given an explicit attention mask, so a left-padded row is
        positionally equivalent to the same sequence run alone.

        That equivalence is an assumption about the implementation, not a
        theorem, so it is tested rather than trusted:
        tests/test_batching_equivalence.py checks batched against unbatched
        predictions on real sequences and fails if they differ. Run it after
        any transformers upgrade.
        """
        settings = settings or GenerationSettings.forward_default()
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        out: list[np.ndarray | None] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            prompts = [tasks.forward_prompt(s) for s in chunk]
            encoded = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
            width = max(len(e) for e in encoded)

            input_ids = torch.full(
                (len(encoded), width), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (len(encoded), width), dtype=torch.long, device=self.device
            )
            for i, e in enumerate(encoded):
                input_ids[i, width - len(e) :] = torch.tensor(e, device=self.device)
                attention_mask[i, width - len(e) :] = 1

            gen = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                max_new_tokens=settings.max_new_tokens,
                num_return_sequences=1,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # Only the newly generated tail is decoded, so left padding cannot
            # leak into the parsed answer.
            for row in gen[:, width:]:
                text = self.tokenizer.decode(row, skip_special_tokens=True)
                out.append(tasks.parse_property_output(text))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return out

    def design_sequences(
        self,
        props: Sequence[float],
        n: int,
        settings: GenerationSettings | None = None,
        seed: int | None = None,
        batch_size: int = 16,
    ) -> tuple[list[str], "object"]:
        """Inverse task. Returns (valid sequences, ParseStats).

        Batched because a 6 GB card cannot hold 32 concurrent 512-token
        generations at this hidden size. Batching changes nothing about the
        sampling distribution - each returned sequence is an independent draw.
        """
        from .metrics import ParseStats

        settings = settings or GenerationSettings.inverse_default()
        prompt = tasks.inverse_prompt(props)
        stats = ParseStats()
        seqs: list[str] = []

        remaining = n
        batch_idx = 0
        while remaining > 0:
            k = min(batch_size, remaining)
            batch_seed = None if seed is None else seed + batch_idx
            texts = self.generate(prompt, settings, num_return_sequences=k, seed=batch_seed)
            for t in texts:
                stats.attempted += 1
                tail = t[len(prompt) :] if t.startswith(prompt) else t
                s = tasks.parse_sequence_output(tail)
                if s is None:
                    # Distinguish "no bracketed group at all" from "bracketed
                    # group containing non-amino-acid characters", because they
                    # mean different things about the model's behaviour.
                    raw = tasks._between_brackets(tail)
                    if raw is None:
                        stats.rejected_unparseable += 1
                    else:
                        stats.rejected_invalid_aa += 1
                    continue
                stats.parsed += 1
                seqs.append(s)
            remaining -= k
            batch_idx += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return seqs, stats

    def predict_solubility(
        self, sequence: str, seed: int | None = None
    ) -> float | None:
        s = GenerationSettings(temperature=0.01, top_k=500, top_p=0.9, max_new_tokens=7)
        prompt = tasks.solubility_prompt(sequence)
        text = self.generate(prompt, s, 1, seed)[0]
        return tasks.parse_solubility_output(text[len(prompt) :])
