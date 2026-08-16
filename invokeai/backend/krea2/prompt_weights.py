"""Per-token prompt weighting for Krea-2.

Krea-2 encodes prompts with Qwen3-VL, a causal LLM. Scaling its hidden states the way compel scales
CLIP embeddings does nothing useful: the transformer pushes those states through ``text_fusion`` and a
12->1 linear projector before they ever reach the image stream. Weighting therefore has to happen inside
the denoiser's attention instead (see ``invokeai.backend.krea2.attention``):

* weight <= 1 scales the token's **value** vector -- ``v_factor = 1 + strength * (weight - 1)``. At
  weight 0 the token contributes nothing; below 0 it subtracts.
* weight > 1 adds a bias to the token's **key** column before the softmax -- ``k_bias =
  2 * strength * (weight - 1)`` -- so the rest of the sequence attends to it more.

This module owns the text-side half: parsing the markers out of the prompt and turning them into a
per-token weight vector aligned with the encoder's token axis.

**Why not compel's parser.** compel collapses newlines and repeated spaces when it flattens a prompt
(``"a.\\n\\nb"`` -> ``"a. b"``). Krea-2 prompts are long natural-language paragraphs, so reusing it would
silently change the *unweighted* text as soon as weighting was enabled. compel also cannot express
negative weights (``(cat)-1`` parses as ``(cat)-`` plus the literal text ``1``). The parser here keeps
the prompt byte-for-byte outside the markers it removes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from invokeai.backend.util.logging import InvokeAILogger

# Bounds on an authored weight. k_bias is unbounded pre-softmax -- weight 3.0 at strength 1.0 already
# gives a token ~55x the attention mass -- so anything past this band is far more likely to be a stray
# number in prose (e.g. "(scale: 1:200)") than an intended weight. Out-of-band matches are emitted as
# literal text rather than clamped, so natural language is never silently reinterpreted as emphasis.
MIN_WEIGHT = -2.0
MAX_WEIGHT = 3.0

_NEUTRAL_WEIGHT = 1.0

# One pass over escapes and both marker notations. A phrase may not contain unescaped parentheses, so
# nested markers are not supported and a plain parenthetical like "(a stone bridge)" -- no trailing
# number -- is left untouched as ordinary prose.
_PHRASE = r"[^()\\]*(?:\\[()\\][^()\\]*)*"
_WEIGHT = r"-?\d+(?:\.\d+)?"
_MARKER_RE = re.compile(
    rf"""
      (?P<escape>\\[()\\])
    | \( (?P<colon_phrase>{_PHRASE}) : \s* (?P<colon_weight>{_WEIGHT}) \s* \)
    | \( (?P<suffix_phrase>{_PHRASE}) \) \s* (?P<suffix_weight>{_WEIGHT})
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class WeightSpan:
    """A weighted phrase, addressed by character offsets into the *clean* prompt."""

    start: int
    end: int
    weight: float
    text: str


def _unescape(text: str) -> str:
    return re.sub(r"\\([()\\])", r"\1", text)


def parse_weighted_prompt(prompt: str) -> tuple[str, list[WeightSpan]]:
    """Split ``prompt`` into the text to encode and the weighted spans within it.

    Two equivalent notations are accepted, so prompts copied from either ComfyUI or InvokeAI's
    Stable Diffusion prompt boxes work unchanged::

        (blonde hair:-1)     (blonde hair)-1     ->  weight -1.0
        (red scarf:1.8)      (red scarf)1.8      ->  weight  1.8

    ``\\(`` and ``\\)`` escape literal parentheses (and ``\\\\`` a literal backslash); every other
    character, **including newlines and runs of spaces**, is preserved exactly. Parentheses that are not
    followed by a weight are ordinary text. Nesting is not supported.

    Returns the clean prompt and the spans, whose offsets index into that clean prompt. Spans are
    non-overlapping by construction, so each marked occurrence is weighted independently -- marking one
    of two identical words weights only that one.
    """
    logger = InvokeAILogger.get_logger()

    if not _MARKER_RE.search(prompt):
        return prompt, []

    parts: list[str] = []
    spans: list[WeightSpan] = []
    clean_len = 0
    pos = 0

    def emit(text: str) -> int:
        nonlocal clean_len
        start = clean_len
        if text:
            parts.append(text)
            clean_len += len(text)
        return start

    for match in _MARKER_RE.finditer(prompt):
        emit(prompt[pos : match.start()])
        pos = match.end()

        escape = match.group("escape")
        if escape is not None:
            emit(escape[1])
            continue

        phrase = match.group("colon_phrase")
        raw_weight = match.group("colon_weight")
        if phrase is None:
            phrase = match.group("suffix_phrase")
            raw_weight = match.group("suffix_weight")

        weight = float(raw_weight)
        phrase_text = _unescape(phrase)

        if not (MIN_WEIGHT <= weight <= MAX_WEIGHT):
            logger.warning(
                f"Krea-2 prompt weight {weight} in '{match.group(0)}' is outside the supported range "
                f"[{MIN_WEIGHT}, {MAX_WEIGHT}]; treating it as literal text."
            )
            emit(match.group(0))
            continue

        if not phrase_text.strip():
            logger.warning(f"Krea-2 prompt weight '{match.group(0)}' has an empty phrase; ignoring it.")
            continue

        start = emit(phrase_text)
        if weight != _NEUTRAL_WEIGHT:
            spans.append(WeightSpan(start=start, end=clean_len, weight=weight, text=phrase_text))

    emit(prompt[pos:])
    return "".join(parts), spans


def build_token_weights(
    offset_mapping: torch.Tensor | Sequence[tuple[int, int]],
    spans: Iterable[WeightSpan],
    char_offset: int,
) -> torch.Tensor | None:
    """Map character spans onto token positions via the tokenizer's offset mapping.

    ``offset_mapping`` is the ``(seq_len, 2)`` mapping for the tokenized *body* text, and ``char_offset``
    is the number of characters that precede the clean prompt in that body (i.e. the length of the
    Krea-2 system-prompt prefix). A token is weighted when its character range overlaps the span at all,
    which handles BPE merges straddling a phrase boundary without any of the tokenizer-specific
    heuristics an id-subsequence search needs.

    Returns a ``(seq_len,)`` float32 tensor -- neutral 1.0 everywhere else -- or ``None`` when no span
    matched a token. Never raises: a marker that resolves to nothing is a warning, not a failed
    generation.
    """
    logger = InvokeAILogger.get_logger()

    offsets = torch.as_tensor(offset_mapping, dtype=torch.long)
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError(f"offset_mapping must have shape (seq_len, 2), got {tuple(offsets.shape)}.")

    starts = offsets[:, 0]
    ends = offsets[:, 1]
    # Special and padding tokens map to an empty (0, 0) range and must never be weighted.
    is_real = ends > starts

    weights = torch.full((offsets.shape[0],), _NEUTRAL_WEIGHT, dtype=torch.float32)
    matched = False
    for span in spans:
        low = span.start + char_offset
        high = span.end + char_offset
        selected = is_real & (starts < high) & (ends > low)
        if not bool(selected.any()):
            logger.warning(
                f"Krea-2 prompt weight for '{span.text}' matched no tokens (the phrase is empty or falls "
                "beyond the prompt truncation limit); ignoring it."
            )
            continue
        weights[selected] = span.weight
        matched = True

    return weights if matched else None


def derive_attention_weights(token_weights: torch.Tensor, strength: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split per-token weights into the value scale and key bias the attention processor applies.

    Mirrors the reference ComfyUI node: de-emphasis acts on the value vector (and turns subtractive
    below zero), emphasis acts on the pre-softmax attention logits. A neutral weight of 1.0 maps to
    exactly ``(1.0, 0.0)``, so unweighted tokens are untouched bit-for-bit.
    """
    ones = torch.ones_like(token_weights)
    zeros = torch.zeros_like(token_weights)
    deviation = token_weights - ones
    value_scale = torch.where(token_weights <= _NEUTRAL_WEIGHT, ones + strength * deviation, ones)
    key_bias = torch.where(token_weights > _NEUTRAL_WEIGHT, 2.0 * strength * deviation, zeros)
    return value_scale, key_bias
