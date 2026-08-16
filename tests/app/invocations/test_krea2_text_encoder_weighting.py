"""Tests for the `token_weighting` switch on the Krea-2 prompt node.

The plumbing that carries weights through denoise lives in ``tests/app/invocations/test_krea2_denoise.py``
and ``tests/backend/krea2``; this file covers the encoder-side mapping from prompt markers to token
positions, which is where an off-by-N would put emphasis on the wrong words.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.krea2_text_encoder import Krea2TextEncoderInvocation
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField
from invokeai.backend.krea2.sampling_utils import KREA2_START_IDX
from invokeai.backend.krea2.text_encoding import KREA2_PREFIX, KREA2_SUFFIX
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType

_SUFFIX_TOKEN_COUNT = 5


def _identifier(key: str, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key, hash=f"hash-{key}", name=key, base=BaseModelType.Krea2, type=model_type, submodel_type=None
    )


class _WordTokenizer:
    """Whitespace tokenizer with real character offsets, standing in for Qwen2TokenizerFast."""

    is_fast = True

    def __init__(self) -> None:
        self.body_texts: list[str] = []

    def __call__(
        self,
        text,
        max_length=None,
        truncation=False,
        padding=None,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        if text == KREA2_SUFFIX:
            input_ids = torch.arange(91, 91 + _SUFFIX_TOKEN_COUNT, dtype=torch.long).unsqueeze(0)
            return SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        self.body_texts.append(text)
        offsets = []
        cursor = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
        offsets = offsets[:max_length]

        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, : len(offsets)] = torch.arange(1, len(offsets) + 1, dtype=torch.long)
        attention_mask[:, : len(offsets)] = 1
        # Padding maps to the empty (0, 0) range, exactly as a real fast tokenizer reports it.
        offset_mapping = torch.zeros((1, max_length, 2), dtype=torch.long)
        offset_mapping[0, : len(offsets)] = torch.tensor(offsets, dtype=torch.long)

        result = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        if return_offsets_mapping:
            result.offset_mapping = offset_mapping
        return result


class _StubEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, input_ids, attention_mask, position_ids, **_kwargs):
        seq_len = input_ids.shape[1]
        return SimpleNamespace(hidden_states=tuple(torch.zeros((1, seq_len, 4)) for _ in range(36)))


def _run_encode(monkeypatch: pytest.MonkeyPatch, prompt: str, *, token_weighting: bool = True):
    tokenizer = _WordTokenizer()
    encoder = _StubEncoder()

    class _TokenizerInfo:
        def __enter__(self):
            return tokenizer

        def __exit__(self, *_args):
            return None

    class _EncoderInfo:
        @contextmanager
        def model_on_device(self):
            yield ({}, encoder)

    encoder_id = _identifier("encoder", ModelType.Qwen3VLEncoder)
    field = Qwen3VLEncoderField(
        tokenizer=encoder_id.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder_id.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=[],
    )
    invocation = Krea2TextEncoderInvocation.model_construct(
        prompt=prompt, token_weighting=token_weighting, qwen3_vl_encoder=field
    )

    def load(identifier):
        return _TokenizerInfo() if identifier.submodel_type is SubModelType.Tokenizer else _EncoderInfo()

    context = SimpleNamespace(
        models=SimpleNamespace(load=load), util=SimpleNamespace(signal_progress=lambda _message: None)
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    return invocation._encode(context), tokenizer


def _encoded_prompt(monkeypatch: pytest.MonkeyPatch, prompt: str, *, token_weighting: bool) -> str:
    """The prompt text that actually reached the tokenizer, with the fixed system prefix removed."""
    _, tokenizer = _run_encode(monkeypatch, prompt, token_weighting=token_weighting)
    assert len(tokenizer.body_texts) == 1
    body = tokenizer.body_texts[0]
    assert body.startswith(KREA2_PREFIX)
    return body[len(KREA2_PREFIX) :]


def _expected_post_drop_index(clean_prompt: str, word: str) -> int:
    """Where _WordTokenizer puts `word`, expressed in post-prefix-drop coordinates."""
    words = [token for token in (KREA2_PREFIX + clean_prompt).split(" ") if token]
    return words.index(word) - KREA2_START_IDX


# The stub tokenizes the prefix to fewer than KREA2_START_IDX tokens, so tests need filler ahead of the
# marked phrase to place it beyond the drop -- as a real prompt always is.
_FILLER = " ".join(f"w{i}" for i in range(KREA2_START_IDX))


def test_switch_off_encodes_the_prompt_exactly_as_written(monkeypatch: pytest.MonkeyPatch) -> None:
    # Off is the default, so an existing prompt that happens to contain "(...)" must reach the encoder
    # untouched -- markers included -- and must not produce weights.
    prompt = "a stone bridge (a rare heirloom variety) and a (target:0.25) tail"
    encoded = _encoded_prompt(monkeypatch, prompt, token_weighting=False)
    (_, _, token_weights), _ = _run_encode(monkeypatch, prompt, token_weighting=False)

    assert encoded == prompt
    assert token_weights is None


def test_switch_on_strips_only_the_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    # The text that reaches the encoder must be what an unweighted run would encode -- otherwise turning
    # the switch on would change the prompt itself, not just the emphasis. Parentheses with no number
    # after them are prose and stay.
    prompt = "a stone bridge (a rare heirloom variety) and a (target:0.25) tail"
    encoded = _encoded_prompt(monkeypatch, prompt, token_weighting=True)

    assert encoded == "a stone bridge (a rare heirloom variety) and a target tail"


def test_switch_on_preserves_newlines_and_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    # The compel regression this parser exists to avoid: compel flattens "a.\n\nb  c" to "a. b c", which
    # would silently change the unweighted text as soon as weighting was switched on.
    prompt = "A quiet harbour at dawn.\n\nFishing boats  rest against the (pier:1.4)."
    encoded = _encoded_prompt(monkeypatch, prompt, token_weighting=True)

    assert encoded == "A quiet harbour at dawn.\n\nFishing boats  rest against the pier."


def test_weights_land_on_the_marked_word_after_the_prefix_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    # The encoder drops the first KREA2_START_IDX tokens, so a weight computed on the pre-drop layout has
    # to be sliced identically or emphasis lands 34 tokens away from the word it was written for.
    (embeds, mask, token_weights), _ = _run_encode(monkeypatch, f"{_FILLER} (target:0.25) tail")

    assert embeds.shape == (1, 512, 12, 4)
    assert mask.shape == (1, 512)
    assert token_weights is not None
    assert token_weights.shape == (1, 512)

    weighted = (token_weights[0] != 1.0).nonzero().flatten().tolist()
    assert len(weighted) == 1
    assert token_weights[0, weighted[0]].item() == pytest.approx(0.25)
    assert weighted[0] == _expected_post_drop_index(f"{_FILLER} target tail", "target")


def test_unweighted_prompt_produces_no_token_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    # Switch on but no markers: still indistinguishable from a plain run.
    (_, _, token_weights), _ = _run_encode(monkeypatch, "a quiet street at dawn")

    assert token_weights is None


def test_weights_survive_a_multi_word_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    (_, _, token_weights), _ = _run_encode(monkeypatch, f"{_FILLER} a (deep red door:1.7) here")

    assert token_weights is not None
    weighted = (token_weights[0] != 1.0).nonzero().flatten().tolist()
    # Three consecutive tokens, one per word of the phrase, starting where "deep" landed.
    assert weighted == [
        _expected_post_drop_index(f"{_FILLER} a deep red door here", "deep") + offset for offset in range(3)
    ]
    assert torch.allclose(token_weights[0, weighted], torch.full((3,), 1.7))


def test_a_phrase_beyond_the_prefix_drop_boundary_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A phrase that falls entirely inside the dropped system-prompt region has no surviving token. That is
    # a warning, never a failed generation.
    (_, _, token_weights), _ = _run_encode(monkeypatch, "(target:0.25)")

    assert token_weights is None
