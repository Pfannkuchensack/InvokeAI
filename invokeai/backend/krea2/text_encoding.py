"""Qwen3-VL text encoding for Krea-2.

Shared by the plain and the weighted prompt nodes so both produce byte-identical conditioning for the
same text. The prompt template is copied from diffusers ``Krea2Pipeline.get_text_hidden_states``: the
prefix is a system turn instructing the model to describe an image (the same "generate" template
Qwen-Image uses), which is why the first ``KREA2_START_IDX`` tokens are dropped from the encoder output.
"""

from __future__ import annotations

import torch

from invokeai.backend.krea2.prompt_weights import WeightSpan, build_token_weights
from invokeai.backend.krea2.sampling_utils import (
    KREA2_MAX_SEQ_LEN,
    KREA2_NUM_SUFFIX_TOKENS,
    KREA2_SELECT_LAYERS,
    KREA2_START_IDX,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

KREA2_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
KREA2_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def encode_krea2_prompt(
    prompt: str,
    tokenizer,
    text_encoder,
    spans: list[WeightSpan] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Encode a prompt into Krea-2 conditioning.

    ``prompt`` must already be the clean text (weighting markers removed); ``spans`` are the weighted
    character ranges within it, or ``None`` for no weighting.

    Returns ``(prompt_embeds, prompt_mask, token_weights)`` with shapes ``(1, 512, 12, hidden)``,
    ``(1, 512)`` and ``(1, 512)`` -- the last is ``None`` unless a span matched a token. All three share
    the same sequence axis, so every downstream slice applies to them identically.
    """
    device = get_effective_device(text_encoder)

    # diffusers tokenizes (prefix + prompt) and the assistant-turn suffix separately, then concatenates -
    # so the suffix always survives truncation. Building one string and truncating it (right-truncation)
    # drops the suffix for long (>~500-token) prompts, corrupting the trained token layout that the fixed
    # prefix-drop (KREA2_START_IDX) and suffix accounting depend on.
    body_text = KREA2_PREFIX + prompt
    # Reserve room for the suffix (diffusers: max_sequence_length + start_idx - num_suffix_tokens).
    body_max_length = KREA2_MAX_SEQ_LEN + KREA2_START_IDX - KREA2_NUM_SUFFIX_TOKENS

    want_weights = bool(spans)
    if want_weights and not getattr(tokenizer, "is_fast", False):
        # The Krea-2 loaders always build a fast tokenizer (they override model_index.json's slow class),
        # so this is defensive only. Offset mappings are the only reliable way to place a phrase, so drop
        # the weighting rather than fall back to a fragile token-id search.
        InvokeAILogger.get_logger().warning(
            f"Krea-2 prompt weighting needs a fast tokenizer for offset mappings, but got "
            f"{type(tokenizer).__name__}; encoding the prompt without weights."
        )
        want_weights = False

    # Only ask for offsets when they will be used, so an unweighted prompt makes the exact same
    # tokenizer call it always did.
    offset_kwargs = {"return_offsets_mapping": True} if want_weights else {}
    body_inputs = tokenizer(
        body_text,
        max_length=body_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        **offset_kwargs,
    )
    # Append the suffix AFTER truncation so it can never be cut, matching the reference layout.
    suffix_inputs = tokenizer(KREA2_SUFFIX, return_tensors="pt")
    input_ids = torch.cat([body_inputs.input_ids, suffix_inputs.input_ids], dim=1).to(device=device)
    attention_mask = torch.cat([body_inputs.attention_mask, suffix_inputs.attention_mask], dim=1).to(
        device=device, dtype=torch.bool
    )
    # Padding sits between the prompt body and assistant suffix. Count only valid tokens when assigning
    # positions so the suffix receives the same mRoPE phase as it did during training.
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # Some VL models nest the language-model output; fall back to that if needed.
    hidden_states_tuple = getattr(outputs, "hidden_states", None)
    if hidden_states_tuple is None:
        lm_output = getattr(outputs, "language_model_outputs", None)
        hidden_states_tuple = getattr(lm_output, "hidden_states", None)
    if hidden_states_tuple is None:
        raise RuntimeError("Qwen3-VL encoder did not return hidden_states; cannot build Krea-2 conditioning.")

    # Stack the selected layers along a new layer axis: (B, seq, 12, hidden).
    stacked = torch.stack([hidden_states_tuple[i] for i in KREA2_SELECT_LAYERS], dim=2)

    # Drop the system-prompt prefix tokens.
    prompt_embeds = stacked[:, KREA2_START_IDX:]
    prompt_mask = attention_mask[:, KREA2_START_IDX:].bool()

    # Match the device-safe compute dtype used by the denoise loop (falls back from bf16 to fp16/fp32 on
    # devices without bf16 support) rather than forcing bfloat16.
    prompt_embeds = prompt_embeds.to(dtype=TorchDevice.choose_bfloat16_safe_dtype(device))

    token_weights = None
    if want_weights:
        assert spans is not None
        body_weights = build_token_weights(body_inputs.offset_mapping[0], spans, len(KREA2_PREFIX))
        if body_weights is not None:
            # The suffix is never weighted, so extend to the full 546-token layout before applying the
            # same prefix drop the embeddings and mask get.
            suffix_weights = torch.ones(suffix_inputs.input_ids.shape[1], dtype=body_weights.dtype)
            dropped = torch.cat([body_weights, suffix_weights])[KREA2_START_IDX:]
            # A phrase can match only tokens inside the dropped system-prompt region, leaving nothing
            # behind. Keep "no effective weighting" as None rather than storing a neutral tensor, so the
            # unweighted path stays identical all the way down.
            if not bool((dropped == 1.0).all()):
                token_weights = dropped.unsqueeze(0)

    return prompt_embeds, prompt_mask, token_weights
