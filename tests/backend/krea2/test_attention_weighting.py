"""Attention-side tests for Krea-2 per-token prompt weighting.

The memory-efficiency behaviour of the processor lives in ``test_attention.py``; this file covers the
value scale and key bias that implement prompt weighting.
"""

import math

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention
from torch.nn.attention import SDPBackend

import invokeai.backend.krea2.attention as krea2_attention
from invokeai.backend.krea2.attention import Krea2MemoryEfficientAttnProcessor, Krea2RegionalPromptingState


def _build_gqa_attention() -> Krea2Attention:
    torch.manual_seed(0)
    attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
    assert attn.num_heads != attn.num_kv_heads
    return attn


def _reference_attention(
    attn: Krea2Attention,
    hidden_states: torch.Tensor,
    value_scale: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit softmax(QK^T/sqrt(d) + bias) @ (V * scale) to check the processor's fused path against."""
    query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    gate = attn.to_gate(hidden_states)

    if value_scale is not None:
        value = value * value_scale

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    repeats = attn.num_heads // attn.num_kv_heads
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)

    scores = query @ key.transpose(-1, -2) / math.sqrt(attn.head_dim)
    if key_bias is not None:
        scores = scores + key_bias
    out = torch.softmax(scores, dim=-1) @ value
    out = out.transpose(1, 2).flatten(2, 3) * torch.sigmoid(gate)
    return attn.to_out[0](out)


@pytest.mark.parametrize("scale", [0.5, 0.0, -1.0])
def test_value_scale_matches_an_explicit_reference(scale: float) -> None:
    # Weight <= 1 scales the token's value vector; below zero it subtracts, which is the direction the
    # reference ComfyUI node relies on for removing a concept entirely.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    value_scale = torch.ones(1, 24, 1, 1)
    value_scale[:, 3:6] = scale
    state = Krea2RegionalPromptingState(value_scale=value_scale)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        actual = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        expected = _reference_attention(attn, hidden_states, value_scale=value_scale)
        unscaled = _reference_attention(attn, hidden_states)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, unscaled, atol=1e-4, rtol=1e-4)


def test_key_bias_matches_an_explicit_reference() -> None:
    # Weight > 1 becomes an additive pre-softmax bias on the token's key column, broadcast over queries.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    key_bias = torch.zeros(1, 1, 1, 24)
    key_bias[..., 2] = 1.6
    state = Krea2RegionalPromptingState(attention_mask=key_bias)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        actual = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        expected = _reference_attention(attn, hidden_states, key_bias=key_bias)
        unbiased = _reference_attention(attn, hidden_states)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, unbiased, atol=1e-4, rtol=1e-4)


def test_combined_regional_mask_and_key_bias_matches_an_explicit_reference() -> None:
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    regional_mask = torch.block_diag(*[torch.ones(12, 12, dtype=torch.bool)] * 2)
    key_bias = torch.zeros(1, 1, 1, 24)
    key_bias[..., 5] = 1.2

    dtype = hidden_states.dtype
    combined = torch.where(regional_mask, key_bias.to(dtype), torch.finfo(dtype).min)
    state = Krea2RegionalPromptingState(attention_mask=combined)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        actual = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        expected = _reference_attention(attn, hidden_states, key_bias=combined)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_unset_weights_are_bitwise_identical_to_no_state() -> None:
    # The zero-regression guarantee: a prompt without weighting markers takes the same code path with the
    # same tensors it does today, not merely a numerically close one.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=Krea2RegionalPromptingState()))
        out_empty_state = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_no_state = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert torch.equal(out_empty_state, out_no_state)


def test_value_scale_applies_even_when_the_regional_mask_does_not() -> None:
    # Odd main blocks skip the regional mask but must still apply per-token weights, matching the
    # reference node's all-blocks patch.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    value_scale = torch.ones(1, 24, 1, 1)
    value_scale[:, 3:6] = 0.0
    state = Krea2RegionalPromptingState(
        attention_mask=torch.block_diag(*[torch.ones(12, 12, dtype=torch.bool)] * 2),
        value_scale=value_scale,
    )

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state, apply_regional_mask=False))
        actual = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        # The mask is skipped, so the reference applies the value scale only.
        expected = _reference_attention(attn, hidden_states, value_scale=value_scale)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_value_scale_sized_for_a_different_conditioning_is_rejected() -> None:
    # The positive and negative prompts tokenize to different lengths, so a stale scale must fail loudly
    # rather than broadcast onto the wrong tokens.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    state = Krea2RegionalPromptingState(value_scale=torch.ones(1, 20, 1, 1))
    attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))

    with pytest.raises(ValueError, match="value scale length 20 does not match"):
        attn(hidden_states, attention_mask=None, image_rotary_emb=None)


def test_broadcast_key_bias_sized_for_a_different_conditioning_is_rejected() -> None:
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    state = Krea2RegionalPromptingState(attention_mask=torch.zeros(1, 1, 1, 20))
    attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))

    with pytest.raises(ValueError, match="does not match the transformer sequence length 24"):
        attn(hidden_states, attention_mask=None, image_rotary_emb=None)


def test_clear_drops_every_retained_tensor() -> None:
    # The processors outlive the invocation on the cached transformer; nothing may be retained.
    state = Krea2RegionalPromptingState(
        attention_mask=torch.ones(4, 4, dtype=torch.bool), value_scale=torch.ones(1, 4, 1, 1)
    )

    state.clear()

    assert state.attention_mask is None
    assert state.value_scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise fused SDPA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_float_bias_is_cast_to_the_query_dtype_and_stays_on_the_efficient_kernel(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    # SDPA tolerates a bool mask of any dtype but requires a *float* mask to match the query dtype exactly
    # ("invalid dtype for bias"), and the query dtype is only settled at runtime for quantized/fp8
    # transformers -- hence the unconditional cast. Pinning the backend turns a silent fall back to the
    # math kernel, the O(seq^2) OOM this module exists to avoid, into a hard failure.
    attn = _build_gqa_attention().to(device="cuda", dtype=dtype)
    hidden_states = torch.randn(1, 24, attn.hidden_size, device="cuda", dtype=dtype)
    key_bias = torch.zeros(1, 1, 1, 24, device="cuda", dtype=torch.float32)
    key_bias[..., 2] = 1.5
    state = Krea2RegionalPromptingState(attention_mask=key_bias)
    monkeypatch.setattr(krea2_attention, "_KREA2_SDPA_BACKENDS", [SDPBackend.EFFICIENT_ATTENTION])

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        output = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert output.dtype == dtype
    assert torch.isfinite(output).all()
