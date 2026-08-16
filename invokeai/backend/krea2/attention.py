"""Memory-efficient attention processor for the Krea-2 transformer.

The stock ``Krea2AttnProcessor`` calls ``scaled_dot_product_attention`` with ``enable_gqa=True`` (Krea-2 uses
grouped-query attention: 48 query heads, 12 key/value heads). PyTorch's fused flash / memory-efficient SDPA
kernels do **not** support ``enable_gqa``, so this forces the *math* backend, which materializes the full
``[heads, seq, seq]`` score matrix. At 1280x720 (3600 image tokens) that is ~5.7 GB **per attention**, and it
grows O(seq^2) — ~40 GB at 2560x1440 — so generation OOMs or the cache offloads the transformer to RAM.

This processor instead expands the K/V heads to match the query heads (``repeat_interleave``) so ``enable_gqa``
is not needed, and runs under the memory-efficient SDPA backend (which supports the additive padding mask and
is O(seq) in memory). Measured: the same 3600-token attention drops from ~5.7 GB to ~0.19 GB.

The math is otherwise identical to ``Krea2AttnProcessor`` (q/k RMSNorm, rotary embeddings, sigmoid output gate).
"""

import re
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import SDPBackend, sdpa_kernel

# Prefer the memory-efficient kernel; fall back to flash (if the build has it) then math so we never hard-fail.
_KREA2_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


@dataclass
class Krea2RegionalPromptingState:
    """Mutable per-forward attention state shared by Krea-2 transformer-block processors.

    Carries the regional attention mask and, for per-token prompt weighting, the value scale applied to
    each token's value vector. The key bias is not stored separately: it is already folded into
    ``attention_mask`` by ``Krea2RegionalPromptingExtension.get_attention_mask_with_bias``, so the
    processor only ever sees one mask.
    """

    attention_mask: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask

    def set_value_scale(self, value_scale: torch.Tensor | None) -> None:
        self.value_scale = value_scale

    def clear(self) -> None:
        """Drop every retained tensor. The processors outlive the invocation on the cached transformer."""
        self.attention_mask = None
        self.value_scale = None


class Krea2MemoryEfficientAttnProcessor:
    """Drop-in replacement for ``Krea2AttnProcessor`` that avoids the ``enable_gqa`` math fallback.

    ``apply_regional_mask`` follows the alternating-block policy regional prompting uses; per-token
    prompt weights are applied by every processor that has state, matching the reference ComfyUI node's
    all-blocks patch.
    """

    def __init__(
        self,
        regional_prompting_state: Krea2RegionalPromptingState | None = None,
        apply_regional_mask: bool = True,
    ) -> None:
        self.regional_prompting_state = regional_prompting_state
        self.apply_regional_mask = apply_regional_mask

    def _shared_attention_mask(self, sequence_length: int) -> torch.Tensor | None:
        state = self.regional_prompting_state
        if state is None or state.attention_mask is None or not self.apply_regional_mask:
            return None
        mask = state.attention_mask
        # A regional mask is (S, S); a pure key bias is (B, 1, 1, S) and broadcasts over query rows. Both
        # must address exactly this conditioning's sequence, so a stale mask from the other CFG pass fails
        # loudly instead of being broadcast into attention.
        valid = mask.shape[-1] == sequence_length and (mask.ndim != 2 or mask.shape[0] == sequence_length)
        if not valid:
            raise ValueError(
                f"Krea-2 regional attention mask shape {tuple(mask.shape)} does not match "
                f"the transformer sequence length {sequence_length}."
            )
        return mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        shared_mask = self._shared_attention_mask(hidden_states.shape[1])
        if shared_mask is not None:
            attention_mask = shared_mask if attention_mask is None else attention_mask & shared_mask

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        # Per-token prompt weighting: scale each token's value vector. Done here, while value is still
        # (B, S, num_kv_heads, head_dim), so it costs 12 heads' worth of multiply instead of 48.
        if self.regional_prompting_state is not None and self.regional_prompting_state.value_scale is not None:
            value_scale = self.regional_prompting_state.value_scale
            if value_scale.shape[1] != hidden_states.shape[1]:
                raise ValueError(
                    f"Krea-2 value scale length {value_scale.shape[1]} does not match the transformer "
                    f"sequence length {hidden_states.shape[1]}."
                )
            value = value * value_scale.to(dtype=value.dtype)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # [B, S, H, D] -> [B, H, S, D] for scaled_dot_product_attention.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Expand K/V heads to the query head count so we can drop enable_gqa (which forces the math backend).
        if attn.num_heads != attn.num_kv_heads:
            repeats = attn.num_heads // attn.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        # A float mask carries the key bias as an additive pre-softmax term. Unlike a bool mask, SDPA
        # requires it to match the query dtype exactly ("invalid dtype for bias"), and the query dtype is
        # only settled at runtime for quantized/fp8 transformers.
        if attention_mask is not None and attention_mask.is_floating_point():
            attention_mask = attention_mask.to(dtype=query.dtype)

        with sdpa_kernel(_KREA2_SDPA_BACKENDS):
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D], matching Krea2AttnProcessor's output layout.
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)


class _Krea2AttentionProcessorContainer(Protocol):
    @property
    def attn_processors(self) -> dict[str, object]: ...


def build_krea2_attention_processors(
    transformer: _Krea2AttentionProcessorContainer,
    regional_prompting_state: Krea2RegionalPromptingState,
) -> dict[str, Krea2MemoryEfficientAttnProcessor]:
    """Build processors for the main transformer blocks, wired to the shared attention state.

    Two independent policies live here. Regional masks apply to alternating blocks only, as they always
    have. Per-token prompt weights apply to every main block, matching the reference ComfyUI node -- with
    a broadcast key bias that costs nothing, there is no reason to halve the effect.

    Only ``transformer_blocks.*`` get the state. ``text_fusion.layerwise_blocks.*`` also uses
    ``Krea2Attention``, but its attention sequence axis is the 12 tapped encoder layers rather than
    prompt tokens, so a token-length mask or value scale there would silently weight the wrong thing.
    """

    processors: dict[str, Krea2MemoryEfficientAttnProcessor] = {}
    for name in transformer.attn_processors:
        match = re.fullmatch(r"transformer_blocks\.(\d+)\.attn\.processor", name)
        block_index = int(match.group(1)) if match is not None else None
        processors[name] = Krea2MemoryEfficientAttnProcessor(
            regional_prompting_state=regional_prompting_state if block_index is not None else None,
            apply_regional_mask=block_index is not None and block_index % 2 == 0,
        )
    return processors
