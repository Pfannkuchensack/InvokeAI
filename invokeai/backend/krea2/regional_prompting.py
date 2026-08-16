from dataclasses import dataclass

import torch
import torchvision

from invokeai.backend.krea2.prompt_weights import derive_attention_weights
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.mask import to_standard_float_mask


@dataclass
class Krea2TextConditioning:
    prompt_embeds: torch.Tensor
    mask: torch.Tensor | None
    token_weights: torch.Tensor | None = None


@dataclass
class Krea2RegionalTextConditioning:
    prompt_embeds: torch.Tensor
    image_masks: list[torch.Tensor | None]
    embedding_ranges: list[Range]
    token_weights: torch.Tensor | None = None


class Krea2RegionalPromptingExtension:
    """Concatenates Krea-2 text conditionings and lazily builds Flux-style regional attention masks."""

    def __init__(self, regional_text_conditioning: Krea2RegionalTextConditioning, image_seq_len: int) -> None:
        self.regional_text_conditioning = regional_text_conditioning
        self.image_seq_len = image_seq_len
        self._attention_mask: torch.Tensor | None = None

    @property
    def has_regional_masks(self) -> bool:
        return any(mask is not None for mask in self.regional_text_conditioning.image_masks)

    @property
    def attention_mask_numel(self) -> int:
        if not self.has_regional_masks:
            return 0
        total_seq_len = self.regional_text_conditioning.prompt_embeds.shape[1] + self.image_seq_len
        return total_seq_len**2

    @property
    def attention_mask_build_scratch_numel(self) -> int:
        """Peak boolean scratch allocation used while constructing the image-to-image attention block."""
        if not self.has_regional_masks:
            return 0
        return self.image_seq_len**2

    @classmethod
    def from_text_conditionings(
        cls, text_conditionings: list[Krea2TextConditioning], image_seq_len: int
    ) -> "Krea2RegionalPromptingExtension":
        if not text_conditionings:
            raise ValueError("At least one Krea-2 text conditioning is required.")

        prompt_embeds: list[torch.Tensor] = []
        image_masks: list[torch.Tensor | None] = []
        embedding_ranges: list[Range] = []
        token_weights: list[torch.Tensor] = []
        any_weighted = False
        current_start = 0
        for conditioning in text_conditionings:
            sequence_length = conditioning.prompt_embeds.shape[1]
            if conditioning.mask is not None and conditioning.mask.numel() != image_seq_len:
                raise ValueError(
                    f"Krea-2 regional mask has {conditioning.mask.numel()} values, expected {image_seq_len}."
                )
            prompt_embeds.append(conditioning.prompt_embeds)
            image_masks.append(conditioning.mask)
            embedding_ranges.append(Range(start=current_start, end=current_start + sequence_length))
            current_start += sequence_length

            # A weighted and an unweighted prompt can share a regional group, so stand in neutral weights
            # for the unweighted ones and concatenate exactly like the embeddings.
            weights = conditioning.token_weights
            if weights is None:
                weights = conditioning.prompt_embeds.new_ones(conditioning.prompt_embeds.shape[:2])
            else:
                if weights.shape != conditioning.prompt_embeds.shape[:2]:
                    raise ValueError(
                        f"Krea-2 token weights shape {tuple(weights.shape)} does not match prompt embedding "
                        f"shape {tuple(conditioning.prompt_embeds.shape[:2])}."
                    )
                any_weighted = True
            token_weights.append(weights)

        regional_text_conditioning = Krea2RegionalTextConditioning(
            prompt_embeds=torch.cat(prompt_embeds, dim=1),
            image_masks=image_masks,
            embedding_ranges=embedding_ranges,
            token_weights=torch.cat(token_weights, dim=1) if any_weighted else None,
        )
        return cls(regional_text_conditioning=regional_text_conditioning, image_seq_len=image_seq_len)

    def get_attention_mask(self) -> torch.Tensor | None:
        if not self.has_regional_masks:
            return None
        if self._attention_mask is None:
            self._attention_mask = self._build_attention_mask()
        return self._attention_mask

    def get_token_weight_vectors(self, strength: float) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build the joint-sequence value scale and key bias for the transformer's attention.

        Returns ``(value_scale, key_bias)`` shaped ``(B, total_seq, 1, 1)`` and ``(B, 1, 1, total_seq)``
        so they broadcast over heads and (for the bias) over query rows. Text tokens come first in the
        joint sequence, so image positions are simply padded with the neutral 1.0 / 0.0. Returns ``None``
        when there is nothing to apply, which keeps the unweighted path on exactly the code and tensors
        it uses today.
        """
        token_weights = self.regional_text_conditioning.token_weights
        if token_weights is None or strength == 0.0:
            return None

        value_scale, key_bias = derive_attention_weights(token_weights, strength)
        if bool((value_scale == 1.0).all()) and bool((key_bias == 0.0).all()):
            return None

        batch_size = token_weights.shape[0]
        image_scale = value_scale.new_ones((batch_size, self.image_seq_len))
        image_bias = key_bias.new_zeros((batch_size, self.image_seq_len))
        value_scale = torch.cat([value_scale, image_scale], dim=1)[:, :, None, None]
        key_bias = torch.cat([key_bias, image_bias], dim=1)[:, None, None, :]
        return value_scale, key_bias

    def get_attention_mask_with_bias(self, key_bias: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
        """Combine the regional mask and the key bias into the single mask SDPA takes.

        With only a bias this is the ``(B, 1, 1, total_seq)`` tensor itself, which broadcasts over query
        rows and costs nothing. With only a regional mask it is today's bool mask, untouched. With both,
        the bool mask is folded into a float mask once per conditioning -- never per block -- because a
        float mask is what SDPA converts a bool mask into internally anyway.
        """
        regional_mask = self.get_attention_mask()
        if key_bias is None:
            return regional_mask
        if regional_mask is None:
            return key_bias.to(dtype)

        # finfo.min rather than -inf: a MATH fallback would turn a fully-masked query row into NaN. The
        # bias is flattened to (1, total_seq) first so the result stays (total_seq, total_seq) rather than
        # growing leading axes, keeping it the same size as the bool mask it replaces.
        combined = torch.where(regional_mask, key_bias.reshape(1, -1).to(dtype), torch.finfo(dtype).min)
        # The bool mask can be hundreds of MB at high resolution; the float mask supersedes it.
        self._attention_mask = None
        return combined

    def _build_attention_mask(self) -> torch.Tensor:
        conditioning = self.regional_text_conditioning
        text_seq_len = conditioning.prompt_embeds.shape[1]
        total_seq_len = text_seq_len + self.image_seq_len
        device = conditioning.prompt_embeds.device
        attention_mask = torch.zeros((total_seq_len, total_seq_len), device=device, dtype=torch.bool)

        background_mask = torch.ones(self.image_seq_len, device=device, dtype=torch.bool)
        for image_mask in conditioning.image_masks:
            if image_mask is not None:
                background_mask &= ~(image_mask.reshape(-1) > 0.5)
        # Canvas graphs always include an unmasked global prompt. If regional masks cover the full image,
        # there is no background for it; fall back to image-wide text/image attention instead of silently
        # disconnecting that conditioning. Keep background_mask unchanged so regional image/image isolation
        # remains intact.
        unmasked_conditioning_mask = background_mask | ~background_mask.any()

        image_attention_mask = attention_mask[text_seq_len:, text_seq_len:]
        for image_mask, embedding_range in zip(conditioning.image_masks, conditioning.embedding_ranges, strict=True):
            text_slice = slice(embedding_range.start, embedding_range.end)
            attention_mask[text_slice, text_slice] = True

            if image_mask is None:
                attention_mask[text_slice, text_seq_len:] = unmasked_conditioning_mask
                attention_mask[text_seq_len:, text_slice] = unmasked_conditioning_mask[:, None]
                continue

            region_mask = image_mask.reshape(-1) > 0.5
            attention_mask[text_slice, text_seq_len:] = region_mask
            attention_mask[text_seq_len:, text_slice] = region_mask[:, None]
            image_attention_mask |= region_mask[:, None] & region_mask[None, :]

        image_attention_mask |= background_mask[:, None] | background_mask[None, :]
        return attention_mask

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: torch.Tensor,
        grid_height: int,
        grid_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = to_standard_float_mask(mask, out_dtype=dtype)
        resize = torchvision.transforms.Resize(
            (grid_height, grid_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        return resize(mask.unsqueeze(0)).flatten(start_dim=2).to(device=device)
