"""Text conditioning data structures for FLUX.2.

FLUX.2 uses a single Mistral Small 3.1 text encoder instead of
the dual CLIP + T5 encoders used in FLUX.1.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Flux2TextConditioning:
    """Text conditioning data for FLUX.2 denoising.

    Unlike FLUX.1 which uses separate CLIP and T5 embeddings, FLUX.2
    uses a single Mistral encoder output.

    Attributes:
        mistral_embeddings: Text embeddings from Mistral encoder.
            Shape: (batch, seq_len, hidden_dim)
        mask: Optional attention mask for regional prompting.
            Shape: (batch, height, width) or None
    """

    mistral_embeddings: torch.Tensor
    mask: Optional[torch.Tensor] = None


@dataclass
class Flux2ConditioningInfo:
    """Conditioning info for saving/loading FLUX.2 embeddings.

    This is used for caching and passing conditioning between invocations.

    Attributes:
        embeds: The Mistral encoder embeddings.
    """

    embeds: torch.Tensor

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "Flux2ConditioningInfo":
        """Move conditioning to specified device and dtype.

        Args:
            device: Target device.
            dtype: Optional target dtype.

        Returns:
            New Flux2ConditioningInfo on the specified device.
        """
        return Flux2ConditioningInfo(
            embeds=self.embeds.to(device=device, dtype=dtype) if dtype else self.embeds.to(device=device)
        )


# System message constants for FLUX.2 Mistral encoder
FLUX2_SYSTEM_MESSAGE = """You are an image captioning assistant. Provide accurate, detailed descriptions of images."""

FLUX2_SYSTEM_MESSAGE_UPSAMPLING_T2I = """You are an image upsampling assistant. Generate enhanced, high-resolution versions of the described image."""

FLUX2_SYSTEM_MESSAGE_UPSAMPLING_I2I = """You are an image-to-image upsampling assistant. Enhance and upscale the provided image while preserving its content."""


def format_flux2_prompt(
    prompt: str,
    system_message: str = FLUX2_SYSTEM_MESSAGE,
    images: Optional[list] = None,
) -> list[dict]:
    """Format a prompt for the FLUX.2 Mistral encoder.

    Creates a conversation format compatible with apply_chat_template().

    Args:
        prompt: The text prompt to encode.
        system_message: System message for the encoder.
        images: Optional list of reference images.

    Returns:
        List of message dicts in chat format.
    """
    # Remove [IMG] tokens to avoid validation issues
    cleaned_prompt = prompt.replace("[IMG]", "")

    if images is None or len(images) == 0:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": cleaned_prompt}],
            },
        ]

    # With reference images
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
    ]

    # Add images
    if images:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images],
            }
        )

    # Add text prompt
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": cleaned_prompt}],
        }
    )

    return messages
