"""FLUX.2 VAE Decode Invocation.

Decodes FLUX.2 32-channel latents to images.
"""

import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert VAE output tensor to PIL Image.

    Args:
        tensor: Tensor of shape (1, 3, H, W) in range [-1, 1].

    Returns:
        PIL Image in RGB mode.
    """
    import numpy as np

    # Clamp to [-1, 1] and convert to [0, 1]
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2

    # Convert to numpy (1, C, H, W) -> (H, W, C)
    np_image = tensor[0].permute(1, 2, 0).cpu().float().numpy()

    # Convert to uint8
    np_image = (np_image * 255).clip(0, 255).astype("uint8")

    return Image.fromarray(np_image, mode="RGB")


@invocation(
    "flux2_vae_decode",
    title="VAE Decode - FLUX.2",
    tags=["vae", "decode", "flux2", "image"],
    category="image",
    version="1.0.0",
)
class Flux2VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decodes FLUX.2 latents (32 channels) to an image."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load latents
        latents = context.tensors.load(self.latents.latents_name)

        # Load VAE
        vae_info = context.models.load(self.vae.vae)

        with vae_info.model_on_device() as (_, vae):
            device = next(vae.parameters()).device
            dtype = next(vae.parameters()).dtype

            # Move latents to device
            latents = latents.to(device=device, dtype=dtype)

            # Undo scaling if present
            if hasattr(vae.config, "scaling_factor"):
                latents = latents / vae.config.scaling_factor

            # Decode
            decoded = vae.decode(latents).sample

            # Convert to image
            image = tensor_to_image(decoded)

        # Save image
        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
