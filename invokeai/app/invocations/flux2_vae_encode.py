"""FLUX.2 VAE Encode Invocation.

Encodes images to FLUX.2 32-channel latent space.
"""

import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


def image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL Image to tensor for FLUX.2 VAE.

    Args:
        image: PIL Image in RGB mode.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (1, 3, H, W) normalized to [-1, 1].
    """
    import numpy as np

    # Convert to numpy and normalize
    np_image = np.array(image).astype(np.float32) / 255.0
    # Normalize to [-1, 1]
    np_image = 2.0 * np_image - 1.0
    # Convert to tensor (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


@invocation(
    "flux2_vae_encode",
    title="VAE Encode - FLUX.2",
    tags=["vae", "encode", "flux2", "latents"],
    category="latents",
    version="1.0.0",
)
class Flux2VaeEncodeInvocation(BaseInvocation):
    """Encodes an image to FLUX.2 latent space (32 channels)."""

    image: ImageField = InputField(description="Image to encode")
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Load image
        image = context.images.get_pil(self.image.image_name)
        image = image.convert("RGB")

        # Load VAE
        vae_info = context.models.load(self.vae.vae)

        with vae_info.model_on_device() as (_, vae):
            device = next(vae.parameters()).device
            dtype = next(vae.parameters()).dtype

            # Convert image to tensor
            image_tensor = image_to_tensor(image, device, dtype)

            # Encode
            latents = vae.encode(image_tensor).latent_dist.sample()

            # FLUX.2 VAE may have scaling factor
            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor

        # Move to CPU and save
        latents = latents.cpu()
        latents_name = context.tensors.save(latents)

        return LatentsOutput.build(
            latents_name=latents_name,
            latents=latents,
            seed=None,
        )
