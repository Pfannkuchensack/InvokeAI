"""FLUX.2 Denoise Invocation.

Main denoising invocation for FLUX.2 image generation.
"""

import math
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Flux2ConditioningField,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids,
    get_flux2_noise,
    get_flux2_schedule,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUX2ConditioningInfo
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


@invocation(
    "flux2_denoise",
    title="FLUX.2 Denoise",
    tags=["denoise", "flux2", "latents"],
    category="latents",
    version="1.0.0",
)
class Flux2DenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Runs the FLUX.2 denoising loop.

    FLUX.2 uses a single Mistral text encoder and 32-channel latent space.
    Supports multi-image reference conditioning (up to 10 images).
    """

    # Model inputs
    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
    )

    # Conditioning
    positive_text_conditioning: Union[Flux2ConditioningField, List[Flux2ConditioningField]] = InputField(
        description="Positive text conditioning",
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[Union[Flux2ConditioningField, List[Flux2ConditioningField]]] = InputField(
        default=None,
        description="Optional negative text conditioning for CFG",
        input=Input.Connection,
    )

    # Optional input latents (for img2img)
    latents: Optional[LatentsField] = InputField(
        default=None,
        description="Optional input latents for img2img",
        input=Input.Connection,
    )

    # Generation parameters
    width: int = InputField(
        default=1024,
        multiple_of=16,
        ge=64,
        le=8192,
        description="Image width",
    )
    height: int = InputField(
        default=1024,
        multiple_of=16,
        ge=64,
        le=8192,
        description="Image height",
    )
    num_steps: int = InputField(
        default=28,
        ge=1,
        le=100,
        description="Number of denoising steps (recommended: 28 for FLUX.2)",
    )
    guidance: float = InputField(
        default=3.5,
        ge=0.0,
        le=20.0,
        description="Guidance scale for FLUX.2 (embedded in model)",
    )
    cfg_scale: float = InputField(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="CFG scale (1.0 = disabled)",
    )
    seed: int = InputField(
        default=0,
        description="Random seed for noise generation",
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Denoising start point (0.0 = full noise, for img2img)",
    )
    denoising_end: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Denoising end point",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)

        # Save latents
        latents_cpu = latents.cpu()
        latents_name = context.tensors.save(latents_cpu)

        return LatentsOutput.build(
            latents_name=latents_name,
            latents=latents_cpu,
            seed=self.seed,
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        """Run the FLUX.2 denoising loop."""
        # Load transformer
        transformer_info = context.models.load(self.transformer.transformer)

        with transformer_info.model_on_device() as (_, transformer):
            device = next(transformer.parameters()).device
            dtype = next(transformer.parameters()).dtype

            # Generate or load initial latents
            if self.latents is not None:
                # img2img: load existing latents
                init_latents = context.tensors.load(self.latents.latents_name)
                init_latents = init_latents.to(device=device, dtype=dtype)
            else:
                init_latents = None

            # Generate noise
            noise = get_flux2_noise(
                num_samples=1,
                height=self.height,
                width=self.width,
                device=device,
                dtype=dtype,
                seed=self.seed,
            )

            # Calculate sequence length for scheduling
            latent_height = 2 * math.ceil(self.height / 16)
            latent_width = 2 * math.ceil(self.width / 16)
            image_seq_len = (latent_height // 2) * (latent_width // 2)

            # Get timestep schedule
            timesteps = get_flux2_schedule(
                num_steps=self.num_steps,
                image_seq_len=image_seq_len,
                shift=True,
            )

            # Apply denoising range
            if self.denoising_start > 0.0 or self.denoising_end < 1.0:
                total = len(timesteps) - 1
                start_idx = int(self.denoising_start * total)
                end_idx = int(self.denoising_end * total)
                timesteps = timesteps[start_idx : end_idx + 1]

            # Prepare initial state
            if init_latents is not None and self.denoising_start > 0.0:
                # img2img: interpolate noise with init latents
                t_start = timesteps[0]
                img = t_start * noise + (1 - t_start) * init_latents
            else:
                img = noise

            # Pack latents for transformer
            img = pack_flux2(img)

            # Generate position IDs
            img_ids = generate_img_ids(
                height=latent_height,
                width=latent_width,
                batch_size=1,
                device=device,
                dtype=dtype,
            )

            # Load conditioning
            pos_conditioning = self._load_conditioning(context, self.positive_text_conditioning)
            pos_txt = pos_conditioning.mistral_embeds.to(device=device, dtype=dtype)

            # Generate text position IDs
            txt_ids = torch.zeros(1, pos_txt.shape[1], 3, device=device, dtype=dtype)

            # Load negative conditioning if CFG is enabled
            neg_txt = None
            neg_txt_ids = None
            do_cfg = self.cfg_scale > 1.0 and self.negative_text_conditioning is not None
            if do_cfg:
                neg_conditioning = self._load_conditioning(context, self.negative_text_conditioning)
                neg_txt = neg_conditioning.mistral_embeds.to(device=device, dtype=dtype)
                neg_txt_ids = torch.zeros(1, neg_txt.shape[1], 3, device=device, dtype=dtype)

            # Denoising loop
            for i, (t_curr, t_prev) in enumerate(
                tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="FLUX.2 Denoising")
            ):
                # Create timestep tensor
                t_vec = torch.full((1,), t_curr, dtype=dtype, device=device)
                guidance_vec = torch.full((1,), self.guidance, dtype=dtype, device=device)

                if do_cfg:
                    # Batched CFG
                    img_batched = torch.cat([img, img], dim=0)
                    img_ids_batched = torch.cat([img_ids, img_ids], dim=0)
                    txt_batched = torch.cat([pos_txt, neg_txt], dim=0)
                    txt_ids_batched = torch.cat([txt_ids, neg_txt_ids], dim=0)
                    t_vec_batched = torch.cat([t_vec, t_vec], dim=0)
                    guidance_batched = torch.cat([guidance_vec, guidance_vec], dim=0)

                    pred = transformer(
                        hidden_states=img_batched,
                        encoder_hidden_states=txt_batched,
                        timestep=t_vec_batched,
                        img_ids=img_ids_batched,
                        txt_ids=txt_ids_batched,
                        guidance=guidance_batched,
                        return_dict=False,
                    )[0]

                    # Apply CFG
                    pred_pos, pred_neg = pred.chunk(2, dim=0)
                    pred = pred_neg + self.cfg_scale * (pred_pos - pred_neg)
                else:
                    # Single pass (no CFG)
                    pred = transformer(
                        hidden_states=img,
                        encoder_hidden_states=pos_txt,
                        timestep=t_vec,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance_vec,
                        return_dict=False,
                    )[0]

                # Flow matching update
                img = img + (t_prev - t_curr) * pred

                # Progress callback
                context.util.signal_progress(
                    message=f"FLUX.2 step {i + 1}/{len(timesteps) - 1}",
                    percentage=(i + 1) / (len(timesteps) - 1),
                )

            # Unpack latents
            latents = unpack_flux2(img, latent_height // 2, latent_width // 2)

        return latents

    def _load_conditioning(
        self, context: InvocationContext, cond_field: Union[Flux2ConditioningField, List[Flux2ConditioningField]]
    ) -> FLUX2ConditioningInfo:
        """Load conditioning from context."""
        if isinstance(cond_field, list):
            cond_field = cond_field[0]  # Take first for now

        cond_data = context.conditioning.load(cond_field.conditioning_name)
        return cond_data.conditionings[0]
