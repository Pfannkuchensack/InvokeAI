"""FLUX.2 denoising loop implementation.

This module provides the core denoising functionality for FLUX.2,
wrapping the diffusers Flux2Transformer2DModel.
"""

from typing import Callable, Optional

import torch
from tqdm import tqdm

from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise_flux2(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    guidance: float,
    step_callback: Optional[Callable[[PipelineIntermediateState], None]] = None,
    cfg_scale: float | list[float] = 1.0,
    neg_txt: Optional[torch.Tensor] = None,
    neg_txt_ids: Optional[torch.Tensor] = None,
    reference_latents: Optional[torch.Tensor] = None,
    reference_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the FLUX.2 denoising loop.

    Args:
        model: FLUX.2 transformer model (Flux2Transformer2DModel).
        img: Packed noisy image latents of shape (B, seq_len, hidden_dim).
        img_ids: Image position IDs of shape (B, seq_len, 3).
        txt: Text embeddings from Mistral encoder of shape (B, txt_len, hidden_dim).
        txt_ids: Text position IDs of shape (B, txt_len, 3).
        timesteps: List of timesteps for denoising (1.0 to 0.0).
        guidance: Guidance scale for FLUX.2 (used in model forward).
        step_callback: Optional callback for progress reporting.
        cfg_scale: CFG scale for classifier-free guidance (1.0 = no CFG).
        neg_txt: Optional negative text embeddings for CFG.
        neg_txt_ids: Optional negative text position IDs.
        reference_latents: Optional reference image latents for multi-image generation.
        reference_ids: Optional reference image position IDs.

    Returns:
        Denoised latents of shape (B, seq_len, hidden_dim).
    """
    batch_size = img.shape[0]
    device = img.device
    dtype = img.dtype

    # Determine if we're doing CFG
    do_cfg = isinstance(cfg_scale, list) or cfg_scale != 1.0
    if do_cfg and neg_txt is None:
        raise ValueError("Negative text embeddings required for CFG")

    # Convert cfg_scale to list if needed
    if isinstance(cfg_scale, float):
        cfg_scales = [cfg_scale] * (len(timesteps) - 1)
    else:
        cfg_scales = cfg_scale

    # Prepend reference latents if provided
    if reference_latents is not None:
        img = torch.cat([reference_latents, img], dim=1)
        img_ids = torch.cat([reference_ids, img_ids], dim=1)

    original_seq_len = img.shape[1] - (reference_latents.shape[1] if reference_latents is not None else 0)

    # Denoising loop
    for i, (t_curr, t_prev) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="FLUX.2 Denoising")):
        # Create timestep tensor
        t_vec = torch.full((batch_size,), t_curr, dtype=dtype, device=device)

        # Guidance embedding for FLUX.2
        guidance_vec = torch.full((batch_size,), guidance, dtype=dtype, device=device)

        # Forward pass
        if do_cfg and cfg_scales[i] != 1.0:
            # Batch positive and negative together
            img_batched = torch.cat([img, img], dim=0)
            img_ids_batched = torch.cat([img_ids, img_ids], dim=0)
            txt_batched = torch.cat([txt, neg_txt], dim=0)
            txt_ids_batched = torch.cat([txt_ids, neg_txt_ids], dim=0)
            t_vec_batched = torch.cat([t_vec, t_vec], dim=0)
            guidance_vec_batched = torch.cat([guidance_vec, guidance_vec], dim=0)

            # Model prediction
            pred_batched = model(
                hidden_states=img_batched,
                encoder_hidden_states=txt_batched,
                timestep=t_vec_batched,
                img_ids=img_ids_batched,
                txt_ids=txt_ids_batched,
                guidance=guidance_vec_batched,
                return_dict=False,
            )[0]

            # Split and apply CFG
            pred_pos, pred_neg = pred_batched.chunk(2, dim=0)
            pred = pred_neg + cfg_scales[i] * (pred_pos - pred_neg)
        else:
            # No CFG - single forward pass
            pred = model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                return_dict=False,
            )[0]

        # Remove reference tokens from prediction if present
        if reference_latents is not None:
            pred = pred[:, -original_seq_len:]

        # Update latents using flow matching equation
        # x_prev = x_curr + (t_prev - t_curr) * v_pred
        img_update = img[:, -original_seq_len:] if reference_latents is not None else img
        img_update = img_update + (t_prev - t_curr) * pred

        if reference_latents is not None:
            img = torch.cat([img[:, :-original_seq_len], img_update], dim=1)
        else:
            img = img_update

        # Callback for progress
        if step_callback is not None:
            step_callback(
                PipelineIntermediateState(
                    step=i,
                    order=1,
                    total_steps=len(timesteps) - 1,
                    timestep=int(t_curr * 1000),
                    latents=img[:, -original_seq_len:] if reference_latents is not None else img,
                )
            )

    # Return only the generated image latents (exclude reference tokens)
    if reference_latents is not None:
        return img[:, -original_seq_len:]
    return img


def prepare_flux2_latents(
    noise: torch.Tensor,
    timestep: float,
    init_latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prepare latents for FLUX.2 denoising.

    For txt2img, returns pure noise. For img2img, mixes noise with
    encoded image latents according to the timestep.

    Args:
        noise: Random noise tensor.
        timestep: Initial timestep (1.0 for txt2img, < 1.0 for img2img).
        init_latents: Optional encoded image latents for img2img.

    Returns:
        Prepared latents ready for denoising.
    """
    if init_latents is None:
        return noise

    # For img2img: interpolate between noise and init_latents
    # At timestep=1.0, we want pure noise
    # At timestep=0.0, we want init_latents
    return timestep * noise + (1 - timestep) * init_latents
