"""Flux2 Klein KV Denoising Function.

This module provides the KV-cached denoising function for FLUX.2 Klein 9B KV models.

On the first denoising step, reference image tokens are included in the full transformer
forward pass and their post-RoPE attention K/V projections are cached per-layer.
On subsequent steps, only the noise latents are forwarded and the cached reference K/V
are injected during attention, avoiding redundant recomputation of reference tokens.

This provides up to 2.5x speedup for multi-reference editing tasks compared to the
standard denoise function which recomputes reference tokens at every step.

NOTE: This requires diffusers with Flux2KleinKVPipeline support (KV-cache-aware
attention processors and extended Flux2Transformer2DModel.forward() parameters).
"""

import inspect
import math
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise_kv(
    model: torch.nn.Module,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    cfg_scale: list[float],
    # Negative conditioning for CFG
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    # Scheduler for stepping (e.g., FlowMatchEulerDiscreteScheduler)
    scheduler: Any = None,
    # Dynamic shifting parameter for FLUX.2 Klein (computed from image resolution)
    mu: float | None = None,
    # Inpainting extension for merging latents during denoising
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
    # Reference image conditioning for KV caching
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Denoise latents using a FLUX.2 Klein 9B KV transformer model with KV-cached reference images.

    Unlike the standard denoise function which concatenates reference tokens to every step,
    this function uses a KV-cache approach:
    - Step 0 ("extract"): Reference tokens are prepended to the noise latents. The transformer
      performs a full forward pass and caches the K/V projections for reference tokens per-layer.
    - Steps 1+ ("cached"): Only noise latents are forwarded. Cached reference K/V are injected
      during attention, skipping redundant reference token computation.
    - No reference images: Falls back to standard forward pass (no KV caching).

    The KV caching provides up to 2.5x speedup for multi-reference editing tasks.

    Note: This requires diffusers with Flux2Transformer2DModel KV-cache support
    (kv_cache, kv_cache_mode, num_ref_tokens, ref_fixed_timestep parameters).

    Args:
        model: The Flux2Transformer2DModel from diffusers (with KV-cache support).
        img: Packed latent image tensor of shape (B, seq_len, channels).
        img_ids: Image position IDs tensor.
        txt: Text encoder hidden states (Qwen3 embeddings).
        txt_ids: Text position IDs tensor.
        timesteps: List of timesteps for denoising schedule (linear sigmas from 1.0 to 1/n).
        step_callback: Callback function for progress updates.
        cfg_scale: List of CFG scale values per step.
        neg_txt: Negative text embeddings for CFG (optional).
        neg_txt_ids: Negative text position IDs (optional).
        scheduler: Optional diffusers scheduler (Euler, Heun, LCM). If None, uses manual Euler.
        mu: Dynamic shifting parameter computed from image resolution.
        img_cond_seq: Packed reference image latents of shape (B, ref_seq_len, channels).
        img_cond_seq_ids: Reference image position IDs tensor.

    Returns:
        Denoised latent tensor.
    """
    total_steps = len(timesteps) - 1
    has_ref_images = img_cond_seq is not None and img_cond_seq_ids is not None
    num_ref_tokens = img_cond_seq.shape[1] if has_ref_images else 0

    # Klein has guidance_embeds=False, but the transformer forward() still requires a guidance tensor
    guidance = torch.full((img.shape[0],), 1.0, device=img.device, dtype=img.dtype)

    # KV cache will be populated on step 0 and reused on subsequent steps
    kv_cache = None

    def _model_forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        t_vec: torch.Tensor,
        img_ids_arg: torch.Tensor,
        txt_ids_arg: torch.Tensor,
        kv_cache_mode: str | None = None,
        kv_cache_arg: Any = None,
        num_ref_tokens_arg: int = 0,
    ):
        """Run the transformer forward pass with optional KV-cache parameters."""
        kwargs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": t_vec,
            "img_ids": img_ids_arg,
            "txt_ids": txt_ids_arg,
            "guidance": guidance,
            "return_dict": False,
        }

        # Add KV-cache parameters when available
        if kv_cache_mode is not None:
            kwargs["kv_cache_mode"] = kv_cache_mode
        if kv_cache_arg is not None:
            kwargs["kv_cache"] = kv_cache_arg
        if num_ref_tokens_arg > 0:
            kwargs["num_ref_tokens"] = num_ref_tokens_arg

        return model(**kwargs)

    if scheduler is not None:
        # Scheduler-based stepping
        sigmas = np.array(timesteps[:-1], dtype=np.float32)

        set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
        supports_sigmas = "sigmas" in set_timesteps_sig.parameters
        if supports_sigmas and mu is not None:
            scheduler.set_timesteps(sigmas=sigmas.tolist(), mu=mu, device=img.device)
        elif supports_sigmas:
            scheduler.set_timesteps(sigmas=sigmas.tolist(), device=img.device)
        else:
            scheduler.set_timesteps(num_inference_steps=len(sigmas), device=img.device)

        num_scheduler_steps = len(scheduler.timesteps)
        user_step = 0

        pbar = tqdm(total=total_steps, desc="Denoising (KV)")
        for step_index in range(num_scheduler_steps):
            timestep = scheduler.timesteps[step_index]
            t_curr = timestep.item() / scheduler.config.num_train_timesteps
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Determine KV-cache mode for this step
            if has_ref_images and step_index == 0:
                # Step 0: include ref tokens, extract KV cache
                latent_model_input = torch.cat([img_cond_seq, img], dim=1)
                latent_image_ids = torch.cat([img_cond_seq_ids, img_ids], dim=1)

                output = _model_forward(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=latent_image_ids,
                    txt_ids_arg=txt_ids,
                    kv_cache_mode="extract",
                    num_ref_tokens_arg=num_ref_tokens,
                )

                # output is (sample_tuple, kv_cache) when kv_cache_mode="extract"
                if isinstance(output, tuple) and len(output) == 2 and not isinstance(output[1], torch.Tensor):
                    pred = output[0][0] if isinstance(output[0], tuple) else output[0]
                    kv_cache = output[1]
                else:
                    pred = output[0] if isinstance(output, tuple) else output

            elif has_ref_images and kv_cache is not None:
                # Steps 1+: use cached ref KV, no ref tokens in input
                output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=txt_ids,
                    kv_cache_mode="cached",
                    kv_cache_arg=kv_cache,
                )
                pred = output[0] if isinstance(output, tuple) else output

            else:
                # No reference images: standard forward
                output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=txt_ids,
                )
                pred = output[0] if isinstance(output, tuple) else output

            step_cfg_scale = cfg_scale[min(user_step, len(cfg_scale) - 1)]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                # For CFG negative pass, always use standard forward (no KV caching for negative)
                neg_output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                )
                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Use scheduler.step() for the update
            step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img)
            img = step_output.prev_sample

            # Get t_prev for inpainting (next sigma value)
            if step_index + 1 < len(scheduler.sigmas):
                t_prev = scheduler.sigmas[step_index + 1].item()
            else:
                t_prev = 0.0

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

            user_step += 1
            if user_step <= total_steps:
                pbar.update(1)
                preview_img = img - t_curr * pred
                if inpaint_extension is not None:
                    preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)
                step_callback(
                    PipelineIntermediateState(
                        step=user_step,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t_curr * 1000),
                        latents=preview_img,
                    ),
                )

        pbar.close()
    else:
        # Manual Euler stepping
        for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Determine KV-cache mode for this step
            if has_ref_images and step_index == 0:
                # Step 0: include ref tokens, extract KV cache
                latent_model_input = torch.cat([img_cond_seq, img], dim=1)
                latent_image_ids = torch.cat([img_cond_seq_ids, img_ids], dim=1)

                output = _model_forward(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=latent_image_ids,
                    txt_ids_arg=txt_ids,
                    kv_cache_mode="extract",
                    num_ref_tokens_arg=num_ref_tokens,
                )

                if isinstance(output, tuple) and len(output) == 2 and not isinstance(output[1], torch.Tensor):
                    pred = output[0][0] if isinstance(output[0], tuple) else output[0]
                    kv_cache = output[1]
                else:
                    pred = output[0] if isinstance(output, tuple) else output

            elif has_ref_images and kv_cache is not None:
                # Steps 1+: use cached ref KV
                output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=txt_ids,
                    kv_cache_mode="cached",
                    kv_cache_arg=kv_cache,
                )
                pred = output[0] if isinstance(output, tuple) else output

            else:
                # No reference images: standard forward
                output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=txt_ids,
                )
                pred = output[0] if isinstance(output, tuple) else output

            step_cfg_scale = cfg_scale[step_index]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output = _model_forward(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    t_vec=t_vec,
                    img_ids_arg=img_ids,
                    txt_ids_arg=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                )
                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Euler step
            preview_img = img - t_curr * pred
            img = img + (t_prev - t_curr) * pred

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)
                preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

            step_callback(
                PipelineIntermediateState(
                    step=step_index + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t_curr),
                    latents=preview_img,
                ),
            )

    # Clean up KV cache
    if kv_cache is not None and hasattr(kv_cache, "clear"):
        kv_cache.clear()

    return img
