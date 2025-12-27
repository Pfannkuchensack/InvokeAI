"""Sampling utilities for FLUX.2 models.

FLUX.2 uses different timestep scheduling with empirical mu calculation
and 32-channel latent space (vs 16 channels in FLUX.1).
"""

import math
from typing import Callable

import torch
from einops import rearrange


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for FLUX.2 timestep scheduling.

    This function calculates the mu parameter for timestep shifting based on
    the image sequence length and number of inference steps. The mu value
    controls how the noise schedule is shifted during denoising.

    Args:
        image_seq_len: Length of the image sequence (after packing).
        num_steps: Number of inference/denoising steps.

    Returns:
        The computed mu value for timestep scheduling.
    """
    # Constants derived from FLUX.2 training
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Apply time shifting to timesteps.

    Args:
        mu: Shift parameter (controls center of sigmoid).
        sigma: Scale parameter (controls steepness).
        t: Input timesteps tensor.

    Returns:
        Shifted timesteps.
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_flux2_schedule(
    num_steps: int,
    image_seq_len: int,
    shift: bool = True,
) -> list[float]:
    """Generate timestep schedule for FLUX.2.

    FLUX.2 uses empirical mu-based timestep shifting for improved
    sampling quality.

    Args:
        num_steps: Number of denoising steps.
        image_seq_len: Sequence length of packed image latents.
        shift: Whether to apply time shifting (True for FLUX.2-dev).

    Returns:
        List of timesteps from 1.0 to ~0.0.
    """
    timesteps = torch.linspace(1, 0, num_steps + 1)

    if shift:
        mu = compute_empirical_mu(image_seq_len, num_steps)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def get_flux2_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate noise for FLUX.2 latent space.

    FLUX.2 uses 32 latent channels (vs 16 in FLUX.1), and latents are
    packed into 2x2 patches.

    Args:
        num_samples: Batch size.
        height: Image height in pixels.
        width: Image width in pixels.
        device: Target device.
        dtype: Target dtype (typically bfloat16).
        seed: Random seed for reproducibility.

    Returns:
        Noise tensor of shape (num_samples, 32, h//8, w//8) where
        the spatial dimensions account for VAE downsampling.
    """
    # FLUX.2 has 32 latent channels
    latent_channels = 32

    # VAE downsamples by 8x, then we pack 2x2 patches
    latent_height = 2 * math.ceil(height / 16)
    latent_width = 2 * math.ceil(width / 16)

    # Generate noise on CPU for reproducibility, then move to device
    noise = torch.randn(
        num_samples,
        latent_channels,
        latent_height,
        latent_width,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    return noise.to(device=device, dtype=dtype)


def pack_flux2(x: torch.Tensor) -> torch.Tensor:
    """Pack latents for FLUX.2 transformer processing.

    Rearranges latents from (B, C, H, W) to (B, H*W/4, C*4) format
    for transformer input.

    Args:
        x: Latent tensor of shape (B, C, H, W).

    Returns:
        Packed tensor of shape (B, seq_len, hidden_dim).
    """
    # Pack 2x2 patches: (B, C, H, W) -> (B, H*W/4, C*4)
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_flux2(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack latents from FLUX.2 transformer output.

    Rearranges latents from (B, seq_len, hidden_dim) back to (B, C, H, W).

    Args:
        x: Packed tensor of shape (B, seq_len, hidden_dim).
        height: Target height (in latent space, after packing).
        width: Target width (in latent space, after packing).

    Returns:
        Unpacked tensor of shape (B, C, H, W).
    """
    # Unpack 2x2 patches: (B, H*W/4, C*4) -> (B, C, H, W)
    return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=height, w=width, ph=2, pw=2)


def generate_img_ids(
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate image position IDs for FLUX.2 transformer.

    Creates position embeddings for the packed image latents.

    Args:
        height: Latent height (after packing).
        width: Latent width (after packing).
        batch_size: Batch size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Position IDs tensor of shape (batch_size, seq_len, 3).
    """
    latent_height = height // 2
    latent_width = width // 2

    # Create grid of positions
    latent_image_ids = torch.zeros(latent_height, latent_width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = (
        torch.arange(latent_height, device=device, dtype=dtype)[:, None]
        .expand(-1, latent_width)
    )
    latent_image_ids[..., 2] = (
        torch.arange(latent_width, device=device, dtype=dtype)[None, :]
        .expand(latent_height, -1)
    )

    latent_image_ids = latent_image_ids.reshape(1, latent_height * latent_width, 3)
    latent_image_ids = latent_image_ids.expand(batch_size, -1, -1)

    return latent_image_ids


def clip_timestep_schedule_fractional(
    timesteps: list[float],
    denoising_start: float,
    denoising_end: float,
) -> list[float]:
    """Clip timestep schedule for partial denoising (img2img).

    Args:
        timesteps: Full list of timesteps from 1.0 to 0.0.
        denoising_start: Start fraction (0.0 = full noise, 1.0 = no noise).
        denoising_end: End fraction (0.0 = full denoise, 1.0 = no denoise).

    Returns:
        Clipped timestep schedule.
    """
    if denoising_start == 0.0 and denoising_end == 1.0:
        return timesteps

    # Find indices for start and end
    total_steps = len(timesteps) - 1
    start_idx = int(denoising_start * total_steps)
    end_idx = int(denoising_end * total_steps)

    return timesteps[start_idx : end_idx + 1]
