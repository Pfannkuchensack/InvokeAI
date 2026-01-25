# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""ComfyUI quantization format dequantization utilities.

Supports nvfp4 (NVidia FP4) and float8_e4m3fn formats used by ComfyUI.
"""

import json
from typing import Any

import torch


# NVidia FP4 (E2M1) lookup table
# 4-bit values: 0-15 map to specific float values
# Format: E2M1 with sign bit
NVFP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NVidia FP4 (E2M1) weights to target dtype.

    Args:
        weight: Packed uint8 tensor [out_features, in_features/2]
        weight_scale: Block-wise scale [out_features, num_blocks] in float8_e4m3fn
        weight_scale_2: Global scale scalar
        target_dtype: Target dtype for output

    Returns:
        Dequantized tensor [out_features, in_features]
    """
    out_features = weight.shape[0]
    packed_in = weight.shape[1]
    in_features = packed_in * 2

    # Unpack uint8 to 2x 4-bit values
    # Low nibble (bits 0-3) and high nibble (bits 4-7)
    low_nibble = weight & 0x0F
    high_nibble = (weight >> 4) & 0x0F

    # Use lookup table to convert FP4 to float
    lut = NVFP4_LUT.to(weight.device)
    low_float = lut[low_nibble.long()]
    high_float = lut[high_nibble.long()]

    # Interleave to get original order: [low0, high0, low1, high1, ...]
    dequant = torch.empty(out_features, in_features, dtype=torch.float32, device=weight.device)
    dequant[:, 0::2] = low_float
    dequant[:, 1::2] = high_float

    # Apply block-wise scaling
    # weight_scale shape: [out_features, num_blocks]
    # Each block covers in_features/num_blocks values
    num_blocks = weight_scale.shape[1]
    block_size = in_features // num_blocks

    # Convert float8_e4m3fn scale to float32
    scale = weight_scale.to(torch.float32)

    # Expand scale to match weight dimensions
    # scale: [out_features, num_blocks] -> [out_features, in_features]
    scale_expanded = scale.repeat_interleave(block_size, dim=1)

    # Apply scales
    dequant = dequant * scale_expanded * weight_scale_2.item()

    return dequant.to(target_dtype)


def dequantize_float8_e4m3fn(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize float8_e4m3fn weights to target dtype.

    Args:
        weight: float8_e4m3fn tensor
        weight_scale: Scale scalar
        target_dtype: Target dtype for output

    Returns:
        Dequantized tensor
    """
    # Convert to float32 first, then apply scale
    dequant = weight.to(torch.float32) * weight_scale.item()
    return dequant.to(target_dtype)


def dequantize_comfy_state_dict(
    sd: dict[str, Any],
    target_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Dequantize a ComfyUI quantized state dict.

    Detects quantized weights by looking for .comfy_quant keys and dequantizes
    them according to their format (nvfp4 or float8_e4m3fn).

    Args:
        sd: State dict with quantized weights
        target_dtype: Target dtype for dequantized weights

    Returns:
        State dict with all weights dequantized to target_dtype
    """
    result: dict[str, torch.Tensor] = {}

    # Find all quantized weight bases (keys that have .comfy_quant)
    quant_bases: set[str] = set()
    for key in sd.keys():
        if key.endswith(".comfy_quant"):
            base = key[: -len(".comfy_quant")]
            quant_bases.add(base)

    # Process each key
    processed_bases: set[str] = set()
    for key in sd.keys():
        # Skip metadata keys
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale") or key.endswith(".weight_scale_2"):
            continue

        # Check if this is a quantized weight
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        if base and base in quant_bases:
            if base in processed_bases:
                continue
            processed_bases.add(base)

            # Get quantization format
            quant_info = sd[f"{base}.comfy_quant"]
            quant_str = bytes(quant_info.tolist()).decode("utf-8", errors="ignore").strip("\x00")
            quant_format = json.loads(quant_str).get("format", "")

            weight = sd[f"{base}.weight"]
            weight_scale = sd.get(f"{base}.weight_scale")
            weight_scale_2 = sd.get(f"{base}.weight_scale_2")

            if quant_format == "nvfp4":
                if weight_scale is None or weight_scale_2 is None:
                    raise ValueError(f"nvfp4 weight {base} missing scale tensors")
                dequant = dequantize_nvfp4(weight, weight_scale, weight_scale_2, target_dtype)
            elif quant_format == "float8_e4m3fn":
                if weight_scale is None:
                    raise ValueError(f"float8_e4m3fn weight {base} missing scale tensor")
                dequant = dequantize_float8_e4m3fn(weight, weight_scale, target_dtype)
            else:
                raise ValueError(f"Unknown ComfyUI quantization format: {quant_format}")

            result[f"{base}.weight"] = dequant
        else:
            # Non-quantized weight, just convert dtype if needed
            tensor = sd[key]
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                result[key] = tensor.to(target_dtype)
            else:
                result[key] = tensor

    return result


def is_comfy_quantized_state_dict(sd: dict[str, Any]) -> bool:
    """Check if a state dict uses ComfyUI quantization format."""
    return any(key.endswith(".comfy_quant") for key in sd.keys())
