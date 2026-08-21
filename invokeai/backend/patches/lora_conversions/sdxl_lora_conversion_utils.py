import bisect
from typing import Dict, List, Tuple, TypeVar

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util import InvokeAILogger

T = TypeVar("T")

# The two UNet key naming conventions that an SDXL LoRA may use. Used to group the keys of a state_dict that mixes both.
_STABILITY = "stability"
_DIFFUSERS = "diffusers"


def convert_sdxl_keys_to_diffusers_format(state_dict: Dict[str, T]) -> dict[str, T]:
    """Convert the keys of an SDXL LoRA state_dict to diffusers format.

    The input state_dict can be in either Stability AI format or diffusers format. If the state_dict is already in
    diffusers format, then this function will have no effect.

    This function is adapted from:
    https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L385-L409

    Args:
        state_dict (Dict[str, Tensor]): The SDXL LoRA state_dict.

    Raises:
        ValueError: If state_dict contains an unrecognized key, or not all keys could be converted.

    Returns:
        Dict[str, Tensor]: The diffusers-format state_dict.
    """
    converted_count = 0  # The number of Stability AI keys converted to diffusers format.
    not_converted_count = 0  # The number of keys that were not converted.

    stability_unet_keys = _sorted_stability_unet_keys()

    new_state_dict: dict[str, T] = {}
    for full_key, value in state_dict.items():
        new_key, source_format = _convert_sdxl_key(full_key, stability_unet_keys)
        new_state_dict[new_key] = value
        if full_key.startswith("lora_unet_"):
            if source_format == _STABILITY:
                converted_count += 1
            else:
                not_converted_count += 1

    if converted_count > 0 and not_converted_count > 0:
        raise ValueError(
            f"The SDXL LoRA could only be partially converted to diffusers format. converted={converted_count},"
            f" not_converted={not_converted_count}"
        )

    return new_state_dict


def lora_model_from_sdxl_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Build a ModelPatchRaw from an SDXL LoRA state_dict in Stability AI format, diffusers format, or a mix of both.

    Unlike `convert_sdxl_keys_to_diffusers_format()`, this function accepts state_dicts that patch the same UNet layer
    under both naming conventions. Such files exist in the wild: they are produced by merge tools that concatenate the
    key sets of two LoRAs instead of normalizing the names and summing the weights. Both halves hold real - and
    different - weights, so neither can be dropped. We apply both, which is the merge that the file was trying to
    express. Loaders that key their patches by layer name instead silently apply whichever half they see last.

    Args:
        state_dict (Dict[str, torch.Tensor]): The SDXL LoRA state_dict.

    Raises:
        ValueError: If state_dict contains an unrecognized key prefix, or if two patches for the same layer have
            incompatible shapes.

    Returns:
        ModelPatchRaw: The patch model, with diffusers-format layer keys.
    """
    stability_unet_keys = _sorted_stability_unet_keys()

    # module key -> source format -> the module's sub-state_dict. The inner dict holds two entries only for modules
    # that this state_dict patches twice.
    grouped_state_dict: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for full_key, value in state_dict.items():
        new_key, source_format = _convert_sdxl_key(full_key, stability_unet_keys)
        module_key, param_name = new_key.split(".", 1)
        grouped_state_dict.setdefault(module_key, {}).setdefault(source_format, {})[param_name] = value

    layers: dict[str, BaseLayerPatch] = {}
    merged_module_keys: list[str] = []
    duplicated_module_count = 0
    for module_key, sub_state_dicts_by_format in grouped_state_dict.items():
        sub_state_dicts = list(sub_state_dicts_by_format.values())
        if len(sub_state_dicts) == 1:
            layers[module_key] = any_lora_layer_from_state_dict(sub_state_dicts[0])
            continue

        if _all_values_are_equal(sub_state_dicts):
            # The same patch is simply stored under both naming conventions - the key set was copied rather than
            # renamed. Summing would double the layer's strength, so keep a single copy.
            layers[module_key] = any_lora_layer_from_state_dict(sub_state_dicts[0])
            duplicated_module_count += 1
            continue

        _assert_layer_shapes_are_compatible(module_key, sub_state_dicts)
        # `None` ranges: every sub-layer patches the full module weight, so their residuals are summed rather than
        # written to disjoint slices.
        layers[module_key] = MergedLayerPatch(
            [any_lora_layer_from_state_dict(sd) for sd in sub_state_dicts], [None] * len(sub_state_dicts)
        )
        merged_module_keys.append(module_key)

    logger = InvokeAILogger.get_logger(__name__)
    if duplicated_module_count:
        logger.info(
            "This SDXL LoRA stores %d layer(s) under both the Stability AI and diffusers key naming conventions with "
            "identical weights. The redundant copies are ignored.",
            duplicated_module_count,
        )
    if merged_module_keys:
        logger.warning(
            "This SDXL LoRA patches %d layer(s) twice with different weights: once in Stability AI key format and "
            "once in diffusers key format (e.g. '%s'). This usually means the file was merged by a tool that did not "
            "normalize its key names. Both patches are applied, so the LoRA may act stronger than expected - try a "
            "lower weight.",
            len(merged_module_keys),
            merged_module_keys[0],
        )

    return ModelPatchRaw(layers=layers)


def _sorted_stability_unet_keys() -> List[str]:
    """Get a sorted list of Stability AI UNet keys so that we can efficiently search for keys with matching prefixes.

    For example, we want to efficiently find `input_blocks_4_1` in the list when searching for
    `input_blocks_4_1_proj_in`.
    """
    stability_unet_keys = list(SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP)
    stability_unet_keys.sort()
    return stability_unet_keys


def _convert_sdxl_key(full_key: str, stability_unet_keys: List[str]) -> Tuple[str, str]:
    """Convert a single SDXL LoRA key to diffusers format.

    Returns:
        Tuple[str, str]: The converted key, and the naming convention that the input key was already in.
    """
    if full_key.startswith("lora_unet_"):
        search_key = full_key.replace("lora_unet_", "")
        # Use bisect to find the key in stability_unet_keys that *may* match the search_key's prefix.
        position = bisect.bisect_right(stability_unet_keys, search_key)
        map_key = stability_unet_keys[position - 1]
        # Now, check if the map_key *actually* matches the search_key.
        if search_key.startswith(map_key):
            return full_key.replace(map_key, SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP[map_key]), _STABILITY
        return full_key, _DIFFUSERS
    if full_key.startswith("lora_te1_") or full_key.startswith("lora_te2_"):
        # The CLIP text encoders have the same keys in both Stability AI and diffusers formats.
        return full_key, _DIFFUSERS
    raise ValueError(f"Unrecognized SDXL LoRA key prefix: '{full_key}'.")


def _all_values_are_equal(sub_state_dicts: List[Dict[str, torch.Tensor]]) -> bool:
    """Check whether every sub-state_dict holds exactly the same tensors."""
    first, *rest = sub_state_dicts
    return all(sd.keys() == first.keys() and all(torch.equal(sd[name], first[name]) for name in first) for sd in rest)


def _assert_layer_shapes_are_compatible(module_key: str, sub_state_dicts: List[Dict[str, torch.Tensor]]) -> None:
    """Raise if the patches for a doubly-patched module do not target the same weight shape.

    Only plain LoRA sub-layers are checked: that is what these merged files contain in practice, and for other layer
    types the target shape cannot be derived from the state_dict alone. A mismatch means the file is broken in a way
    that summing cannot fix, and failing here is clearer than failing later inside the patcher.
    """
    shapes = {
        (sd["lora_up.weight"].shape[0], tuple(sd["lora_down.weight"].shape[1:]))
        for sd in sub_state_dicts
        if "lora_up.weight" in sd and "lora_down.weight" in sd
    }
    if len(shapes) > 1:
        raise ValueError(
            f"The SDXL LoRA patches '{module_key}' more than once with incompatible weight shapes: {sorted(shapes)}."
        )


# code from
# https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L15C1-L97C32
def _make_sdxl_unet_conversion_map() -> List[Tuple[str, str]]:
    """Create a dict mapping state_dict keys from Stability AI SDXL format to diffusers SDXL format."""
    unet_conversion_map_layer: list[tuple[str, str]] = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map: list[tuple[str, str]] = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j + 1}."
        sd_time_embed_prefix = f"time_embed.{j * 2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j + 1}."
        sd_label_embed_prefix = f"label_emb.0.{j * 2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


SDXL_UNET_STABILITY_TO_DIFFUSERS_MAP = {
    sd.rstrip(".").replace(".", "_"): hf.rstrip(".").replace(".", "_") for sd, hf in _make_sdxl_unet_conversion_map()
}
