import pytest
import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.lora_conversions.sdxl_lora_conversion_utils import (
    convert_sdxl_keys_to_diffusers_format,
    lora_model_from_sdxl_state_dict,
)

# A single UNet attention projection, expressed under both naming conventions. Both map to the same diffusers module.
STABILITY_MODULE = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"
DIFFUSERS_MODULE = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"

IN_FEATURES = 640
OUT_FEATURES = 640
RANK = 4


def _lora_keys(module_key: str, seed: int, rank: int = RANK) -> dict[str, torch.Tensor]:
    """Build the state_dict entries for one plain LoRA layer, with values that depend on `seed`."""
    generator = torch.Generator().manual_seed(seed)
    return {
        f"{module_key}.lora_down.weight": torch.randn(rank, IN_FEATURES, generator=generator),
        f"{module_key}.lora_up.weight": torch.randn(OUT_FEATURES, rank, generator=generator),
        f"{module_key}.alpha": torch.tensor(float(rank)),
    }


def _text_encoder_keys() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.randn(
            RANK, 768, generator=generator
        ),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": torch.randn(
            768, RANK, generator=generator
        ),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.alpha": torch.tensor(float(RANK)),
    }


@pytest.mark.parametrize("module_key", [STABILITY_MODULE, DIFFUSERS_MODULE], ids=["stability", "diffusers"])
def test_single_format_state_dict_produces_diffusers_keyed_lora_layers(module_key: str):
    state_dict = _lora_keys(module_key, seed=1) | _text_encoder_keys()

    model = lora_model_from_sdxl_state_dict(state_dict)

    # Both naming conventions must end up under the diffusers module key, and the text encoder passes through.
    assert set(model.layers) == {DIFFUSERS_MODULE, "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"}
    assert isinstance(model.layers[DIFFUSERS_MODULE], LoRALayer)


def test_duplicate_layer_is_merged_into_the_sum_of_both_patches():
    """A layer patched under both naming conventions must apply both patches, not silently drop one."""
    stability_sd = _lora_keys(STABILITY_MODULE, seed=1)
    diffusers_sd = _lora_keys(DIFFUSERS_MODULE, seed=2)

    merged_layer = lora_model_from_sdxl_state_dict(stability_sd | diffusers_sd).layers[DIFFUSERS_MODULE]
    assert isinstance(merged_layer, MergedLayerPatch)

    orig_parameters = {"weight": torch.zeros(OUT_FEATURES, IN_FEATURES)}
    expected = (
        lora_model_from_sdxl_state_dict(stability_sd).layers[DIFFUSERS_MODULE].get_parameters(orig_parameters, 1.0)
    )["weight"] + (
        lora_model_from_sdxl_state_dict(diffusers_sd).layers[DIFFUSERS_MODULE].get_parameters(orig_parameters, 1.0)
    )["weight"]

    torch.testing.assert_close(merged_layer.get_parameters(orig_parameters, 1.0)["weight"], expected)


def test_duplicate_layer_respects_the_patch_weight():
    stability_sd = _lora_keys(STABILITY_MODULE, seed=1)
    diffusers_sd = _lora_keys(DIFFUSERS_MODULE, seed=2)
    merged_layer = lora_model_from_sdxl_state_dict(stability_sd | diffusers_sd).layers[DIFFUSERS_MODULE]

    orig_parameters = {"weight": torch.zeros(OUT_FEATURES, IN_FEATURES)}
    full = merged_layer.get_parameters(orig_parameters, 1.0)["weight"]
    half = merged_layer.get_parameters(orig_parameters, 0.5)["weight"]

    torch.testing.assert_close(half, full * 0.5)


def test_duplicate_layers_with_different_ranks_are_merged():
    """The two halves of a merged file need not share a rank."""
    state_dict = _lora_keys(STABILITY_MODULE, seed=1, rank=4) | _lora_keys(DIFFUSERS_MODULE, seed=2, rank=16)

    layer = lora_model_from_sdxl_state_dict(state_dict).layers[DIFFUSERS_MODULE]

    assert isinstance(layer, MergedLayerPatch)
    assert layer.get_parameters({"weight": torch.zeros(OUT_FEATURES, IN_FEATURES)}, 1.0)["weight"].shape == (
        OUT_FEATURES,
        IN_FEATURES,
    )


def test_duplicate_layers_with_incompatible_shapes_raise():
    state_dict = _lora_keys(STABILITY_MODULE, seed=1)
    state_dict.update(
        {
            f"{DIFFUSERS_MODULE}.lora_down.weight": torch.randn(RANK, IN_FEATURES),
            f"{DIFFUSERS_MODULE}.lora_up.weight": torch.randn(OUT_FEATURES * 2, RANK),
            f"{DIFFUSERS_MODULE}.alpha": torch.tensor(float(RANK)),
        }
    )

    with pytest.raises(ValueError, match="incompatible weight shapes"):
        lora_model_from_sdxl_state_dict(state_dict)


def test_unrecognized_key_prefix_raises():
    with pytest.raises(ValueError, match="Unrecognized SDXL LoRA key prefix"):
        lora_model_from_sdxl_state_dict({"unet.down_blocks.1.attentions.0.lora_A.weight": torch.zeros(1)})


def test_strict_converter_still_rejects_mixed_key_formats():
    """`convert_sdxl_keys_to_diffusers_format()` cannot represent duplicate layers, so it must keep rejecting them."""
    state_dict = _lora_keys(STABILITY_MODULE, seed=1) | _lora_keys(DIFFUSERS_MODULE, seed=2)

    with pytest.raises(ValueError, match="could only be partially converted"):
        convert_sdxl_keys_to_diffusers_format(state_dict)


def test_identical_duplicate_layer_is_applied_only_once():
    """A key set that was copied rather than renamed must not be applied twice."""
    stability_sd = _lora_keys(STABILITY_MODULE, seed=1)
    diffusers_sd = {k.replace(STABILITY_MODULE, DIFFUSERS_MODULE): v for k, v in stability_sd.items()}

    layer = lora_model_from_sdxl_state_dict(stability_sd | diffusers_sd).layers[DIFFUSERS_MODULE]

    assert isinstance(layer, LoRALayer)
    orig_parameters = {"weight": torch.zeros(OUT_FEATURES, IN_FEATURES)}
    expected = (
        lora_model_from_sdxl_state_dict(stability_sd)
        .layers[DIFFUSERS_MODULE]
        .get_parameters(orig_parameters, 1.0)["weight"]
    )
    torch.testing.assert_close(layer.get_parameters(orig_parameters, 1.0)["weight"], expected)
