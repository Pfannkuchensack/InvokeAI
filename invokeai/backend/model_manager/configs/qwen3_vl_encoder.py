from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen3VLVariantType

# Text-tower hidden sizes of the two Qwen3-VL sizes we support. Both have 36 layers.
_QWEN3_VL_HIDDEN_SIZES = {
    2560: Qwen3VLVariantType.Qwen3VL_4B,
    4096: Qwen3VLVariantType.Qwen3VL_8B,
}
_QWEN3_VL_NUM_HIDDEN_LAYERS = 36


def _detect_qwen3_vl_variant(config_path: Path) -> Qwen3VLVariantType:
    """Classify a Qwen3-VL encoder by its text-tower hidden size.

    Krea-2 needs the 4B, Ideogram 4 the 8B; both tap hidden states up to layer 35, so a model with a
    different layer count is rejected outright. Recording the size as a variant (rather than
    rejecting everything but 4B, as this did while Krea-2 was the only consumer) is what lets the
    pickers offer each base the encoder it can actually use — a mismatch would otherwise surface as
    a shape error deep inside inference.
    """
    config = get_config_dict_or_raise(config_path)
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise NotAMatchError("Qwen3-VL text_config must be an object")

    num_hidden_layers = text_config.get("num_hidden_layers")
    if num_hidden_layers != _QWEN3_VL_NUM_HIDDEN_LAYERS:
        raise NotAMatchError(f"expected {_QWEN3_VL_NUM_HIDDEN_LAYERS} Qwen3-VL layers, got {num_hidden_layers}")

    hidden_size = text_config.get("hidden_size")
    variant = _QWEN3_VL_HIDDEN_SIZES.get(hidden_size) if isinstance(hidden_size, int) else None
    if variant is None:
        raise NotAMatchError(
            f"unsupported Qwen3-VL hidden size {hidden_size}; expected one of {sorted(_QWEN3_VL_HIDDEN_SIZES)}"
        )
    return variant


def _has_complete_pretrained_weights(weights_path: Path) -> bool:
    if (weights_path / "model.safetensors").is_file() or (weights_path / "pytorch_model.bin").is_file():
        return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = weights_path / index_name
        if not index_path.is_file():
            continue
        index = get_config_dict_or_raise(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        filenames = list(weight_map.values())
        if not all(isinstance(filename, str) and filename for filename in filenames):
            return False
        root = weights_path.resolve()
        referenced_files: set[Path] = set()
        for filename in filenames:
            filename_path = Path(filename)
            if filename_path.is_absolute():
                return False
            candidate = (weights_path / filename_path).resolve()
            if not candidate.is_relative_to(root):
                return False
            referenced_files.add(candidate)
        return bool(referenced_files) and all(path.is_file() for path in referenced_files)
    return False


def _detect_qwen3_vl_checkpoint_variant(state_dict: dict[str | int, Any]) -> Qwen3VLVariantType:
    """Classify a single-file Qwen3-VL encoder by the embedding's hidden size. See ``_detect_qwen3_vl_variant``."""
    embed_keys = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    embed = next((state_dict[key] for key in embed_keys if key in state_dict), None)
    shape = getattr(embed, "shape", ())
    hidden_size = shape[1] if len(shape) >= 2 else None
    variant = _QWEN3_VL_HIDDEN_SIZES.get(hidden_size) if isinstance(hidden_size, int) else None
    if variant is None:
        raise NotAMatchError(
            f"unsupported Qwen3-VL hidden size {hidden_size}; expected one of {sorted(_QWEN3_VL_HIDDEN_SIZES)}"
        )
    if not any(isinstance(key, str) and ".layers.35." in key for key in state_dict):
        raise NotAMatchError("Qwen3-VL encoder checkpoint is missing language-model layer 35")
    return variant


class Qwen3VLEncoder_Qwen3VLEncoder_Config(Config_Base):
    """Configuration for standalone Qwen3-VL text encoder models (diffusers-like directory format).

    Used by Krea-2 (4B) and Ideogram 4 (8B), whose text conditioning comes from a Qwen3-VL model
    (``Qwen3VLModel``). The model weights are expected either in a ``text_encoder`` subfolder of the
    model directory or directly at the root (standalone download). This is distinct from the text-only
    ``Qwen3Encoder`` (Z-Image / FLUX.2 Klein) and the Qwen2.5-VL ``QwenVLEncoder`` (Qwen Image).
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Qwen3VLEncoder] = Field(default=ModelFormat.Qwen3VLEncoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    # No default on purpose: a variant field with one would be folded into the discriminator tag.
    variant: Qwen3VLVariantType = Field(description="Qwen3-VL model size variant (4B or 8B)")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Exclude full pipeline models - these should be matched as main models, not just encoders.
        model_index_path = mod.path / "model_index.json"
        transformer_path = mod.path / "transformer"
        if model_index_path.exists() or transformer_path.exists():
            raise NotAMatchError(
                "directory looks like a full diffusers pipeline (has model_index.json or transformer folder), "
                "not a standalone Qwen3-VL encoder"
            )

        # Support both a nested text_encoder/config.json and a standalone config.json at the root.
        config_path_nested = mod.path / "text_encoder" / "config.json"
        config_path_direct = mod.path / "config.json"

        if config_path_nested.exists():
            expected_config_path = config_path_nested
        elif config_path_direct.exists():
            expected_config_path = config_path_direct
        else:
            raise NotAMatchError(f"unable to load config file: {config_path_nested} does not exist")

        # Qwen3-VL uses the Qwen3VLModel / Qwen3VLForConditionalGeneration architecture.
        raise_for_class_name(
            expected_config_path,
            {
                "Qwen3VLModel",
                "Qwen3VLForConditionalGeneration",
            },
        )
        variant = override_fields.pop("variant", None) or _detect_qwen3_vl_variant(expected_config_path)

        if config_path_nested.exists():
            weights_path = mod.path / "text_encoder"
            tokenizer_path = mod.path / "tokenizer"
        else:
            weights_path = mod.path
            tokenizer_path = mod.path

        has_weights = _has_complete_pretrained_weights(weights_path)
        has_tokenizer = (tokenizer_path / "tokenizer.json").exists() or (
            (tokenizer_path / "vocab.json").exists() and (tokenizer_path / "merges.txt").exists()
        )
        if not has_weights:
            raise NotAMatchError("standalone Qwen3-VL encoder directory does not contain model weights")
        if not has_tokenizer:
            raise NotAMatchError("standalone Qwen3-VL encoder directory does not contain tokenizer files")

        return cls(**override_fields, variant=variant)


def _is_qwen3_vl_encoder_state_dict(state_dict: dict[str | int, Any]) -> bool:
    """True for a single-file Qwen3-VL encoder: a Qwen3 text decoder PLUS a visual tower.

    The visual tower (``visual.*`` / ``model.visual.*``) distinguishes Qwen3-VL from the text-only
    ``Qwen3Encoder`` (Z-Image / FLUX.2 Klein), which has ``model.layers.*`` but no visual tower.
    """
    str_keys = [k for k in state_dict if isinstance(k, str)]
    has_text_decoder = any(".layers." in k and ("model." in k or k.startswith("layers.")) for k in str_keys)
    has_visual_tower = any(k.startswith(("visual.", "model.visual.")) or ".visual." in k for k in str_keys)
    return has_text_decoder and has_visual_tower


class Qwen3VLEncoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a single-file Qwen3-VL text encoder checkpoint (e.g. ComfyUI ``qwen3vl_4b_*``).

    Distinguished from the text-only ``Qwen3Encoder`` checkpoint (Z-Image) by the presence of the
    Qwen3-VL visual tower. Single-file checkpoints bundle neither config nor tokenizer; the loader
    pulls both from HuggingFace, picking the repo that matches ``variant``.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    # No default on purpose: a variant field with one would be folded into the discriminator tag.
    variant: Qwen3VLVariantType = Field(description="Qwen3-VL model size variant (4B or 8B)")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        state_dict = mod.load_state_dict()
        if not _is_qwen3_vl_encoder_state_dict(state_dict):
            raise NotAMatchError("state dict does not look like a single-file Qwen3-VL encoder")

        variant = override_fields.pop("variant", None) or _detect_qwen3_vl_checkpoint_variant(state_dict)

        return cls(**override_fields, variant=variant)
