"""Mistral Encoder model configuration for FLUX.2.

FLUX.2 uses Mistral Small 3.1 as its text encoder instead of CLIP + T5.
"""

from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _name_looks_like_mistral(name: str) -> bool:
    """Check if filename/path suggests this is a Mistral model.

    Since Mistral and Qwen3 have similar state dict structures,
    we need to rely on naming conventions for checkpoint files.
    """
    name_lower = name.lower()
    return "mistral" in name_lower


def _has_llm_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains LLM model keys (common to Mistral, Qwen, etc.)."""
    pytorch_indicators = ["model.layers.0.", "model.embed_tokens.weight"]

    for key in state_dict.keys():
        if isinstance(key, str):
            for indicator in pytorch_indicators:
                if key.startswith(indicator) or key == indicator:
                    return True
    return False


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains GGML tensors (GGUF quantized)."""
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


class MistralEncoder_Diffusers_Config(Config_Base):
    """Configuration for Mistral Encoder models in diffusers format.

    Used for FLUX.2 text encoding with Mistral Small 3.1.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check for config.json with Mistral architecture
        config_path = mod.path / "config.json"
        if not config_path.exists():
            raise NotAMatchError(f"config.json not found at {mod.path}")

        # Mistral uses Mistral3ForConditionalGeneration
        raise_for_class_name(
            config_path,
            {
                "Mistral3ForConditionalGeneration",
                "MistralForCausalLM",
            },
        )

        return cls(**override_fields)


class MistralEncoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for single-file Mistral Encoder models (safetensors)."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_mistral_model(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_mistral_model(cls, mod: ModelOnDisk) -> None:
        # First check name - since LLM state dicts are similar
        if not _name_looks_like_mistral(mod.path.name):
            raise NotAMatchError("filename does not suggest a Mistral model")

        # Then verify it has LLM keys
        has_llm_keys = _has_llm_keys(mod.load_state_dict())
        if not has_llm_keys:
            raise NotAMatchError("state dict does not look like an LLM model")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml:
            raise NotAMatchError("state dict looks like GGUF quantized")


class MistralEncoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for GGUF-quantized Mistral Encoder models."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_mistral_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_mistral_model(cls, mod: ModelOnDisk) -> None:
        # First check name
        if not _name_looks_like_mistral(mod.path.name):
            raise NotAMatchError("filename does not suggest a Mistral model")

        # Then verify it has LLM keys
        has_llm_keys = _has_llm_keys(mod.load_state_dict())
        if not has_llm_keys:
            raise NotAMatchError("state dict does not look like an LLM model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml:
            raise NotAMatchError("state dict does not look like GGUF quantized")
