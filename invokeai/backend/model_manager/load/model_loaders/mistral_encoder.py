# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Mistral Encoder model loading in InvokeAI.

Used for FLUX.2 Dev text encoding with Mistral Small 3.1.
"""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoProcessor, AutoTokenizer

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Checkpoint_Config,
    MistralEncoder_Diffusers_Config,
    MistralEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.devices import TorchDevice


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.Diffusers)
class MistralEncoderDiffusersLoader(ModelLoader):
    """Class to load standalone Mistral Encoder models for FLUX.2 Dev (directory format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_Diffusers_Config):
            raise ValueError("Only MistralEncoder_Diffusers_Config models are supported here.")

        model_path = Path(config.path)

        # Support both structures:
        # 1. Full model: model_root/text_encoder/ and model_root/tokenizer/
        # 2. Standalone download: model_root/ contains text_encoder files directly
        text_encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"

        # Check if this is a standalone text_encoder download (no nested text_encoder folder)
        is_standalone = not text_encoder_path.exists() and (model_path / "config.json").exists()

        if is_standalone:
            text_encoder_path = model_path
            tokenizer_path = model_path  # Tokenizer/processor files should also be in root

        match submodel_type:
            case SubModelType.Tokenizer:
                # Mistral uses AutoProcessor which wraps the tokenizer
                # Try AutoProcessor first, fall back to AutoTokenizer
                try:
                    return AutoProcessor.from_pretrained(tokenizer_path, local_files_only=True)
                except Exception:
                    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            case SubModelType.TextEncoder:
                # Determine safe dtype based on target device capabilities
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

                # Mistral 3 uses Mistral3ForConditionalGeneration
                # Fall back to MistralForCausalLM for older models
                try:
                    from transformers import Mistral3ForConditionalGeneration

                    return Mistral3ForConditionalGeneration.from_pretrained(
                        text_encoder_path,
                        torch_dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                    )
                except (ImportError, OSError):
                    # Fall back to MistralForCausalLM if Mistral3 is not available
                    from transformers import MistralForCausalLM

                    return MistralForCausalLM.from_pretrained(
                        text_encoder_path,
                        torch_dtype=model_dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                    )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.Checkpoint)
class MistralEncoderCheckpointLoader(ModelLoader):
    """Class to load single-file Mistral Encoder models (safetensors format)."""

    # Default HuggingFace model to load tokenizer from when using single-file Mistral encoder
    DEFAULT_TOKENIZER_SOURCE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_Checkpoint_Config):
            raise ValueError("Only MistralEncoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_singlefile(config)
            case SubModelType.Tokenizer:
                # For single-file Mistral, load tokenizer from HuggingFace
                return self._load_tokenizer_with_offline_fallback()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_tokenizer_with_offline_fallback(self) -> AnyModel:
        """Load tokenizer with local_files_only fallback for offline support."""
        try:
            # Try AutoProcessor first for Mistral 3
            try:
                return AutoProcessor.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE, local_files_only=True)
            except Exception:
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE, local_files_only=True)
        except OSError:
            # Not in cache yet, download from HuggingFace
            try:
                return AutoProcessor.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)
            except Exception:
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.quantization.comfy_quant import dequantize_comfy_state_dict, is_comfy_quantized_state_dict
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, MistralEncoder_Checkpoint_Config):
            raise TypeError(
                f"Expected MistralEncoder_Checkpoint_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the state dict from safetensors file
        sd = load_file(model_path)

        # Check for ComfyUI quantized format and dequantize if needed
        if is_comfy_quantized_state_dict(sd):
            logger.info("Detected ComfyUI quantized format (nvfp4/float8_e4m3fn mixed), dequantizing...")
            sd = dequantize_comfy_state_dict(sd, target_dtype=model_dtype)
            logger.info("Dequantization complete")

        # Detect Mistral configuration from state dict
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get vocab size and hidden size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        vocab_size = embed_weight.shape[0]
        hidden_size = embed_weight.shape[1]

        # Detect intermediate_size from MLP weights
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")
        if gate_proj_weight is not None:
            intermediate_size = gate_proj_weight.shape[0]
        else:
            intermediate_size = hidden_size * 4  # Default fallback

        # Detect attention parameters from attention weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")

        # Default head_dim for Mistral models
        head_dim = 128

        if q_proj_weight is not None and k_proj_weight is not None:
            # q_proj shape: [num_heads * head_dim, hidden_size]
            # k_proj shape: [num_kv_heads * head_dim, hidden_size]
            q_out_features = q_proj_weight.shape[0]
            k_out_features = k_proj_weight.shape[0]

            # Estimate head_dim (commonly 128 for Mistral models)
            # Try common head dimensions
            for hd in [128, 64, 96, 256]:
                if q_out_features % hd == 0 and k_out_features % hd == 0:
                    head_dim = hd
                    num_attention_heads = q_out_features // head_dim
                    num_key_value_heads = k_out_features // head_dim
                    break
            else:
                # Fallback: assume head_dim = hidden_size / 32
                head_dim = max(hidden_size // 32, 64)
                num_attention_heads = q_out_features // head_dim
                num_key_value_heads = k_out_features // head_dim
        else:
            # Default values for Mistral Small 3.1 24B
            num_attention_heads = 32
            num_key_value_heads = 8

        logger.info(
            f"Mistral config: hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
            f"layers={layer_count}, heads={num_attention_heads}, kv_heads={num_key_value_heads}, "
            f"head_dim={head_dim}, vocab_size={vocab_size}"
        )

        # Create Mistral config and model
        from transformers import MistralConfig, MistralForCausalLM

        mistral_config = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            torch_dtype=model_dtype,
        )

        # Handle memory management
        new_sd_size = sum([ten.nelement() * model_dtype.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)

        # Convert to target dtype (if not already done during dequantization)
        for k in sd.keys():
            if sd[k].dtype != model_dtype:
                sd[k] = sd[k].to(model_dtype)

        # Create model with empty weights and load state dict
        with accelerate.init_empty_weights():
            model = MistralForCausalLM(mistral_config)

        model.load_state_dict(sd, strict=False, assign=True)

        # Handle tied weights
        if mistral_config.tie_word_embeddings:
            model.tie_weights()

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.GGUFQuantized)
class MistralEncoderGGUFLoader(ModelLoader):
    """Class to load GGUF-quantized Mistral Encoder models."""

    # Default HuggingFace model to load tokenizer from when using GGUF Mistral encoder
    DEFAULT_TOKENIZER_SOURCE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MistralEncoder_GGUF_Config):
            raise ValueError("Only MistralEncoder_GGUF_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_gguf(config)
            case SubModelType.Tokenizer:
                return self._load_tokenizer_with_offline_fallback()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_tokenizer_with_offline_fallback(self) -> AnyModel:
        """Load tokenizer with local_files_only fallback for offline support."""
        try:
            try:
                return AutoProcessor.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE, local_files_only=True)
            except Exception:
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE, local_files_only=True)
        except OSError:
            try:
                return AutoProcessor.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)
            except Exception:
                return AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_SOURCE)

    def _load_from_gguf(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from transformers import MistralConfig, MistralForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, MistralEncoder_GGUF_Config):
            raise TypeError(
                f"Expected MistralEncoder_GGUF_Config, got {type(config).__name__}. Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        # Check if this is llama.cpp format and convert if needed
        is_llamacpp_format = any(k.startswith("blk.") for k in sd.keys() if isinstance(k, str))
        if is_llamacpp_format:
            logger.info("Detected llama.cpp GGUF format, converting keys to PyTorch format")
            sd = self._convert_llamacpp_to_pytorch(sd)

        # Detect configuration from state dict
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get dimensions from embed_tokens
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        embed_shape = embed_weight.shape if hasattr(embed_weight, "shape") else embed_weight.tensor_shape
        vocab_size = embed_shape[0]
        hidden_size = embed_shape[1]

        # Detect intermediate_size from MLP weights
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")
        if gate_proj_weight is not None:
            gate_shape = gate_proj_weight.shape if hasattr(gate_proj_weight, "shape") else gate_proj_weight.tensor_shape
            intermediate_size = gate_shape[0]
        else:
            intermediate_size = hidden_size * 4  # Default fallback

        # Detect attention parameters from attention weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")

        # Default head_dim for Mistral models
        head_dim = 128

        if q_proj_weight is not None and k_proj_weight is not None:
            q_shape = q_proj_weight.shape if hasattr(q_proj_weight, "shape") else q_proj_weight.tensor_shape
            k_shape = k_proj_weight.shape if hasattr(k_proj_weight, "shape") else k_proj_weight.tensor_shape
            q_out_features = q_shape[0]
            k_out_features = k_shape[0]

            # Estimate head_dim (commonly 128 for Mistral models)
            for hd in [128, 64, 96, 256]:
                if q_out_features % hd == 0 and k_out_features % hd == 0:
                    head_dim = hd
                    num_attention_heads = q_out_features // head_dim
                    num_key_value_heads = k_out_features // head_dim
                    break
            else:
                head_dim = max(hidden_size // 32, 64)
                num_attention_heads = q_out_features // head_dim
                num_key_value_heads = k_out_features // head_dim
        else:
            num_attention_heads = 32
            num_key_value_heads = 8

        logger.info(
            f"Mistral GGUF config: hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
            f"layers={layer_count}, heads={num_attention_heads}, kv_heads={num_key_value_heads}, "
            f"head_dim={head_dim}"
        )

        # Create Mistral config
        mistral_config = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            torch_dtype=compute_dtype,
        )

        # Create model with empty weights
        with accelerate.init_empty_weights():
            model = MistralForCausalLM(mistral_config)

        # Load the GGUF weights
        model.load_state_dict(sd, strict=False, assign=True)

        # Dequantize embed_tokens weight for embedding lookups
        from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

        embed_tokens_weight = model.model.embed_tokens.weight
        if isinstance(embed_tokens_weight, GGMLTensor):
            dequantized = embed_tokens_weight.get_dequantized_tensor()
            model.model.embed_tokens.weight = torch.nn.Parameter(dequantized, requires_grad=False)
            logger.info("Dequantized embed_tokens weight for embedding lookups")

        # Handle tied weights
        if mistral_config.tie_word_embeddings:
            if model.lm_head.weight.is_meta:
                model.lm_head.weight = model.model.embed_tokens.weight
                logger.info("Tied lm_head.weight to embed_tokens.weight")
            else:
                model.tie_weights()

        return model

    def _convert_llamacpp_to_pytorch(self, sd: dict[str, Any]) -> dict[str, Any]:
        """Convert llama.cpp GGUF keys to PyTorch/HuggingFace format for Mistral models."""
        import re

        key_map = {
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
        }

        new_sd: dict[str, Any] = {}
        blk_pattern = re.compile(r"^blk\.(\d+)\.(.+)$")

        for key, value in sd.items():
            if not isinstance(key, str):
                new_sd[key] = value
                continue

            match = blk_pattern.match(key)
            if match:
                layer_idx = match.group(1)
                rest = match.group(2)

                parts = rest.split(".", 1)
                component = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""

                if component in key_map:
                    new_component = key_map[component]
                    new_key = f"model.layers.{layer_idx}.{new_component}"
                    if suffix:
                        new_key += f".{suffix}"
                    new_sd[new_key] = value
                else:
                    new_sd[f"model.layers.{layer_idx}.{rest}"] = value
                continue

            # Handle non-block keys
            if key == "token_embd.weight":
                new_sd["model.embed_tokens.weight"] = value
            elif key == "output_norm.weight":
                new_sd["model.norm.weight"] = value
            elif key == "output.weight":
                new_sd["lm_head.weight"] = value
            else:
                new_sd[key] = value

        return new_sd
