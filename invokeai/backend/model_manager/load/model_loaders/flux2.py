# Copyright (c) 2024, InvokeAI Development Team
"""Class for FLUX.2 model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_BnBNF4_FLUX2_Config,
    Main_Checkpoint_FLUX2_Config,
    Main_Diffusers_FLUX2_Config,
    Main_GGUF_FLUX2_Config,
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
from invokeai.backend.util.silence_warnings import SilenceWarnings

try:
    from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

    bnb_available = True
except ImportError:
    bnb_available = False

app_config = get_config()


def _maybe_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a GGML tensor if needed, otherwise return as-is."""
    if hasattr(tensor, "get_dequantized_tensor"):
        return tensor.get_dequantized_tensor()
    return tensor


def _fix_context_embedder_for_checkpoint(model: "torch.nn.Module", sd: dict[str, torch.Tensor]) -> None:
    """Adjust context_embedder layer to match the checkpoint weight shape.

    Different checkpoints have different formats:
    - BFL format: txt_in.weight is (15360, 6144) -> Linear(6144, 15360)
    - Diffusers format: context_embedder.weight is (6144, 15360) -> Linear(15360, 6144)

    We check the checkpoint weight shape and create the appropriate layer.
    """
    import torch.nn as nn

    key = "context_embedder.weight"
    if key not in sd or not hasattr(model, "context_embedder"):
        return

    weight = sd[key]
    if hasattr(weight, "get_dequantized_tensor"):
        weight = weight.get_dequantized_tensor()

    weight_shape = weight.shape
    # PyTorch Linear weight is (out_features, in_features)
    out_features, in_features = weight_shape

    # Determine if we need bias
    has_bias = "context_embedder.bias" in sd

    # Replace with correctly-sized layer
    model.context_embedder = nn.Linear(in_features, out_features, bias=has_bias)


def convert_bfl_flux2_to_diffusers(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert BFL FLUX.2 checkpoint keys to diffusers format.

    BFL uses a different naming convention than diffusers for FLUX.2 transformers.
    This function handles the key renaming and QKV weight splitting.

    Note: For GGUF quantized models, tensors are dequantized during conversion
    because the key renaming breaks the GGML dispatch mechanism.

    Args:
        sd: State dict with BFL key naming.

    Returns:
        State dict with diffusers key naming.
    """
    # Check if already in diffusers format (skip conversion)
    # Diffusers format has keys like "transformer_blocks.0.attn.to_q.weight"
    # BFL format has keys like "double_blocks.0.img_attn.qkv.weight"
    diffusers_keys = [k for k in sd.keys() if k.startswith("transformer_blocks.") or k.startswith("single_transformer_blocks.")]
    bfl_keys = [k for k in sd.keys() if k.startswith("double_blocks.") or k.startswith("single_blocks.")]

    if diffusers_keys and not bfl_keys:
        # Already in diffusers format
        return sd

    if not bfl_keys:
        # Unknown format, return as-is
        return sd

    new_sd: dict[str, torch.Tensor] = {}

    # Simple key renames (excluding context_embedder which needs special handling)
    rename_map = {
        "img_in.weight": "x_embedder.weight",
        "img_in.bias": "x_embedder.bias",
        # NOTE: txt_in (context_embedder) is handled separately below - it needs transposing
        # BFL txt_in: Linear(6144 -> 15360), but diffusers context_embedder: Linear(15360 -> 6144)
        "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.bias": "time_guidance_embed.timestep_embedder.linear_1.bias",
        "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
        "time_in.out_layer.bias": "time_guidance_embed.timestep_embedder.linear_2.bias",
        "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
        "guidance_in.in_layer.bias": "time_guidance_embed.guidance_embedder.linear_1.bias",
        "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
        "guidance_in.out_layer.bias": "time_guidance_embed.guidance_embedder.linear_2.bias",
        "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_img.lin.bias": "double_stream_modulation_img.linear.bias",
        "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
        "double_stream_modulation_txt.lin.bias": "double_stream_modulation_txt.linear.bias",
        "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
        "single_stream_modulation.lin.bias": "single_stream_modulation.linear.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
    }

    for old_key, new_key in rename_map.items():
        if old_key in sd:
            new_sd[new_key] = _maybe_dequantize(sd[old_key])

    # Handle context_embedder (txt_in) - keep weight as-is, we replace the layer architecture
    # BFL txt_in: Linear(text_embed_dim=6144 -> joint_attention_dim=15360), weight (15360, 6144)
    # We replace diffusers' context_embedder with BFL's architecture, so no transpose needed
    if "txt_in.weight" in sd:
        new_sd["context_embedder.weight"] = _maybe_dequantize(sd["txt_in.weight"])
    if "txt_in.bias" in sd:
        new_sd["context_embedder.bias"] = _maybe_dequantize(sd["txt_in.bias"])

    # Process double blocks (transformer_blocks)
    for key, value in sd.items():
        if key.startswith("double_blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])

            new_prefix = f"transformer_blocks.{block_idx}"

            # Handle img_attn (image attention)
            if rest == "img_attn.qkv.weight":
                # Split fused QKV into separate Q, K, V
                # Shape is [3 * num_heads * head_dim, hidden_size]
                # For GGML tensors, we need to dequantize first
                tensor_to_split = value
                if hasattr(value, "get_dequantized_tensor"):
                    tensor_to_split = value.get_dequantized_tensor()
                q, k, v = tensor_to_split.chunk(3, dim=0)
                new_sd[f"{new_prefix}.attn.to_q.weight"] = q
                new_sd[f"{new_prefix}.attn.to_k.weight"] = k
                new_sd[f"{new_prefix}.attn.to_v.weight"] = v
            elif rest == "img_attn.qkv.bias":
                tensor_to_split = value
                if hasattr(value, "get_dequantized_tensor"):
                    tensor_to_split = value.get_dequantized_tensor()
                q, k, v = tensor_to_split.chunk(3, dim=0)
                new_sd[f"{new_prefix}.attn.to_q.bias"] = q
                new_sd[f"{new_prefix}.attn.to_k.bias"] = k
                new_sd[f"{new_prefix}.attn.to_v.bias"] = v
            elif rest == "img_attn.norm.query_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_q.weight"] = _maybe_dequantize(value)
            elif rest == "img_attn.norm.key_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_k.weight"] = _maybe_dequantize(value)
            elif rest == "img_attn.proj.weight":
                new_sd[f"{new_prefix}.attn.to_out.0.weight"] = _maybe_dequantize(value)
            elif rest == "img_attn.proj.bias":
                new_sd[f"{new_prefix}.attn.to_out.0.bias"] = _maybe_dequantize(value)

            # Handle txt_attn (text/context attention)
            elif rest == "txt_attn.qkv.weight":
                tensor_to_split = value
                if hasattr(value, "get_dequantized_tensor"):
                    tensor_to_split = value.get_dequantized_tensor()
                q, k, v = tensor_to_split.chunk(3, dim=0)
                new_sd[f"{new_prefix}.attn.add_q_proj.weight"] = q
                new_sd[f"{new_prefix}.attn.add_k_proj.weight"] = k
                new_sd[f"{new_prefix}.attn.add_v_proj.weight"] = v
            elif rest == "txt_attn.qkv.bias":
                tensor_to_split = value
                if hasattr(value, "get_dequantized_tensor"):
                    tensor_to_split = value.get_dequantized_tensor()
                q, k, v = tensor_to_split.chunk(3, dim=0)
                new_sd[f"{new_prefix}.attn.add_q_proj.bias"] = q
                new_sd[f"{new_prefix}.attn.add_k_proj.bias"] = k
                new_sd[f"{new_prefix}.attn.add_v_proj.bias"] = v
            elif rest == "txt_attn.norm.query_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_added_q.weight"] = _maybe_dequantize(value)
            elif rest == "txt_attn.norm.key_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_added_k.weight"] = _maybe_dequantize(value)
            elif rest == "txt_attn.proj.weight":
                new_sd[f"{new_prefix}.attn.to_add_out.weight"] = _maybe_dequantize(value)
            elif rest == "txt_attn.proj.bias":
                new_sd[f"{new_prefix}.attn.to_add_out.bias"] = _maybe_dequantize(value)

            # Handle MLPs
            elif rest == "img_mlp.0.weight":
                new_sd[f"{new_prefix}.ff.linear_in.weight"] = _maybe_dequantize(value)
            elif rest == "img_mlp.0.bias":
                new_sd[f"{new_prefix}.ff.linear_in.bias"] = _maybe_dequantize(value)
            elif rest == "img_mlp.2.weight":
                new_sd[f"{new_prefix}.ff.linear_out.weight"] = _maybe_dequantize(value)
            elif rest == "img_mlp.2.bias":
                new_sd[f"{new_prefix}.ff.linear_out.bias"] = _maybe_dequantize(value)
            elif rest == "txt_mlp.0.weight":
                new_sd[f"{new_prefix}.ff_context.linear_in.weight"] = _maybe_dequantize(value)
            elif rest == "txt_mlp.0.bias":
                new_sd[f"{new_prefix}.ff_context.linear_in.bias"] = _maybe_dequantize(value)
            elif rest == "txt_mlp.2.weight":
                new_sd[f"{new_prefix}.ff_context.linear_out.weight"] = _maybe_dequantize(value)
            elif rest == "txt_mlp.2.bias":
                new_sd[f"{new_prefix}.ff_context.linear_out.bias"] = _maybe_dequantize(value)

        # Process single blocks (single_transformer_blocks)
        elif key.startswith("single_blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])

            new_prefix = f"single_transformer_blocks.{block_idx}"

            if rest == "linear1.weight":
                new_sd[f"{new_prefix}.attn.to_qkv_mlp_proj.weight"] = _maybe_dequantize(value)
            elif rest == "linear1.bias":
                new_sd[f"{new_prefix}.attn.to_qkv_mlp_proj.bias"] = _maybe_dequantize(value)
            elif rest == "linear2.weight":
                new_sd[f"{new_prefix}.attn.to_out.weight"] = _maybe_dequantize(value)
            elif rest == "linear2.bias":
                new_sd[f"{new_prefix}.attn.to_out.bias"] = _maybe_dequantize(value)
            elif rest == "norm.query_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_q.weight"] = _maybe_dequantize(value)
            elif rest == "norm.key_norm.scale":
                new_sd[f"{new_prefix}.attn.norm_k.weight"] = _maybe_dequantize(value)

    return new_sd


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Diffusers)
class Flux2DiffusersModel(ModelLoader):
    """Class to load FLUX.2 models from diffusers format."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_FLUX2_Config):
            raise ValueError("Only Main_Diffusers_FLUX2_Config models are supported here.")

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_transformer(model_path)
            case SubModelType.VAE:
                return self._load_vae(model_path)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(model_path)
            case SubModelType.Tokenizer:
                return self._load_tokenizer(model_path)

        raise ValueError(
            f"Unsupported submodel type: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_transformer(self, model_path: Path) -> AnyModel:
        """Load FLUX.2 transformer from diffusers format."""
        from diffusers import Flux2Transformer2DModel

        transformer_path = model_path / "transformer"
        if not transformer_path.exists():
            transformer_path = model_path

        return Flux2Transformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def _load_vae(self, model_path: Path) -> AnyModel:
        """Load FLUX.2 VAE (32-channel) from diffusers format."""
        from diffusers import AutoencoderKLFlux2

        vae_path = model_path / "vae"
        if not vae_path.exists():
            vae_path = model_path

        return AutoencoderKLFlux2.from_pretrained(
            vae_path,
            torch_dtype=torch.bfloat16,
        )

    def _load_text_encoder(self, model_path: Path) -> AnyModel:
        """Load Mistral Small 3.1 text encoder for FLUX.2."""
        from transformers import Mistral3ForConditionalGeneration

        encoder_path = model_path / "text_encoder"
        if not encoder_path.exists():
            encoder_path = model_path

        return Mistral3ForConditionalGeneration.from_pretrained(
            encoder_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def _load_tokenizer(self, model_path: Path) -> AnyModel:
        """Load tokenizer for Mistral encoder."""
        from transformers import AutoProcessor

        tokenizer_path = model_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = model_path

        return AutoProcessor.from_pretrained(tokenizer_path)


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Flux2CheckpointModel(ModelLoader):
    """Class to load FLUX.2 models from checkpoint format."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        """Load FLUX.2 transformer from single checkpoint file."""
        assert isinstance(config, Main_Checkpoint_FLUX2_Config)

        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)
        sd = load_file(model_path)

        # Convert BFL checkpoint keys to diffusers format
        sd = convert_bfl_flux2_to_diffusers(sd)

        # FLUX.2 architecture: 8 double-stream blocks, 48 single-stream blocks
        with accelerate.init_empty_weights():
            model = Flux2Transformer2DModel(
                patch_size=1,
                in_channels=128,  # FLUX.2 uses 128-dim latent space
                num_layers=8,  # 8 double-stream blocks
                num_single_layers=48,  # 48 single-stream blocks
                attention_head_dim=128,
                num_attention_heads=48,
                joint_attention_dim=15360,  # Mistral conditioning dimension
                timestep_guidance_channels=256,
                mlp_ratio=3.0,
                axes_dims_rope=(32, 32, 32, 32),
                rope_theta=2000,
                eps=1e-6,
            )

        # Convert to bfloat16 for inference
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

        # Fix context_embedder layer to match checkpoint weight shape
        _fix_context_embedder_for_checkpoint(model, sd)

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class Flux2GGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized FLUX.2 models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        """Load GGUF-quantized FLUX.2 transformer."""
        assert isinstance(config, Main_GGUF_FLUX2_Config)

        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)

        # Load and convert state dict
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)
        sd = convert_bfl_flux2_to_diffusers(sd)

        with accelerate.init_empty_weights():
            # FLUX.2 architecture: 8 double-stream blocks, 48 single-stream blocks
            model = Flux2Transformer2DModel(
                patch_size=1,
                in_channels=128,
                num_layers=8,
                num_single_layers=48,
                attention_head_dim=128,
                num_attention_heads=48,
                joint_attention_dim=15360,
                timestep_guidance_channels=256,
                mlp_ratio=3.0,
                axes_dims_rope=(32, 32, 32, 32),
                rope_theta=2000,
                eps=1e-6,
            )

        # Fix context_embedder layer to match checkpoint weight shape
        _fix_context_embedder_for_checkpoint(model, sd)

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.BnbQuantizednf4b)
class Flux2BnbNF4CheckpointModel(ModelLoader):
    """Class to load BnB NF4 quantized FLUX.2 models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        """Load BnB NF4 quantized FLUX.2 transformer."""
        assert isinstance(config, Main_BnBNF4_FLUX2_Config)

        if not bnb_available:
            raise ImportError(
                "The bnb modules are not available. Please install bitsandbytes if available on your platform."
            )

        from diffusers import Flux2Transformer2DModel

        model_path = Path(config.path)

        # Load and convert state dict
        sd = load_file(model_path)
        sd = convert_bfl_flux2_to_diffusers(sd)

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                # FLUX.2 architecture: 8 double-stream blocks, 48 single-stream blocks
                model = Flux2Transformer2DModel(
                    patch_size=1,
                    in_channels=128,
                    num_layers=8,
                    num_single_layers=48,
                    attention_head_dim=128,
                    num_attention_heads=48,
                    joint_attention_dim=15360,
                    timestep_guidance_channels=256,
                    mlp_ratio=3.0,
                    axes_dims_rope=(32, 32, 32, 32),
                    rope_theta=2000,
                    eps=1e-6,
                )
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)

            # Fix context_embedder layer to match checkpoint weight shape
            _fix_context_embedder_for_checkpoint(model, sd)

            model.load_state_dict(sd, assign=True)

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class Flux2VAELoader(ModelLoader):
    """Class to load FLUX.2 VAE models (32-channel)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from diffusers import AutoencoderKLFlux2

        model_path = Path(config.path)
        sd = load_file(model_path)

        # Create VAE with FLUX.2 configuration (32 latent channels)
        with accelerate.init_empty_weights():
            model = AutoencoderKLFlux2(
                in_channels=3,
                out_channels=3,
                latent_channels=32,  # 32 channels for FLUX.2
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
            )

        model.load_state_dict(sd, assign=True)

        # Use bfloat16 for VAE
        if self._torch_dtype == torch.float16:
            try:
                vae_dtype = torch.tensor([1.0], dtype=torch.bfloat16, device=self._torch_device).dtype
            except TypeError:
                vae_dtype = torch.float32
        else:
            vae_dtype = self._torch_dtype

        model.to(vae_dtype)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.VAE, format=ModelFormat.Diffusers)
class Flux2VAEDiffusersLoader(ModelLoader):
    """Class to load FLUX.2 VAE from diffusers format."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from diffusers import AutoencoderKLFlux2

        model_path = Path(config.path)

        model = AutoencoderKLFlux2.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        return model


# =============================================================================
# MistralEncoder Loaders for FLUX.2
# =============================================================================


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.Diffusers)
class MistralEncoderDiffusersLoader(ModelLoader):
    """Class to load Mistral Encoder models from diffusers format.

    Uses bitsandbytes 4-bit quantization by default for the 24B parameter
    Mistral model to reduce memory usage.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.TextEncoder:
                # FLUX.2 uses Mistral3ForConditionalGeneration (multimodal model)
                # This is the same model class used by diffusers
                # Use bitsandbytes 4-bit quantization to fit in memory
                if bnb_available:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    )
                    return Mistral3ForConditionalGeneration.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
                else:
                    # Fallback without quantization (may OOM on smaller GPUs)
                    return Mistral3ForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                    )
            case SubModelType.Tokenizer:
                # Use AutoProcessor for Mistral3 (handles chat templates properly)
                return AutoProcessor.from_pretrained(model_path)

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.Checkpoint)
class MistralEncoderCheckpointLoader(ModelLoader):
    """Class to load single-file Mistral Encoder models (safetensors format)."""

    # Default HuggingFace model to load processor from (for chat template support)
    # Use mistralai/Mistral-Small-3.1-24B-Instruct-2503 for the processor
    DEFAULT_PROCESSOR_SOURCE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import AutoProcessor

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_singlefile(model_path)
            case SubModelType.Tokenizer:
                # For single-file Mistral, load processor from HuggingFace
                # AutoProcessor has chat_template support
                return AutoProcessor.from_pretrained(self.DEFAULT_PROCESSOR_SOURCE)

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(self, model_path: Path) -> AnyModel:
        """Load Mistral encoder from single safetensors file.

        FLUX.2 uses a text-only Mistral model (not multimodal Mistral3).
        The BFL checkpoint may include FP8 quantization scales.
        We infer model config from the state dict shapes.

        Note: BFL's text encoder uses a non-standard Mistral architecture where
        head_dim != hidden_size/num_attention_heads. We must set head_dim explicitly.
        The checkpoint is encoder-only - uses MistralModel (not MistralForCausalLM).
        """
        from transformers import MistralModel, MistralConfig

        sd = load_file(model_path)

        # Filter out non-model keys (FP8 scales, tokenizer, etc.)
        filtered_sd = {}
        for k, v in sd.items():
            # Skip metadata keys
            if k in ("scaled_fp8", "tekken_model"):
                continue
            # Skip FP8 quantization scales - we load as bfloat16
            if k.endswith(".input_scale") or k.endswith(".weight_scale"):
                continue
            filtered_sd[k] = v

        # MistralModel expects keys without "model." prefix (it IS the model)
        # Convert "model.layers.X..." -> "layers.X...", "model.embed_tokens" -> "embed_tokens"
        renamed_sd = {}
        for k, v in filtered_sd.items():
            if k.startswith("model."):
                renamed_sd[k[6:]] = v  # Remove "model." prefix
            else:
                renamed_sd[k] = v

        # Infer model configuration from state dict
        # Get hidden_size from embed_tokens
        embed_weight = renamed_sd.get("embed_tokens.weight")
        vocab_size = embed_weight.shape[0] if embed_weight is not None else 131072
        hidden_size = embed_weight.shape[1] if embed_weight is not None else 5120

        # Get intermediate_size from MLP
        gate_proj = renamed_sd.get("layers.0.mlp.gate_proj.weight")
        intermediate_size = gate_proj.shape[0] if gate_proj is not None else 32768

        # Count number of layers
        layer_indices = set()
        for k in renamed_sd.keys():
            if k.startswith("layers."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
        num_hidden_layers = max(layer_indices) + 1 if layer_indices else 30

        # Get attention dimensions - BFL uses head_dim=128 which doesn't match hidden_size/num_heads
        q_proj = renamed_sd.get("layers.0.self_attn.q_proj.weight")
        k_proj = renamed_sd.get("layers.0.self_attn.k_proj.weight")

        # Infer head_dim from the ratio of q_proj output and hidden_size
        # BFL model: q_proj is [4096, 5120], so attention output is 4096
        # With 32 heads, head_dim = 4096/32 = 128
        q_proj_out_dim = q_proj.shape[0] if q_proj is not None else 4096
        k_proj_out_dim = k_proj.shape[0] if k_proj is not None else 1024

        # Try common head_dims to find num_attention_heads
        head_dim = 128  # Default
        num_attention_heads = 32
        num_key_value_heads = 8
        for candidate_head_dim in [128, 64, 96, 256]:
            if q_proj_out_dim % candidate_head_dim == 0 and k_proj_out_dim % candidate_head_dim == 0:
                head_dim = candidate_head_dim
                num_attention_heads = q_proj_out_dim // candidate_head_dim
                num_key_value_heads = k_proj_out_dim // candidate_head_dim
                break

        # BFL FLUX.2 Mistral encoder configuration (inferred from checkpoint)
        # Note: head_dim must be set explicitly as it differs from hidden_size/num_attention_heads
        mistral_config = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,  # Explicitly set - BFL uses 128 with hidden_size=5120
            hidden_act="silu",
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            sliding_window=None,
            attention_dropout=0.0,
        )

        # Convert state dict to bfloat16 first
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in renamed_sd.values()])
        self._ram_cache.make_room(new_sd_size)
        for k in renamed_sd.keys():
            renamed_sd[k] = renamed_sd[k].to(torch.bfloat16)

        # BFL checkpoint is missing norm.weight - add it to state dict
        # This is the final layer norm, initialized to ones (standard for RMSNorm)
        if "norm.weight" not in renamed_sd:
            renamed_sd["norm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)

        # Create model with empty weights to avoid doubling memory usage
        with accelerate.init_empty_weights():
            model = MistralModel(mistral_config)

        # Load state dict - now all required keys are present
        model.load_state_dict(renamed_sd, assign=True, strict=True)

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.MistralEncoder, format=ModelFormat.GGUFQuantized)
class MistralEncoderGGUFLoader(ModelLoader):
    """Class to load GGUF-quantized Mistral Encoder models."""

    # Default HuggingFace model to load processor from (for chat template support)
    DEFAULT_PROCESSOR_SOURCE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import AutoProcessor

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_gguf(model_path)
            case SubModelType.Tokenizer:
                # For GGUF Mistral, load processor from HuggingFace
                # AutoProcessor has chat_template support for proper tokenization
                return AutoProcessor.from_pretrained(self.DEFAULT_PROCESSOR_SOURCE)

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_gguf(self, model_path: Path) -> AnyModel:
        """Load GGUF-quantized Mistral encoder."""
        from transformers import MistralModel, MistralConfig

        # Load GGUF state dict first to infer config
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # Rename keys if needed (GGUF may have "model." prefix)
        renamed_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                renamed_sd[k[6:]] = v
            else:
                renamed_sd[k] = v

        # Infer configuration from state dict
        embed_weight = renamed_sd.get("embed_tokens.weight")
        vocab_size = embed_weight.shape[0] if embed_weight is not None else 131072
        hidden_size = embed_weight.shape[1] if embed_weight is not None else 5120

        gate_proj = renamed_sd.get("layers.0.mlp.gate_proj.weight")
        intermediate_size = gate_proj.shape[0] if gate_proj is not None else 32768

        # Count layers
        layer_indices = set()
        for k in renamed_sd.keys():
            if k.startswith("layers."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
        num_hidden_layers = max(layer_indices) + 1 if layer_indices else 30

        # Get attention dimensions
        q_proj = renamed_sd.get("layers.0.self_attn.q_proj.weight")
        k_proj = renamed_sd.get("layers.0.self_attn.k_proj.weight")
        q_proj_out_dim = q_proj.shape[0] if q_proj is not None else 4096
        k_proj_out_dim = k_proj.shape[0] if k_proj is not None else 1024

        head_dim = 128
        num_attention_heads = 32
        num_key_value_heads = 8
        for candidate_head_dim in [128, 64, 96, 256]:
            if q_proj_out_dim % candidate_head_dim == 0 and k_proj_out_dim % candidate_head_dim == 0:
                head_dim = candidate_head_dim
                num_attention_heads = q_proj_out_dim // candidate_head_dim
                num_key_value_heads = k_proj_out_dim // candidate_head_dim
                break

        mistral_config = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act="silu",
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            sliding_window=None,
            attention_dropout=0.0,
        )

        # Add missing norm.weight if needed
        if "norm.weight" not in renamed_sd:
            renamed_sd["norm.weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)

        # Create model with empty weights to avoid doubling memory usage
        with accelerate.init_empty_weights():
            model = MistralModel(mistral_config)

        model.load_state_dict(renamed_sd, assign=True, strict=True)
        return model
