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

        # Determine model configuration from state dict
        # FLUX.2 has 8 double blocks and 48 single blocks
        with accelerate.init_empty_weights():
            model = Flux2Transformer2DModel(
                in_channels=32,  # 32 latent channels for FLUX.2
                num_layers=8,  # 8 double-stream blocks
                num_single_layers=48,  # 48 single-stream blocks
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=2048,  # Mistral embedding dim
                pooled_projection_dim=2048,
                guidance_embeds=True,
            )

        # Convert to bfloat16 for inference
        new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

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

        with accelerate.init_empty_weights():
            model = Flux2Transformer2DModel(
                in_channels=32,
                num_layers=8,
                num_single_layers=48,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=2048,
                pooled_projection_dim=2048,
                guidance_embeds=True,
            )

        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)
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

        with SilenceWarnings():
            with accelerate.init_empty_weights():
                model = Flux2Transformer2DModel(
                    in_channels=32,
                    num_layers=8,
                    num_single_layers=48,
                    attention_head_dim=128,
                    num_attention_heads=24,
                    joint_attention_dim=2048,
                    pooled_projection_dim=2048,
                    guidance_embeds=True,
                )
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=torch.bfloat16)

            sd = load_file(model_path)
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
