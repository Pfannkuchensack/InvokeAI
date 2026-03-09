from typing import Literal, Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import CLIPField, ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.flux.util import get_flux_max_seq_length
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("flux_model_loader_output")
class FluxModelLoaderOutput(BaseInvocationOutput):
    """Flux base model loader output"""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="The max sequence length to used for the T5 encoder. (256 for schnell transformer, 512 for dev transformer)",
        title="Max Seq Length",
    )


@invocation(
    "flux_model_loader",
    title="Main Model - FLUX",
    tags=["model", "flux"],
    category="model",
    version="1.1.0",
)
class FluxModelLoaderInvocation(BaseInvocation):
    """Loads a flux base model, outputting its submodels.

    When using a bundle checkpoint (a single safetensors file containing transformer, VAE, CLIP-L, and T5-XXL),
    the VAE, CLIP, and T5 encoder fields are optional and will be extracted from the main model automatically.
    You can override any of them with standalone models.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.Main,
    )

    t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone T5 Encoder model. "
        "If not provided, T5 encoder will be loaded from the main model if it is a bundle checkpoint.",
        input=Input.Direct,
        title="T5 Encoder",
        ui_model_type=ModelType.T5Encoder,
    )

    clip_embed_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone CLIP Embed model. "
        "If not provided, CLIP will be loaded from the main model if it is a bundle checkpoint.",
        input=Input.Direct,
        title="CLIP Embed",
        ui_model_type=ModelType.CLIPEmbed,
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. "
        "If not provided, VAE will be loaded from the main model if it is a bundle checkpoint.",
        input=Input.Direct,
        title="VAE",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> FluxModelLoaderOutput:
        # Validate that the main model exists
        if not context.models.exists(self.model.key):
            raise ValueError(f"Unknown model: {self.model.key}")

        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Check if main model is a bundle checkpoint
        main_config = context.models.get_config(self.model)
        main_is_bundle = isinstance(main_config, Main_Checkpoint_FLUX_Config) and main_config.is_bundle

        # Resolve VAE
        vae = self._resolve_vae(main_is_bundle)

        # Resolve CLIP
        tokenizer, clip_encoder = self._resolve_clip(main_is_bundle)

        # Resolve T5
        tokenizer2, t5_encoder = self._resolve_t5(main_is_bundle)

        assert isinstance(main_config, Checkpoint_Config_Base)

        return FluxModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            clip=CLIPField(tokenizer=tokenizer, text_encoder=clip_encoder, loras=[], skipped_layers=0),
            t5_encoder=T5EncoderField(tokenizer=tokenizer2, text_encoder=t5_encoder, loras=[]),
            vae=VAEField(vae=vae),
            max_seq_len=get_flux_max_seq_length(main_config.variant),
        )

    def _resolve_vae(self, main_is_bundle: bool) -> ModelIdentifierField:
        if self.vae_model is not None:
            return self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_bundle:
            return self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Standalone checkpoint models require a separate VAE. "
                "Options:\n"
                "  1. Set 'VAE' to a standalone FLUX VAE model\n"
                "  2. Use a bundle checkpoint that includes VAE, CLIP, and T5"
            )

    def _resolve_clip(self, main_is_bundle: bool) -> tuple[ModelIdentifierField, ModelIdentifierField]:
        if self.clip_embed_model is not None:
            tokenizer = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            clip_encoder = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
            return tokenizer, clip_encoder
        elif main_is_bundle:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            clip_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
            return tokenizer, clip_encoder
        else:
            raise ValueError(
                "No CLIP source provided. Standalone checkpoint models require a separate CLIP Embed model. "
                "Options:\n"
                "  1. Set 'CLIP Embed' to a standalone CLIP-L text encoder model\n"
                "  2. Use a bundle checkpoint that includes VAE, CLIP, and T5"
            )

    def _resolve_t5(self, main_is_bundle: bool) -> tuple[ModelIdentifierField, ModelIdentifierField]:
        if self.t5_encoder_model is not None:
            tokenizer2 = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
            t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)
            return tokenizer2, t5_encoder
        elif main_is_bundle:
            tokenizer2 = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer2})
            t5_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder2})
            return tokenizer2, t5_encoder
        else:
            raise ValueError(
                "No T5 Encoder source provided. Standalone checkpoint models require a separate T5 Encoder. "
                "Options:\n"
                "  1. Set 'T5 Encoder' to a standalone T5 text encoder model\n"
                "  2. Use a bundle checkpoint that includes VAE, CLIP, and T5"
            )
