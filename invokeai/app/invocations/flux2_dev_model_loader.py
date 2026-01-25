"""Flux2 Dev Model Loader Invocation.

Loads a Flux2 Dev model with its Mistral text encoder and VAE.
Unlike FLUX.2 Klein which uses Qwen3, Dev uses Mistral Small 3.1.
"""

from typing import Literal, Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    MistralEncoderField,
    ModelIdentifierField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@invocation_output("flux2_dev_model_loader_output")
class Flux2DevModelLoaderOutput(BaseInvocationOutput):
    """Flux2 Dev model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    mistral_encoder: MistralEncoderField = OutputField(
        description="Mistral Small 3.1 text encoder for FLUX.2 Dev.", title="Mistral Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="The max sequence length for the Mistral encoder.",
        title="Max Seq Length",
    )


@invocation(
    "flux2_dev_model_loader",
    title="Main Model - Flux2 Dev",
    tags=["model", "flux", "flux2", "dev", "mistral"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevModelLoaderInvocation(BaseInvocation):
    """Loads a Flux2 Dev model, outputting its submodels.

    Flux2 Dev uses Mistral Small 3.1 as the text encoder.
    It uses a 32-channel VAE (AutoencoderKLFlux2) like Klein.

    When using a Diffusers format model, both VAE and Mistral encoder are extracted
    automatically from the main model. You can override with standalone models:
    - Transformer: Always from Flux2 Dev main model
    - VAE: From main model (Diffusers) or standalone VAE
    - Mistral Encoder: From main model (Diffusers) or standalone Mistral model
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. Flux2 Dev uses a 32-channel VAE. "
        "If not provided, VAE will be loaded from the Mistral Source model.",
        input=Input.Direct,
        ui_model_base=[BaseModelType.Flux, BaseModelType.Flux2],
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    mistral_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Mistral Encoder model. "
        "If not provided, encoder will be loaded from the Mistral Source model.",
        input=Input.Direct,
        ui_model_type=ModelType.MistralEncoder,
        title="Mistral Encoder",
    )

    mistral_source_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Flux2 Dev model to extract VAE and/or Mistral encoder from. "
        "Use this if you don't have separate VAE/Mistral models. "
        "Ignored if both VAE and Mistral Encoder are provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Mistral Source (Diffusers)",
    )

    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Mistral encoder.",
        title="Max Seq Length",
    )

    def invoke(self, context: InvocationContext) -> Flux2DevModelLoaderOutput:
        # Validate this is a Dev variant
        main_config = context.models.get_config(self.model)
        if hasattr(main_config, "variant") and main_config.variant != Flux2VariantType.Dev:
            raise ValueError(
                f"This loader is for FLUX.2 Dev models only. "
                f"The selected model is a {main_config.variant.value} variant. "
                f"Please use the Flux2 Klein Model Loader for Klein models."
            )

        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Check if main model is Diffusers format (can extract VAE directly)
        main_is_diffusers = main_config.format == ModelFormat.Diffusers

        # Determine VAE source
        if self.vae_model is not None:
            # Use standalone VAE
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            # Extract VAE from main model
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.mistral_source_model is not None:
            # Extract from Mistral source Diffusers model
            self._validate_diffusers_format(context, self.mistral_source_model, "Mistral Source")
            vae = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Standalone safetensors/GGUF models require a separate VAE. "
                "Options:\n"
                "  1. Set 'VAE' to a standalone FLUX VAE model\n"
                "  2. Set 'Mistral Source' to a Diffusers Flux2 Dev model to extract the VAE from"
            )

        # Determine Mistral Encoder source
        if self.mistral_encoder_model is not None:
            # Use standalone Mistral Encoder
            mistral_tokenizer = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            mistral_encoder = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif main_is_diffusers:
            # Extract from main model
            mistral_tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            mistral_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.mistral_source_model is not None:
            # Extract from separate Diffusers model
            self._validate_diffusers_format(context, self.mistral_source_model, "Mistral Source")
            mistral_tokenizer = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            mistral_encoder = self.mistral_source_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Mistral Encoder source provided. Standalone safetensors/GGUF models require a separate text encoder. "
                "Options:\n"
                "  1. Set 'Mistral Encoder' to a standalone Mistral text encoder model\n"
                "  2. Set 'Mistral Source' to a Diffusers Flux2 Dev model to extract the encoder from"
            )

        return Flux2DevModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            mistral_encoder=MistralEncoderField(tokenizer=mistral_tokenizer, text_encoder=mistral_encoder),
            vae=VAEField(vae=vae),
            max_seq_len=self.max_seq_len,
        )

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> None:
        """Validate that a model is in Diffusers format."""
        config = context.models.get_config(model)
        if config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The {model_name} model must be a Diffusers format model. "
                f"The selected model '{config.name}' is in {config.format.value} format."
            )
