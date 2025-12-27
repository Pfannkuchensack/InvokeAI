"""FLUX.2 Model Loader Invocation.

Loads a FLUX.2 model and its submodels (transformer, Mistral encoder, VAE).
"""

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import MistralEncoderField, ModelIdentifierField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("flux2_model_loader_output")
class Flux2ModelLoaderOutput(BaseInvocationOutput):
    """FLUX.2 base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    mistral_encoder: MistralEncoderField = OutputField(
        description="Mistral Small 3.1 text encoder for FLUX.2", title="Mistral Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: int = OutputField(
        description="The max sequence length for the Mistral encoder (default: 512)",
        title="Max Seq Length",
    )


@invocation(
    "flux2_model_loader",
    title="Main Model - FLUX.2",
    tags=["model", "flux2"],
    category="model",
    version="1.0.0",
)
class Flux2ModelLoaderInvocation(BaseInvocation):
    """Loads a FLUX.2 base model, outputting its submodels.

    FLUX.2 uses a single Mistral Small 3.1 text encoder instead of
    the dual CLIP + T5 encoders used in FLUX.1, and a 32-channel VAE.
    """

    model: ModelIdentifierField = InputField(
        description="FLUX.2 main model to load",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
    )

    mistral_encoder_model: ModelIdentifierField = InputField(
        description="Mistral Small 3.1 text encoder model",
        input=Input.Direct,
        title="Mistral Encoder",
        ui_model_type=ModelType.MistralEncoder,
    )

    vae_model: ModelIdentifierField = InputField(
        description="FLUX.2 VAE model (32-channel)",
        title="VAE",
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> Flux2ModelLoaderOutput:
        # Validate models exist
        for key in [self.model.key, self.mistral_encoder_model.key, self.vae_model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        # Create submodel references
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Mistral encoder uses tokenizer (actually AutoProcessor) and text_encoder
        tokenizer = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.mistral_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return Flux2ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            mistral_encoder=MistralEncoderField(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                loras=[],
            ),
            vae=VAEField(vae=vae),
            max_seq_len=512,  # FLUX.2 default max sequence length
        )
