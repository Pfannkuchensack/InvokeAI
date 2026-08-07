from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    Ideogram4TransformerField,
    ModelIdentifierField,
    Qwen3EncoderField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("ideogram4_model_loader_output")
class Ideogram4ModelLoaderOutput(BaseInvocationOutput):
    """Ideogram 4 model loader output."""

    transformer: Ideogram4TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(
        description=FieldDescriptions.qwen3_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "ideogram4_model_loader",
    title="Main Model - Ideogram 4",
    tags=["model", "ideogram4"],
    category="model",
    version="2.0.0",
    classification=Classification.Prototype,
)
class Ideogram4ModelLoaderInvocation(BaseInvocation):
    """Loads an Ideogram 4 model, outputting its submodels.

    Components can be mixed and matched, mirroring the Z-Image and FLUX.2 Klein loaders:

    - Transformer: a Diffusers Ideogram 4 model carries both CFG branches in one folder. A GGUF
      model is a single branch, so the unconditional half must be wired to
      ``Unconditional Transformer``.
    - VAE: standalone FLUX.2 VAE (bit-identical to Ideogram's own) > the Diffusers source model.
    - Qwen3-VL encoder: standalone encoder > the Diffusers source model.

    The source model only comes into play for a GGUF transformer, which ships no encoder or VAE.
    """

    model: ModelIdentifierField = InputField(
        description="The Ideogram 4 model to load. For GGUF, this is the conditional transformer.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    unconditional_transformer_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The unconditional (negative) transformer, required when the main model is a GGUF. "
        "Ideogram 4 runs both CFG branches on every step and they are separately trained weights. "
        "Ignored for Diffusers models, which bundle both branches.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.GGUFQuantized,
        title="Unconditional Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE. Ideogram 4 uses the same VAE as FLUX.2. If not set, the VAE is "
        "loaded from the main model (when Diffusers) or from the source model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3-VL encoder. If not set, the encoder is loaded from the main "
        "model (when Diffusers) or from the source model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3VLEncoder,
        title="Qwen3-VL Encoder",
    )

    source_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Ideogram 4 model to take the Qwen3-VL encoder and/or VAE from. "
        "Use this when the transformer is a GGUF and you have no standalone encoder/VAE. "
        "Ignored if both are provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Source (Diffusers)",
    )

    def invoke(self, context: InvocationContext) -> Ideogram4ModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        main_is_diffusers = main_config.format is ModelFormat.Diffusers

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        unconditional_transformer = self._resolve_unconditional_transformer(context, main_is_diffusers)

        # A Diffusers main model already provides the encoder and VAE, so it doubles as its own source.
        component_source = self.source_model if not main_is_diffusers else (self.source_model or self.model)

        vae = self._resolve_component(
            context,
            explicit=self.vae_model,
            source=component_source,
            submodel_type=SubModelType.VAE,
            what="VAE",
            standalone_hint="a FLUX.2 VAE",
        )
        text_encoder = self._resolve_component(
            context,
            explicit=self.qwen3_encoder_model,
            source=component_source,
            submodel_type=SubModelType.TextEncoder,
            what="Qwen3-VL encoder",
            standalone_hint="a standalone Qwen3-VL encoder",
        )
        tokenizer = self._resolve_component(
            context,
            explicit=self.qwen3_encoder_model,
            source=component_source,
            submodel_type=SubModelType.Tokenizer,
            what="Qwen3-VL tokenizer",
            standalone_hint="a standalone Qwen3-VL encoder",
        )

        return Ideogram4ModelLoaderOutput(
            transformer=Ideogram4TransformerField(
                transformer=transformer,
                unconditional_transformer=unconditional_transformer,
                loras=[],
            ),
            qwen3_encoder=Qwen3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )

    def _resolve_unconditional_transformer(
        self, context: InvocationContext, main_is_diffusers: bool
    ) -> Optional[ModelIdentifierField]:
        """Pick the unconditional branch, rejecting combinations that would silently disable CFG.

        The conditional/unconditional distinction is recorded on the config at install time (from
        the filename — the GGUFs carry no metadata and are otherwise identical). The UI filters the
        pickers on it, but a hand-built graph or an API call can still pair two conditional halves,
        which does not crash: it just produces an image with no effective guidance. So verify here.
        """
        if main_is_diffusers:
            # Both branches come from the one folder; a second transformer would be ignored, so say so.
            if self.unconditional_transformer_model is not None:
                raise ValueError(
                    "'Unconditional Transformer' must be empty for a Diffusers Ideogram 4 model — it already "
                    "bundles both CFG branches. Clear the field, or set the main model to a GGUF transformer."
                )
            return None

        if self.unconditional_transformer_model is None:
            raise ValueError(
                "A GGUF Ideogram 4 transformer is only one CFG branch. Set 'Unconditional Transformer' to the "
                "matching unconditional GGUF (upstream names it '...-unconditional_transformer-*.gguf')."
            )

        conditional_branch = getattr(context.models.get_config(self.model), "branch", None)
        unconditional_branch = getattr(context.models.get_config(self.unconditional_transformer_model), "branch", None)
        if conditional_branch == unconditional_branch:
            raise ValueError(
                f"Both selected transformers are the '{conditional_branch}' CFG branch. Ideogram 4 needs one "
                "conditional and one unconditional transformer; using the same branch twice yields an image "
                "with no effective guidance. Check that the files are named as upstream ships them."
            )

        return self.unconditional_transformer_model.model_copy(update={"submodel_type": SubModelType.Transformer})

    def _resolve_component(
        self,
        context: InvocationContext,
        *,
        explicit: Optional[ModelIdentifierField],
        source: Optional[ModelIdentifierField],
        submodel_type: SubModelType,
        what: str,
        standalone_hint: str,
    ) -> ModelIdentifierField:
        """Standalone model wins; otherwise take the submodel from the Diffusers source model."""
        if explicit is not None:
            return explicit.model_copy(update={"submodel_type": submodel_type})

        if source is None:
            raise ValueError(
                f"No {what} source provided. Either set it to {standalone_hint}, or set 'Source (Diffusers)' "
                "to an installed Diffusers Ideogram 4 model to take it from."
            )

        source_config = context.models.get_config(source)
        if source_config.format is not ModelFormat.Diffusers:
            raise ValueError(
                f"The source model must be a Diffusers Ideogram 4 model to provide the {what}. "
                f"'{source_config.name}' is in {source_config.format.value} format."
            )

        return source.model_copy(update={"submodel_type": submodel_type})
