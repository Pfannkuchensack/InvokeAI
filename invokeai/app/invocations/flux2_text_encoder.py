"""FLUX.2 Text Encoder Invocation.

Encodes text prompts using Mistral Small 3.1 for FLUX.2 models.
"""

from typing import Optional

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    Flux2ConditioningField,
    Input,
    InputField,
    OutputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import MistralEncoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.text_conditioning import format_flux2_prompt, FLUX2_SYSTEM_MESSAGE
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUX2ConditioningInfo


@invocation_output("flux2_conditioning_output")
class Flux2ConditioningOutput(BaseInvocationOutput):
    """FLUX.2 conditioning output."""

    conditioning: Flux2ConditioningField = OutputField(
        description="FLUX.2 conditioning tensor",
        title="Conditioning",
    )


@invocation(
    "flux2_text_encoder",
    title="Prompt - FLUX.2",
    tags=["prompt", "conditioning", "flux2"],
    category="conditioning",
    version="1.0.0",
)
class Flux2TextEncoderInvocation(BaseInvocation):
    """Encodes text prompts using Mistral Small 3.1 for FLUX.2.

    FLUX.2 uses a single Mistral text encoder instead of the dual
    CLIP + T5 encoders used in FLUX.1.
    """

    mistral_encoder: MistralEncoderField = InputField(
        title="Mistral Encoder",
        description="Mistral Small 3.1 encoder for FLUX.2",
        input=Input.Connection,
    )
    max_seq_len: int = InputField(
        default=512,
        description="Max sequence length for the Mistral encoder",
    )
    prompt: str = InputField(
        description="Text prompt to encode",
        ui_component=UIComponent.Textarea,
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Flux2ConditioningOutput:
        """Encode the prompt using Mistral encoder."""
        mistral_embeddings = self._mistral_encode(context)

        conditioning_data = ConditioningFieldData(
            conditionings=[FLUX2ConditioningInfo(mistral_embeds=mistral_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return Flux2ConditioningOutput(
            conditioning=Flux2ConditioningField(
                conditioning_name=conditioning_name,
                mask=self.mask,
            )
        )

    def _mistral_encode(self, context: InvocationContext) -> torch.Tensor:
        """Encode prompt using Mistral Small 3.1."""
        prompt = [self.prompt]

        # Load encoder and tokenizer
        encoder_info = context.models.load(self.mistral_encoder.text_encoder)

        with (
            encoder_info.model_on_device() as (_, mistral_encoder),
            context.models.load(self.mistral_encoder.tokenizer) as tokenizer,
        ):
            # Format prompt for Mistral chat template
            formatted_messages = format_flux2_prompt(
                prompt=self.prompt,
                system_message=FLUX2_SYSTEM_MESSAGE,
            )

            # Apply chat template and tokenize
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )

            # Move to device
            device = next(mistral_encoder.parameters()).device
            dtype = next(mistral_encoder.parameters()).dtype
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Get embeddings from Mistral
            # Use the language model's hidden states
            outputs = mistral_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Use intermediate hidden states (known to be more beneficial for conditioning)
            # Stack and average selected layers
            hidden_states = outputs.hidden_states
            # Use layers from the middle and end for best conditioning signal
            selected_layers = [
                hidden_states[len(hidden_states) // 4],
                hidden_states[len(hidden_states) // 2],
                hidden_states[3 * len(hidden_states) // 4],
                hidden_states[-1],
            ]
            embeddings = torch.stack(selected_layers, dim=0).mean(dim=0)

            # Ensure correct dtype
            embeddings = embeddings.to(dtype=dtype)

        return embeddings
