"""FLUX.2 Text Encoder Invocation.

Encodes text prompts using Mistral Small 3.1 for FLUX.2 models.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger("InvokeAI")

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
from invokeai.backend.flux2.text_conditioning import FLUX2_SYSTEM_MESSAGE, format_flux2_prompt
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUX2ConditioningInfo

# Diffusers stacks 3 intermediate layers to create text embeddings
# Each layer outputs 5120-dim, stacking gives 3 * 5120 = 15360 dim
# This matches the transformer's joint_attention_dim (15360)
# Default layer indices (will be adjusted based on actual model layer count)
DEFAULT_HIDDEN_STATES_LAYERS = (10, 20, 30)  # Layer indices to stack (same as diffusers)


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
        """Encode prompt using Mistral Small 3.1.

        Note: We run the text encoder on CPU to save VRAM for the main transformer.
        This is slower but allows running FLUX.2 on GPUs with limited memory.
        """
        # Load encoder and tokenizer
        encoder_info = context.models.load(self.mistral_encoder.text_encoder)

        with (
            encoder_info.model_on_device() as (_, mistral_encoder),
            context.models.load(self.mistral_encoder.tokenizer) as processor,
        ):
            # processor can be AutoProcessor (wraps tokenizer) or a plain tokenizer
            # AutoProcessor has the tokenizer as processor.tokenizer, but also has apply_chat_template
            # Check for chat_template support in both processor and its wrapped tokenizer
            has_chat_template = (
                hasattr(processor, "apply_chat_template")
                and (
                    (hasattr(processor, "chat_template") and processor.chat_template is not None)
                    or (hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "chat_template") and processor.tokenizer.chat_template is not None)
                )
            )

            if has_chat_template:
                # Format prompt as chat messages (same as diffusers)
                messages = format_flux2_prompt(self.prompt, system_message=FLUX2_SYSTEM_MESSAGE)
                inputs = processor.apply_chat_template(
                    [messages],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_len,
                )
            else:
                # Fallback: simple text format with system message (for tokenizers without chat template)
                text = f"[INST] {FLUX2_SYSTEM_MESSAGE}\n\n{self.prompt} [/INST]"
                # Use the tokenizer directly if it's a processor, otherwise use as-is
                tokenizer = getattr(processor, "tokenizer", processor)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_len,
                )

            # Determine inference device based on model type
            # bitsandbytes quantized models must stay on GPU
            # Check if model has quantized layers (bnb 4-bit quantization)
            is_quantized = any(
                hasattr(module, "weight") and hasattr(module.weight, "quant_state")
                for module in mistral_encoder.modules()
            )

            if is_quantized:
                # Quantized model - run on its current device (GPU)
                inference_device = next(mistral_encoder.parameters()).device
            else:
                # Non-quantized model - can move to CPU to save VRAM
                inference_device = torch.device("cpu")
                mistral_encoder = mistral_encoder.to(inference_device)

            # Move inputs to the same device
            input_ids = inputs["input_ids"].to(inference_device)
            attention_mask = inputs["attention_mask"].to(inference_device)

            # Get embeddings from Mistral
            # MistralModel returns hidden states directly (not wrapped in .model like MistralForCausalLM)
            outputs = mistral_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Stack outputs from intermediate layers (same approach as diffusers)
            # This creates 3 * 5120 = 15360 dim embeddings matching joint_attention_dim
            hidden_states = outputs.hidden_states
            num_hidden_layers = len(hidden_states)

            # Determine which layers to use based on actual model size
            # Diffusers uses layers 10, 20, 30 for a 30-layer model
            # We scale proportionally for different model sizes
            if num_hidden_layers >= 31:
                # Use standard diffusers layers for 30+ layer models
                layer_indices = DEFAULT_HIDDEN_STATES_LAYERS
            else:
                # Scale layer indices proportionally
                # Select layers at ~1/3, ~2/3, and last position
                layer_indices = (
                    num_hidden_layers // 3,
                    2 * num_hidden_layers // 3,
                    num_hidden_layers - 1,
                )

            # Stack selected layers: (batch, seq, hidden) -> (batch, num_layers, seq, hidden)
            out = torch.stack([hidden_states[k] for k in layer_indices], dim=1)

            # Ensure correct dtype
            out = out.to(dtype=torch.bfloat16)

            # Reshape to concatenate layer outputs: (batch, num_layers, seq, hidden) -> (batch, seq, num_layers * hidden)
            batch_size, num_layers, seq_len, hidden_dim = out.shape
            embeddings = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)

            # Log embedding info for debugging
            logger.info(f"FLUX.2 text encoder: {num_hidden_layers} hidden states, using layers {layer_indices}")
            logger.info(f"FLUX.2 text embeddings shape: {embeddings.shape}")

            # Move to CPU for storage (will be moved to GPU when needed by the denoiser)
            embeddings = embeddings.to(device="cpu")

        return embeddings
