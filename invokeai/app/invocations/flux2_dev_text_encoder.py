"""Flux2 Dev Text Encoder Invocation.

Flux2 Dev uses Mistral Small 3.1 as the text encoder.
The key difference from Klein is that it extracts hidden states from layers (10, 20, 30)
instead of (9, 18, 27) and uses a different prompt format with a system message.
"""

from contextlib import ExitStack
from typing import Iterator, Literal, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import MistralEncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.dev_text_conditioning import DEV_EXTRACTION_LAYERS, format_flux2_dev_prompt
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# Default max sequence length for Dev models
DEV_MAX_SEQ_LEN = 512


@invocation(
    "flux2_dev_text_encoder",
    title="Prompt - Flux2 Dev",
    tags=["prompt", "conditioning", "flux", "flux2", "dev", "mistral"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DevTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for Flux2 Dev image generation.

    Flux2 Dev uses Mistral Small 3.1 as the text encoder, extracting hidden states from
    layers (10, 20, 30) and stacking them for richer text representations.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    mistral_encoder: MistralEncoderField = InputField(
        title="Mistral Encoder",
        description="Mistral Small 3.1 text encoder for FLUX.2 Dev.",
        input=Input.Connection,
    )
    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Mistral encoder.",
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        mistral_embeds, pooled_embeds = self._encode_prompt(context)

        # Use FLUXConditioningInfo for compatibility with existing Flux denoiser
        # t5_embeds -> mistral stacked embeddings
        # clip_embeds -> pooled mistral embedding
        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=pooled_embeds, t5_embeds=mistral_embeds)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _encode_prompt(self, context: InvocationContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using Mistral text encoder with Dev-style layer extraction.

        Returns:
            Tuple of (stacked_embeddings, pooled_embedding):
            - stacked_embeddings: Hidden states from layers (10, 20, 30) stacked together.
              Shape: (1, seq_len, hidden_size * 3)
            - pooled_embedding: Pooled representation for global conditioning.
              Shape: (1, hidden_size)
        """
        prompt = self.prompt
        device = TorchDevice.choose_torch_device()

        text_encoder_info = context.models.load(self.mistral_encoder.text_encoder)
        tokenizer_info = context.models.load(self.mistral_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            # Apply LoRA models to the text encoder
            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,  # Reuse T5 prefix for Mistral LoRAs
                    dtype=lora_dtype,
                    cached_weights=cached_weights,
                )
            )

            context.util.signal_progress("Running Mistral text encoder (Dev)")

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(
                    f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}. "
                    "The Mistral encoder model may be corrupted or incompatible."
                )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}. "
                    "The Mistral tokenizer may be corrupted or incompatible."
                )

            # Format messages with system prompt for Mistral
            messages = format_flux2_dev_prompt(prompt)

            # Apply chat template to get formatted text
            text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize the formatted text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass through the model
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # Validate hidden_states output
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    "Text encoder did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported by this model."
                )

            num_hidden_layers = len(outputs.hidden_states)

            # Extract and stack hidden states from Dev-specific layers
            hidden_states_list = []
            for layer_idx in DEV_EXTRACTION_LAYERS:
                if layer_idx >= num_hidden_layers:
                    layer_idx = num_hidden_layers - 1
                hidden_states_list.append(outputs.hidden_states[layer_idx])

            # Stack along dim=1, then permute and reshape
            out = torch.stack(hidden_states_list, dim=1)
            out = out.to(dtype=text_encoder.dtype, device=device)

            batch_size, num_channels, seq_len, hidden_dim = out.shape
            prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

            # Create pooled embedding for global conditioning
            # Use mean pooling over the sequence (excluding padding)
            last_hidden_state = outputs.hidden_states[-1]
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeds = (last_hidden_state * expanded_mask).sum(dim=1)
            num_tokens = expanded_mask.sum(dim=1).clamp(min=1)
            pooled_embeds = sum_embeds / num_tokens

        return prompt_embeds, pooled_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the Mistral text encoder."""
        for lora in self.mistral_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
