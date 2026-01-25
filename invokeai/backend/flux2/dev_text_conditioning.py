"""Text conditioning utilities for FLUX.2 Dev using Mistral Small 3.1.

FLUX.2 Dev uses Mistral Small 3.1 as its text encoder, extracting hidden states
from layers (10, 20, 30) and concatenating them for the context embedding.
"""

# Layers to extract from Mistral encoder for FLUX.2 Dev
# These are different from Klein's (9, 18, 27) Qwen3 layers
DEV_EXTRACTION_LAYERS = (10, 20, 30)

# System message for FLUX.2 Dev prompt formatting
# This is used with Mistral's chat template
FLUX2_DEV_SYSTEM_MESSAGE = (
    "You are an assistant designed to generate superior images with the highest degree of detail and quality. "
    "Describe the most detailed and unique form of the image generation request from the user. "
    "Do not ask for clarification and do not output anything other than the description of the image itself. "
    "Only output single scene and single subject. Do not output multiple images."
)


def format_flux2_dev_prompt(prompt: str) -> list[dict[str, str]]:
    """Format a prompt for FLUX.2 Dev using Mistral's chat template.

    Args:
        prompt: The user's image generation prompt.

    Returns:
        A list of message dictionaries for Mistral's chat template.
    """
    return [
        {"role": "system", "content": FLUX2_DEV_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
