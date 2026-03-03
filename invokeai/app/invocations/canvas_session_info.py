from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    ImageField,
    InputField,
    OutputField,
    UIComponent,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX


@invocation_output("canvas_session_info_output")
class CanvasSessionInfoOutput(BaseInvocationOutput):
    """Output containing all canvas session info parameters."""

    # Bbox
    bbox_x: int = OutputField(description="X position of the bounding box")
    bbox_y: int = OutputField(description="Y position of the bounding box")
    bbox_width: int = OutputField(description="Width of the bounding box")
    bbox_height: int = OutputField(description="Height of the bounding box")
    scaled_width: int = OutputField(description="Scaled width for generation")
    scaled_height: int = OutputField(description="Scaled height for generation")

    # Model info
    model_name: str = OutputField(description="Name of the selected model")
    model_base: str = OutputField(description="Base type of the selected model (e.g. sd-1, sdxl, flux)")
    model_key: str = OutputField(description="Key identifier of the selected model")

    # Prompts
    positive_prompt: str = OutputField(description="Positive prompt text")
    negative_prompt: str = OutputField(description="Negative prompt text")

    # Generation params
    steps: int = OutputField(description="Number of inference steps")
    cfg_scale: float = OutputField(description="CFG scale value")
    guidance: float = OutputField(description="Guidance value (for Flux-type models)")
    seed: int = OutputField(description="Random seed")
    scheduler: str = OutputField(description="Scheduler name")
    img2img_strength: float = OutputField(description="img2img denoising strength")


@invocation(
    "canvas_session_info",
    title="Canvas Session Info",
    tags=["canvas", "session", "info", "bbox", "params"],
    category="canvas",
    version="1.0.0",
    classification=Classification.Beta,
)
class CanvasSessionInfoInvocation(BaseInvocation):
    """Provides canvas session parameters as outputs for use in workflows.

    When run from the canvas context menu, input values are automatically
    populated from the current canvas state (bbox, model, prompts, etc.).
    Can also be used in the workflow editor with manually set values."""

    # Bbox fields
    bbox_x: int = InputField(default=0, description="X position of the bounding box")
    bbox_y: int = InputField(default=0, description="Y position of the bounding box")
    bbox_width: int = InputField(default=512, ge=64, description="Width of the bounding box")
    bbox_height: int = InputField(default=512, ge=64, description="Height of the bounding box")
    scaled_width: int = InputField(default=512, ge=64, description="Scaled width for generation")
    scaled_height: int = InputField(default=512, ge=64, description="Scaled height for generation")

    # Model info
    model_name: str = InputField(default="", description="Name of the selected model")
    model_base: str = InputField(default="", description="Base type of the selected model (e.g. sd-1, sdxl, flux)")
    model_key: str = InputField(default="", description="Key identifier of the selected model")

    # Prompts
    positive_prompt: str = InputField(
        default="", description="Positive prompt text", ui_component=UIComponent.Textarea
    )
    negative_prompt: str = InputField(
        default="", description="Negative prompt text", ui_component=UIComponent.Textarea
    )

    # Generation params
    steps: int = InputField(default=30, ge=1, description="Number of inference steps")
    cfg_scale: float = InputField(default=7.5, ge=1.0, description="CFG scale value")
    guidance: float = InputField(default=4.0, ge=0.0, description="Guidance value (for Flux-type models)")
    seed: int = InputField(default=0, ge=0, le=SEED_MAX, description="Random seed")
    scheduler: str = InputField(default="euler", description="Scheduler name")
    img2img_strength: float = InputField(
        default=0.75, ge=0.0, le=1.0, description="img2img denoising strength"
    )

    def invoke(self, context: InvocationContext) -> CanvasSessionInfoOutput:
        return CanvasSessionInfoOutput(
            bbox_x=self.bbox_x,
            bbox_y=self.bbox_y,
            bbox_width=self.bbox_width,
            bbox_height=self.bbox_height,
            scaled_width=self.scaled_width,
            scaled_height=self.scaled_height,
            model_name=self.model_name,
            model_base=self.model_base,
            model_key=self.model_key,
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            guidance=self.guidance,
            seed=self.seed,
            scheduler=self.scheduler,
            img2img_strength=self.img2img_strength,
        )
