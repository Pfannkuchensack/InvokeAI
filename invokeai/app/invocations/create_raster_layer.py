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
)
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("create_raster_layer_output")
class CreateRasterLayerOutput(BaseInvocationOutput):
    """Output from creating a raster layer in the canvas."""

    layer_name: Optional[str] = OutputField(default=None, description="The name of the created layer")
    position_x: int = OutputField(description="X position of the layer")
    position_y: int = OutputField(description="Y position of the layer")
    width: int = OutputField(description="Width of the layer")
    height: int = OutputField(description="Height of the layer")
    image: Optional[ImageField] = OutputField(default=None, description="Image placed on the layer, if any")
    opacity: float = OutputField(description="Opacity of the layer (0-1)")
    is_selected: bool = OutputField(description="Whether the layer was selected after creation")


@invocation(
    "create_raster_layer",
    title="Create Raster Layer",
    tags=["canvas", "layer", "raster", "create"],
    category="canvas",
    version="1.0.0",
    classification=Classification.Beta,
)
class CreateRasterLayerInvocation(BaseInvocation):
    """Creates a new raster layer in the canvas session.

    The layer is created on the frontend when this node completes execution.
    Optionally places an image on the new layer."""

    name: Optional[str] = InputField(default=None, description="Optional name for the new layer")
    position_x: int = InputField(default=0, description="X position of the layer on the canvas")
    position_y: int = InputField(default=0, description="Y position of the layer on the canvas")
    width: int = InputField(default=512, ge=1, description="Width of the layer area")
    height: int = InputField(default=512, ge=1, description="Height of the layer area")
    image: Optional[ImageField] = InputField(default=None, description="Optional image to place on the layer")
    opacity: float = InputField(default=1.0, ge=0.0, le=1.0, description="Layer opacity (0 = transparent, 1 = opaque)")
    is_selected: bool = InputField(default=True, description="Whether to select the new layer after creation")

    def invoke(self, context: InvocationContext) -> CreateRasterLayerOutput:
        # Validate image exists if provided
        if self.image is not None:
            context.images.get_dto(self.image.image_name)

        return CreateRasterLayerOutput(
            layer_name=self.name,
            position_x=self.position_x,
            position_y=self.position_y,
            width=self.width,
            height=self.height,
            image=self.image,
            opacity=self.opacity,
            is_selected=self.is_selected,
        )
