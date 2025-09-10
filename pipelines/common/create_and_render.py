import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path to import modules
sys.path.append(os.getcwd())

from pydantic import BaseModel, Field
from pipelines.common.basic.create_scene import create_scene, CreateSceneInput, CreateSceneOutput
from pipelines.common.basic.render_scene import render_scene, RenderSceneInput, RenderSceneOutput
from pipelines.common.prep_scene import ScenePrepOutput

class CreateAndRenderInput(BaseModel):
    layout_path: Path = Field(..., description="Path to the layout JSON file containing asset positions and properties")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from prep_scene pipeline containing generated model files")
    output_blend_path: Path = Field(..., description="Path where the final Blender scene file will be saved")
    render_folder: Path = Field(..., description="Folder to save rendered images")
    image_prefix: Optional[str] = Field(
        None,
        description="Prefix for rendered image filenames. If None, defaults to 'render'."
    )

class CreateAndRenderOutput(BaseModel):
    create_scene_output: CreateSceneOutput = Field(..., description="Output from the create_scene pipeline")
    render_scene_output: RenderSceneOutput = Field(..., description="Output from the render_scene pipeline")

def create_and_render(input_data: CreateAndRenderInput) -> CreateAndRenderOutput:
    """
    Pipeline that creates a Blender scene and renders it sequentially.
    
    Args:
        input_data: CreateAndRenderInput containing all necessary parameters
        
    Returns:
        CreateAndRenderOutput with results from both create_scene and render_scene
    """
    
    # Step 1: Create the scene
    create_scene_input = CreateSceneInput(
        layout_path=input_data.layout_path,
        scene_prep_output=input_data.scene_prep_output,
        output_blend_path=input_data.output_blend_path
    )
    
    create_scene_output = create_scene(create_scene_input)
    
    # Step 2: Render the scene
    render_scene_input = RenderSceneInput(
        scene_blend_file=create_scene_output.blend_file_path,
        render_folder=input_data.render_folder,
        image_prefix=input_data.image_prefix
    )
    
    render_scene_output = render_scene(render_scene_input)
    
    return CreateAndRenderOutput(
        create_scene_output=create_scene_output,
        render_scene_output=render_scene_output
    )


if __name__ == '__main__':
    # Example usage for testing
    from test_data.examples import example_layout, example_sceneprep_output
    
    test_input = CreateAndRenderInput(
        layout_path=Path("test_data/temp_layout_test_scaling.json"),
        scene_prep_output=example_sceneprep_output,
        output_blend_path=Path("temp/create_and_render_scene.blend"),
        render_folder=Path("temp/create_and_render_images"),
        image_prefix="test_render"
    )
    
    output = create_and_render(test_input)
    print(f"Scene created and rendered successfully!")
    print(f"Blend file saved to: {output.create_scene_output.blend_file_path}")
    print(f"Rendered images: {output.render_scene_output.rendered_images}")
