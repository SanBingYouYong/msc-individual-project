import bpy
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.getcwd())

from pydantic import BaseModel, Field
from pipelines.common.prep_scene import ScenePrepOutput

class CreateSceneInput(BaseModel):
    layout_path: Path = Field(..., description="Path to the layout JSON file containing asset positions and properties")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from prep_scene pipeline containing generated model files")
    output_blend_path: Path = Field(..., description="Path where the final Blender scene file will be saved")

class CreateSceneOutput(BaseModel):
    blend_file_path: Path = Field(..., description="Path to the saved Blender scene file")

def create_scene(input_data: CreateSceneInput) -> CreateSceneOutput:
    """
    Create a Blender scene by importing 3D models and positioning them according to the layout specification.
    Assumption: layout has been verified that layout items match exactly what 3D models we generated.
    
    Args:
        input_data: CreateSceneInput containing layout path, scene prep output, and output path
        
    Returns:
        CreateSceneOutput with scene creation results and statistics
    """
    # Load layout data
    with open(input_data.layout_path, "r") as f:
        layout_data = json.load(f)
    
    # Get available models from scene prep output
    available_models = input_data.scene_prep_output.obtained_asset_files
    
    # Initialize a new Blender scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Process each asset in the layout
    for asset_name, asset_data in layout_data.items():
        if asset_name not in available_models:
            raise ValueError(f"Asset '{asset_name}' not found in scene prep output. Available assets: {list(available_models.keys())}")
        
        glb_path = available_models[asset_name]
        
        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
        
        # Find the imported empty object (assumption: text-to-3d (trellis) pipeline creates geometry under an empty object named 'world')
        imported_asset = bpy.data.objects.get("world")
        if imported_asset is None:
            # retrieved shapenet asset will be named "model_normalized"
            imported_asset = bpy.data.objects.get('model_normalized')
        if imported_asset is None:
            raise ValueError(f"Imported asset '{asset_name}' does not contain a valid object. Check the GLB file.")
        
        # Rename the empty object to the asset name
        imported_asset.name = asset_name
        
        # Apply layout parameters to the empty object
        imported_asset.location = asset_data.get("location", (0, 0, 0))
        imported_asset.rotation_euler = asset_data.get("rotation", (0, 0, 0))
        
        # Calculate scale from min/max coordinates
        # TODO: min/max output is inherited from scenecraft; consider replacing with unified scale param
        min_coord = asset_data.get("min", (0, 0, 0))
        max_coord = asset_data.get("max", (1, 1, 1))
        scale_x = max_coord[0] - min_coord[0]
        scale_y = max_coord[1] - min_coord[1] 
        scale_z = max_coord[2] - min_coord[2]
        # scale_x *= 2
        # scale_y *= 2
        # scale_z *= 2
        imported_asset.scale = (scale_x, scale_y, scale_z)
    
    # Ensure output directory exists
    input_data.output_blend_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the scene to a .blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(input_data.output_blend_path))
    
    return CreateSceneOutput(
        blend_file_path=input_data.output_blend_path
    )


if __name__ == '__main__':
    # Example usage for testing
    from pipelines.common.prep_scene import ScenePrepOutput
    from agents.decomposer import DecomposerOutput

    from test_data.examples import example_layout, example_sceneprep_output
    
    test_input = CreateSceneInput(
        layout_path=Path("test_data/example_layout.json"),
        scene_prep_output=example_sceneprep_output,
        output_blend_path=Path("temp/coder-layout_scene.blend")
    )
    
    output = create_scene(test_input)
    print(f"Scene created successfully!")
    print(f"Blend file saved to: {output.blend_file_path}")
