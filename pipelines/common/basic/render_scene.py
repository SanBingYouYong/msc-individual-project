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

import bpy
import mathutils
import math
from typing import Optional

from pipelines.common.utils import stdout_redirected

def adjust_cam_and_render(output_folder: Path, prefix: Optional[str] = None):
    # Ensure the output folder is absolute and exists
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Set up file naming
    if prefix is None:
        prefix = "render"
    
    # Define output file names for X, Y, Z renders
    axes = ['x', 'y', 'z']
    output_file_paths = [
        output_folder / f"{prefix}-{axis}.png" for axis in axes
    ]
    
    # Get all objects in the scene
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not objects:
        print("No objects found in the scene to adjust the camera.")
        return None

    # Calculate the bounding box of all objects
    min_coords = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_coords = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            min_coords = mathutils.Vector((min(min_coords[i], world_corner[i]) for i in range(3)))
            max_coords = mathutils.Vector((max(max_coords[i], world_corner[i]) for i in range(3)))

    # Calculate the center and size of the bounding box
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords

    # Adjust the camera
    camera = bpy.context.scene.camera
    if not camera:
        # Create a new camera if none exists
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        bpy.context.scene.camera = camera

    # Set the camera to orthographic mode
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(size.x, size.y, size.z) * 1.1  # Scale to fit the scene

    # Render from positive X, Y, and Z axes
    axes = ['X', 'Y', 'Z']
    positions = [
        (center.x + size.x * 2, center.y, center.z),  # Positive X
        (center.x, center.y + size.y * 2, center.z),  # Positive Y
        (center.x, center.y, center.z + size.z * 2)   # Positive Z
    ]
    rotations = [
        (math.radians(90), 0, math.radians(90)),  # Look along X axis
        (math.radians(90), 0, math.radians(180)), # Look along Y axis
        (0, 0, 0)                  # Look along Z axis
    ]

    for i, (position, rotation) in enumerate(zip(positions, rotations)):
        camera.location = position
        camera.rotation_euler = rotation

        # Render the scene
        bpy.context.scene.render.filepath = str(output_file_paths[i])
        bpy.ops.render.render(write_still=True)
    
    return output_file_paths

def render_prep():
    # add background lighting
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]
    bpy.data.worlds["World"].use_nodes = True
    # my favorite #B8B8B8 but hey maybe we should just use white
    # bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.479317, 0.479321, 0.47932, 1)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)  # white background
    # set render settings
    # bpy.context.scene.render.engine = 'CYCLES'  # this is slow but the visual quality is good
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    # bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    # bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

class RenderSceneInput(BaseModel):
    scene_blend_file: Path = Field(..., description="Path to the Blender scene file.")
    render_folder: Path = Field(..., description="Folder to save rendered images.")
    image_prefix: Optional[str] = Field(
        None,
        description="Prefix for rendered image filenames. If None, defaults to 'render'."
    )

class RenderSceneOutput(BaseModel):
    rendered_images: list[Path] = Field(
        ...,
        description="List of paths to the rendered images for X, Y, and Z axes."
    )

def render_scene(input_data: RenderSceneInput) -> RenderSceneOutput:
    
    # with stdout_redirected():  # let's save this for debug purposes
    # Load the Blender scene
    bpy.ops.wm.open_mainfile(filepath=str(input_data.scene_blend_file.resolve()))

    # Ensure the render folder exists
    input_data.render_folder.mkdir(parents=True, exist_ok=True)

    render_prep()

    # Adjust camera and render the scene
    rendered_images = adjust_cam_and_render(
        output_folder=input_data.render_folder,
        prefix=input_data.image_prefix
    )

    if rendered_images is None:
        raise RuntimeError("No objects found in the scene to render.")
    
    # For testing: save the current .blend file next to the input blend file
    # test_save_path = input_data.scene_blend_file.with_name(
    #     input_data.scene_blend_file.stem + "_after_render.blend"
    # )
    # bpy.ops.wm.save_as_mainfile(filepath=str(test_save_path))

    return RenderSceneOutput(rendered_images=rendered_images)


if __name__ == '__main__':
    from test_data.examples import example_scene_blend_file

    from yaspin import yaspin

    with yaspin(text="Rendering...", color="green", timer=True) as spinner:
        example_render_folder = Path("temp/rendered_images")
        example_input = RenderSceneInput(
            scene_blend_file=example_scene_blend_file,
            render_folder=example_render_folder,
            image_prefix="example_render"
        )
        example_output = render_scene(example_input)
        spinner.ok("✔")
    # print(f"Rendered images saved to: {example_output.rendered_images}")
