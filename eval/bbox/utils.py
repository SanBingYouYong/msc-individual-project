import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
import sys
import bpy
import mathutils
import math

import os
import sys
sys.path.append(os.getcwd())

from eval.utils import normalize_scene

def create_bounding_box(name, location, min_coords, max_coords, orientation):
    # Calculate scale from min and max
    scale = [(max_coords[i] - min_coords[i]) for i in range(3)]
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = orientation
    obj.display_type = 'WIRE'


def export_all_mesh_as_obj(output_path):
    """Export the current scene as an OBJ file without materials."""
    # Select all mesh objects for export
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # Export as OBJ without materials
    bpy.ops.wm.obj_export(
        filepath=str(output_path),
        export_selected_objects=True,
        export_materials=False,
    )

def create_normalized_bounding_boxes(layout_data, obj_export_path):
    """
    Create normalized bounding boxes from layout data and export as OBJ.
    
    Args:
        layout_data (dict): Dictionary containing layout data
        obj_export_path (str): Path where the OBJ file should be exported
    """
    # reads homefile
    bpy.ops.wm.read_homefile(use_empty=True)

    # Create bounding boxes for each object in the scene
    for object_name, object_data in layout_data.items():
        location = object_data['location']
        min_coords = object_data['min']
        max_coords = object_data['max']
        orientation = object_data['orientation']
        
        # Convert orientation from degrees to radians for Blender
        orientation_rad = [math.radians(angle) for angle in orientation]
        
        create_bounding_box(object_name, location, min_coords, max_coords, orientation_rad)

    # Normalize the scene to fit within a unit cube
    normalize_scene()

    # Export as OBJ
    export_all_mesh_as_obj(obj_export_path)

if __name__ == '__main__':
    with open("eval/_generated_layout.json", 'r') as f:
        layout_data = json.load(f)
    create_normalized_bounding_boxes(layout_data, "eval/_generated_layout.obj")
