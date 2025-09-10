'''
Based on scene_with_bbox.json in each subdir under InstructScene/threed_front_bedroom/, this file reads in the object bboxes and instantiates the bounding boxes in Blender, rotate them together around X-aixs for 90 degrees to fit Blender axis convention, and normalizes them together to fit in a 1x1x1 unit cube centered at origin. The normalized bboxes are saved in blender_normalized_bboxes.json under each subdir.
NOTE: symlinks are moved under dataset dir after this script have been run - updated found path references but may not still have some old references in effect.
'''

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

from eval.utils import rotate_and_normalize_scene

def create_bounding_box(name, location, min_coords, max_coords, orientation):
    # Calculate scale from min and max
    scale = [(max_coords[i] - min_coords[i]) for i in range(3)]
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.rotation_euler = orientation
    obj.display_type = 'WIRE'

def clear_scene():
    """Clear the Blender scene by removing all mesh objects."""
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)

def create_objects_from_data(scene_data):
    """Create bounding box objects from scene data."""
    objects = scene_data.get('normalized_layout_bboxes', {}).get('objects', [])
    for idx, obj_data in enumerate(objects):
        name = f"obj_{idx}"
        location = obj_data.get('location', [0,0,0])
        min_coords = obj_data.get('min', [-0.5,-0.5,-0.5])
        max_coords = obj_data.get('max', [0.5,0.5,0.5])
        orientation = obj_data.get('orientation', [0,0,0])  # Assuming orientation is in radians
        
        create_bounding_box(name, location, min_coords, max_coords, orientation)

def export_scene_as_obj(output_path):
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

def process_subdirectory(subdir):
    """Process a single subdirectory to create layout ground truth."""
    # print(f"Processing {subdir}...")
    json_path = subdir / 'scene_with_bbox.json'
    if not json_path.exists():
        print(f"Warning: {json_path} does not exist, skipping.")
        return

    with open(json_path, 'r') as f:
        scene_data = json.load(f)
    
    # print(f"Found {len(scene_data.get('normalized_layout_bboxes', {}).get('objects', []))} objects in {json_path}")

    # Clear scene and create objects
    clear_scene()
    create_objects_from_data(scene_data)
    rotate_and_normalize_scene()

    # Save the scene as an OBJ file
    output_obj_path = subdir / 'blender_layout_gt.obj'
    export_scene_as_obj(output_obj_path)

    # print(f"Processed {subdir}, saved scene layout to {output_obj_path}")

def main():
    """Main function to process all subdirectories."""
    # Loop over all subdirs under InstructScene/threed_front_bedroom/
    base_dir = Path('/home/shuyuan/ip-scenecraft/InstructScene/threed_front_bedroom/')
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    # Skip _test and _train
    subdirs = [d for d in subdirs if not d.name.startswith('_')]

    for subdir in tqdm(subdirs):  # too lazy to suppress blender export messages let it be
        process_subdirectory(subdir)

if __name__ == "__main__":
    main()

