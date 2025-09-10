'''
This file loops over 4k subdir under InstructScene/threed_front_bedroom/ and reads the preprocessed extracted_scene.json file to get object JIDs, then retrieve corresponding object model OBJ files under 3D-FUTURE-model/<JID>/raw_model.obj, and read their vertex coordinates to compute their sizes; combining OBJ model sizes with boxes.npz file under each subdir for scale information, it calculates a normalized bounding box for each object in the scene, and save the updated scene_with_bbox.json file under each subdir.
NOTE: 
- it turns out we can use 3D-future model size directly for bbox, and ignore the mysterious 0.5 scale factor from instruct scene bboxes.npz file
- the scale factor between scene size and original object size seems to always be 0.5x, so we can ignore it as well
- recording object_size and scale_factor in the bbox json for reference
- to work in blender, they need to be (as a whole) rotated 90 degree around X axis
NOTE: symlinks are moved under dataset dir after this script have been run - updated found path references but may not still have some old references in effect.
'''

import os
from pathlib import Path
import numpy as np
import time
from pydantic import BaseModel, Field
import sys
import json
from typing import Tuple, List, Dict
from tqdm import tqdm
sys.path.append(os.getcwd())

class LayoutExtra(BaseModel):
    location: Tuple[float, float, float] = Field(..., description="location of the asset in 3D space")
    min: Tuple[float, float, float] = Field(..., description="minimum corner of the AABB bounding box")
    max: Tuple[float, float, float] = Field(..., description="maximum corner of the AABB bounding box")
    orientation: Tuple[float, float, float] = Field(..., description="Euler angles (pitch, yaw, roll) in radians")
    object_size: Tuple[float, float, float] = Field(..., description="actual object dimensions from OBJ model")
    scale_factor: Tuple[float, float, float] = Field(..., description="scale factor applied in the scene")


def read_obj_vertices(obj_path: Path) -> np.ndarray:
    """
    Read vertex coordinates from OBJ file.
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        Array of vertex coordinates (Nx3)
    """
    vertices = []
    
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):  # vertex line
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
    except Exception as e:
        print(f"Error reading OBJ file {obj_path}: {e}")
        return np.array([[0, 0, 0]])
    
    return np.array(vertices) if vertices else np.array([[0, 0, 0]])


def compute_object_size(vertices: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute object size from vertex coordinates.
    
    Args:
        vertices: Array of vertex coordinates (Nx3)
        
    Returns:
        Object dimensions (width, height, depth)
    """
    if len(vertices) == 0:
        return (0.0, 0.0, 0.0)
    
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    size = max_coords - min_coords
    return (float(size[0]), float(size[1]), float(size[2]))


def extract_normalized_bbox(npz_path: Path, object_jids: List[str], 
                          model_base_path: Path) -> Dict[str, List[LayoutExtra]]:
    """
    Extract normalized bounding box information combining scene data and OBJ model sizes.
    
    Args:
        npz_path: Path to the boxes.npz file containing scene data
        object_jids: List of object JIDs in the scene
        model_base_path: Base path to 3D-FUTURE-model directory
        
    Returns:
        Dictionary containing list of Layout objects with normalized bounding box information
    """
    try:
        # Load the npz file
        data = np.load(npz_path)
        
        # Extract the required arrays
        translations = data['translations']  # Object positions (x, y, z)
        sizes = data['sizes']               # Object sizes/scales in scene (width, height, depth)
        angles = data['angles']             # Object rotations (in radians)
        
        # Create Layout objects for each object
        layouts = []
        
        for i in range(len(translations)):
            if i >= len(object_jids):
                break
                
            jid = object_jids[i]
            
            # Get position, size, and rotation for this object from scene data
            position = translations[i]  # (x, y, z)
            scene_size = sizes[i]      # (width, height, depth) - scaled size in scene
            angle = angles[i]          # rotation angle in radians
            
            # Read OBJ model to get actual object dimensions
            obj_path = model_base_path / jid / "raw_model.obj"
            if obj_path.exists():
                vertices = read_obj_vertices(obj_path)
                obj_size = compute_object_size(vertices)
            else:
                print(f"Warning: OBJ file not found for {jid}")
                obj_size = (1.0, 1.0, 1.0)  # Default size
            
            # Calculate scale factor between scene size and original object size
            if obj_size[0] > 0 and obj_size[1] > 0 and obj_size[2] > 0:
                scale_factor = (
                    float(scene_size[0] / obj_size[0]),  # NOTE: it seems obj_size is correct already, the size from bboxes.npz simply halves them for no reason
                    float(scene_size[1] / obj_size[1]),
                    float(scene_size[2] / obj_size[2])
                )
            else:
                scale_factor = (1.0, 1.0, 1.0)
            
            # Location is the center position of the object
            location = (float(position[0]), float(position[1]), float(position[2]))
            
            # Calculate AABB (Axis-Aligned Bounding Box) corners using scene size
            # half_size = scene_size# / 2.0
            half_size = np.asarray(obj_size) / 2.0  # NOTE: so we use actual object size for bbox
            min_corner = (
                float(position[0] - half_size[0]),
                float(position[1] - half_size[1]), 
                float(position[2] - half_size[2])
            )
            max_corner = (
                float(position[0] + half_size[0]),
                float(position[1] + half_size[1]),
                float(position[2] + half_size[2])
            )
            
            # Orientation: convert single angle to Euler angles (pitch, yaw, roll)
            # Assuming the angle represents rotation around Y-axis (yaw)
            orientation = (0.0, float(angle[0]), 0.0)  # (pitch, yaw, roll)
            
            # Create Layout object with normalized bounding box
            layout = LayoutExtra(
                location=location,
                min=min_corner,
                max=max_corner,
                orientation=orientation,
                object_size=obj_size,
                scale_factor=scale_factor
            )
            layouts.append(layout)
        
        return {"objects": layouts}
        
    except Exception as e:
        print(f"Error processing npz file {npz_path}: {e}")
        return {"objects": []}


def make_serializable(obj):
    """Convert numpy arrays and complex objects to serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, LayoutExtra):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj


def process_normalized_bbox_data(dataset_path: Path = None, model_base_path: Path = None):
    """
    Process InstructScene dataset to compute normalized bounding boxes by combining
    scene data with actual OBJ model dimensions.
    
    Args:
        dataset_path: Path to InstructScene/threed_front_bedroom/ directory
        model_base_path: Path to 3D-FUTURE-model/ directory
    """
    if dataset_path is None:
        dataset_path = Path("dataset/InstructScene/threed_front_bedroom/")
    if model_base_path is None:
        model_base_path = Path("dataset/3D-FUTURE-model/")
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return
    
    if not model_base_path.exists():
        print(f"Error: Model base path {model_base_path} does not exist")
        return
    
    processed_scenes = 0
    skipped_scenes = 0
    
    # Get list of valid subdirectories
    valid_subdirs = []
    for subdir in dataset_path.iterdir():
        if not subdir.is_dir():
            continue
        # Skip _train* and _test* directories
        if subdir.name.startswith("_train") or subdir.name.startswith("_test"):
            continue
        
        # Check if extracted_scene.json exists
        extracted_scene_file = subdir / "extracted_scene.json"
        if not extracted_scene_file.exists():
            continue
            
        valid_subdirs.append(subdir)
    
    print(f"Found {len(valid_subdirs)} valid scenes to process")
    
    # Process each subdirectory
    for subdir in tqdm(valid_subdirs, desc="Processing scenes", unit="scene"):
        try:
            # Read the preprocessed extracted_scene.json
            extracted_scene_file = subdir / "extracted_scene.json"
            with open(extracted_scene_file, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            # Get object JIDs
            object_jids = scene_data.get('object_jids', [])
            if not object_jids:
                tqdm.write(f"Warning: No object JIDs found in {subdir.name}")
                skipped_scenes += 1
                continue
            
            # Check if boxes.npz exists
            boxes_file = subdir / "boxes.npz"
            if not boxes_file.exists():
                tqdm.write(f"Warning: boxes.npz not found in {subdir.name}")
                skipped_scenes += 1
                continue
            
            # Extract normalized bounding boxes
            normalized_layouts = extract_normalized_bbox(
                boxes_file, object_jids, model_base_path
            )
            
            # Update scene data with normalized bounding boxes
            scene_data['normalized_layout_bboxes'] = normalized_layouts
            
            # Save updated scene data to scene_with_bbox.json
            serializable_scene_data = make_serializable(scene_data)
            
            output_file = subdir / "scene_with_bbox.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_scene_data, f, indent=2, ensure_ascii=False)
            
            processed_scenes += 1
            
        except Exception as e:
            tqdm.write(f"Error processing scene {subdir.name}: {e}")
            skipped_scenes += 1
            continue
    
    print(f"\nProcessing completed:")
    print(f"- Successfully processed: {processed_scenes} scenes")
    print(f"- Skipped: {skipped_scenes} scenes")
    print(f"- Total valid scenes: {len(valid_subdirs)}")
    
    return processed_scenes


if __name__ == '__main__':
    # Process the dataset to compute normalized bounding boxes
    print("Processing InstructScene dataset to compute normalized bounding boxes...")
    process_normalized_bbox_data()