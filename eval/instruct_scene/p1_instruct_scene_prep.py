'''
This file loops over 4k data entries under InstructScene/threed_front_bedroom/ and picks 2/4/6th images under each ./blender_rendered_scene_256 to describe the scene with local ollama-hosted gemma3. (captions) update: used all available images
Extracted scene info and scene description are saved in extracted_scene.json under each subdir.
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
from llms.llms import helper

class Layout(BaseModel):
    location: Tuple[float, float, float] = Field(..., description="location of the asset in 3D space")
    min: Tuple[float, float, float] = Field(..., description="minimum corner of the AABB bounding box")
    max: Tuple[float, float, float] = Field(..., description="maximum corner of the AABB bounding box")
    orientation: Tuple[float, float, float] = Field(..., description="Euler angles (pitch, yaw, roll) in radians")

# NOTE: this bbox size is dependent on object size from 3D-Front, which were not properly normalized
# UPDATE: it seems to just be half-size, 3D-FUTURE-models are in their proper sizes just not normalized
def extract_bbox(npz_path: Path) -> Dict[str, List[Layout]]:
    """
    Extract bounding box information from npz file and convert to Layout objects.
    
    Args:
        npz_path: Path to the npz file containing scene data
        
    Returns:
        Dictionary containing list of Layout objects with bounding box information
    """
    # Load the npz file
    data = np.load(npz_path)
    
    # Extract the required arrays
    translations = data['translations']  # Object positions (x, y, z)
    sizes = data['sizes']               # Object sizes (width, height, depth)
    angles = data['angles']             # Object rotations (in radians)
    
    # Create Layout objects for each object
    layouts = []
    
    for i in range(len(translations)):
        # Get position, size, and rotation for this object
        position = translations[i]  # (x, y, z)
        size = sizes[i]            # (width, height, depth)
        angle = angles[i]          # rotation angle in radians
        
        # Location is the center position of the object
        location = (float(position[0]), float(position[1]), float(position[2]))
        
        # Calculate AABB (Axis-Aligned Bounding Box) corners
        # For axis-aligned boxes, we can calculate min/max directly from center and size
        # half_size = size / 2.0
        half_size = size
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
        
        # Create Layout object
        layout = Layout(
            location=location,
            min=min_corner,
            max=max_corner,
            orientation=orientation
        )
        layouts.append(layout)
    
    return {"objects": layouts}

# object_desc_path = Path("InstructScene/3D-FUTURE-chatgpt")
# dataset_path = Path("InstructScene/threed_front_bedroom/")
# example subdir: 0a9f5311-49e1-414c-ba7b-b42a171459a3_SecondBedroom-18509
# get image paths under subdir/blender_rendered_scene_256
# read boxes.npz file under subdir, and use boxes_data['jids'] to get object JIDs
# JID.txt files under object_desc_path contains object descriptions
# obtain a list of object descriptions from JID.txt files and feed the images to VLMs for scene description, ask them to compactly describe the scene and object relataionships
# extract layout bounding boxes using the extract_bbox function and the subdir/boxes.npz file
# save the data in a structured format to be able to link: subdir, image paths, object JIDs, generated scene descriptions and extracted layout bounding boxes. 

# Convert numpy arrays and complex objects to serializable format
def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, Layout):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj

def prep_instruct_scene_data(object_desc_path: Path=None, dataset_path: Path=None):
    """
    Process InstructScene dataset to extract scene information including:
    - Image paths from rendered scenes
    - Object descriptions from JID files
    - Layout bounding boxes from npz files
    - Generated scene descriptions using VLMs
    """
    if object_desc_path is None:
        object_desc_path = Path("dataset/InstructScene/3D-FUTURE-chatgpt")
    if dataset_path is None:
        dataset_path = Path("dataset/InstructScene/threed_front_bedroom/")
    
    # Initialize data collection
    # scene_data = []
    processed_scenes = 0
    
    # Get list of valid subdirectories first for progress bar
    valid_subdirs = []
    for subdir in dataset_path.iterdir():
        if not subdir.is_dir():
            continue
        # skip _train* and _test* directories
        if subdir.name.startswith("_train") or subdir.name.startswith("_test"):
            continue
        valid_subdirs.append(subdir)
    
    # Iterate through all subdirectories in the dataset with progress bar
    for subdir in tqdm(valid_subdirs, desc="Processing scenes", unit="scene"):
            
        # tqdm.write(f"Processing scene: {subdir.name}")

        # Get image paths under subdir/blender_rendered_scene_256
        image_dir = subdir / "blender_rendered_scene_256"
        image_paths = []
        if image_dir.exists():
            for img_file in image_dir.glob("*.png"):
                image_paths.append(str(img_file))
        
        # Read boxes.npz file and extract object JIDs
        boxes_file = subdir / "boxes.npz"
        if not boxes_file.exists():
            tqdm.write(f"Warning: boxes.npz not found in {subdir.name}")
            continue
            
        # Load boxes data to get JIDs
        boxes_data = np.load(boxes_file)
        if 'jids' not in boxes_data:
            tqdm.write(f"Warning: 'jids' not found in boxes.npz for {subdir.name}")
            continue
            
        object_jids = boxes_data['jids']
        
        # Get object descriptions from JID.txt files
        object_descriptions = []
        for jid in object_jids:
            jid_file = object_desc_path / f"{jid}.txt"
            if jid_file.exists():
                with open(jid_file, 'r', encoding='utf-8') as f:
                    description = f.read().strip()
                    object_descriptions.append(description)
            else:
                tqdm.write(f"Warning: JID file {jid}.txt not found")
                object_descriptions.append(f"Object {jid}")
        
        # Generate scene description using VLM with all available images
        scene_description = ""
        selected_images = []
        if len(image_paths) > 0:
            # Use all available images
            selected_images = image_paths
            
            # Create prompt for VLM
            objects_list = "\n".join([f"- {desc}" for desc in object_descriptions])
            prompt = f"""Please provide a compact description of this bedroom scene and the spatial relationships between objects. 

The scene contains these objects:
{objects_list}

Focus on:
1. Overall layout and room type
2. Key furniture placement and relationships
3. Spatial arrangements (near, adjacent, opposite, etc.)
4. Describe only the above objects

Keep the description concise but informative. Do not respond with any other text or explanations, just the scene description."""

            try:
                # Generate scene description using VLM
                scene_description = helper.query(
                    provider="ollama",
                    model="gemma3",  # GPT-4o-mini with vision support
                    user_prompt=prompt,
                    image_paths=selected_images,
                    temperature=0
                )
                
                # Save the scene description to scene_description.txt in the subdir
                scene_desc_file = subdir / "scene_description.txt"
                with open(scene_desc_file, 'w', encoding='utf-8') as f:
                    f.write(scene_description)
                # tqdm.write(f"Saved scene description to {scene_desc_file}")
                
            except Exception as e:
                tqdm.write(f"Error generating scene description for {subdir.name}: {e}")
                scene_description = "ERROR: Scene description unavailable"
        else:
            tqdm.write(f"Warning: No images found in {subdir.name}")
            scene_description = "No images available for scene description"
        
        # Skip layout bounding boxes extraction for now (bbox data needs extra work)
        # layouts = extract_bbox(boxes_file)
        
        # Store all data for this scene
        scene_info = {
            "scene_id": subdir.name,
            "subdir_path": str(subdir),
            "image_paths": image_paths,
            "object_jids": object_jids.tolist() if hasattr(object_jids, 'tolist') else list(object_jids),
            "object_descriptions": object_descriptions,
            "scene_description": scene_description,
            # "layout_bboxes": layouts  # Commented out until bbox data extraction is ready
        }
        
        # Save individual scene config to subdir/extracted_scene.json
        serializable_scene_info = make_serializable(scene_info)
        scene_config_file = subdir / "extracted_scene.json"
        with open(scene_config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_scene_info, f, indent=2, ensure_ascii=False)
        # tqdm.write(f"Saved scene config to {scene_config_file}")
        processed_scenes += 1
        
        # scene_data.append(scene_info)
        
        # tqdm.write(f"Completed processing {subdir.name}: {len(selected_images)} images, {len(object_descriptions)} objects")
    
    # print(f"Processed {len(scene_data)} scenes. Individual configs saved to each subdirectory as extracted_scene.json")
    # return scene_data
    print(f"Processed {processed_scenes} scenes. Individual configs saved to each subdirectory as extracted_scene.json")
    return processed_scenes

if __name__ == '__main__':
    # Test with example file
    # path = "InstructScene/threed_front_bedroom/0a9f5311-49e1-414c-ba7b-b42a171459a3_SecondBedroom-18509/boxes.npz"
    # if Path(path).exists():
    #     layouts = extract_bbox(Path(path))
    #     print("Example layout extraction:")
    #     print(layouts)
    
    # Process the full dataset
    # print("\nProcessing InstructScene/ dataset...")
    prep_instruct_scene_data()
