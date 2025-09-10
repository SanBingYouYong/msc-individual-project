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
import shutil

import os
import sys
sys.path.append(os.getcwd())



'''
Given a job folder (e.g. eval/evals/is_bedroom_10), a dataset path (e.g. dataset/is_bedroom_10/), a path to a exp on dataset folder (e.g. exp/is_bedroom_10) containing each subdir named after scene_id: 
For each scene_id subdir, there should be both a csp/ and direct/ folder for both tested methods, for both: 
1. checks the existence of final/ folder and layout.json inside
    if not exists, early terminate
2. read and record the content of the layout.json as `generated_layout_<method>`
3. read and record dataset_path/dataset/<scene_id>.json as `gt_layout`
    the json file's normalized_layout_bboxes.objects field is a list of object layout dicts, we need to combine the list with the json's object_descriptions field to produce a layout dict with object names and their layouts
    also checks the existence of bbox/<scene_id>.obj, if not exists, raise error
4. under the job folder, create a <scene_id>-layouts.json file containing the above 3 fields (generated_layout_csp, generated_layout_direct, gt_layout), and copy the corresponding bbox/<scene_id>.obj to <scene_id>-bbox-gt.obj
5. also create a metadata.json file recording the dataset_path and exp_on_dataset_path
'''
def prep_subset_eval(job_folder: Path, dataset_path: Path, exp_on_dataset_path: Path):
    """
    Prepare subset evaluation data by collecting gt/generated layouts and ground truth layouts bbox obj files. Also prepares metadata.json.
    
    Args:
        job_folder: Path to the evaluation job folder (e.g. eval/evals/is_bedroom_10)
        dataset_path: Path to the dataset folder (e.g. dataset/is_bedroom_10/)
        exp_on_dataset_path: Path to the experiment folder (e.g. exp/is_bedroom_10)
    """
    
    # Create job folder if it doesn't exist
    job_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all scene_id subdirectories from the experiment folder
    scene_dirs = [d for d in exp_on_dataset_path.iterdir() if d.is_dir()]
    
    processed_scenes = []
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        
        # Initialize data structure for this scene
        scene_data = {
            "scene_id": scene_id,
            "generated_layout_csp": None,
            "generated_layout_direct": None,
            "gt_layout": None
        }
        
        # Check and process CSP method
        csp_final_dir = scene_dir / "csp" / "final"
        csp_layout_file = csp_final_dir / "layout.json"
        
        if csp_layout_file.exists():
            try:
                with open(csp_layout_file, 'r') as f:
                    scene_data["generated_layout_csp"] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read CSP layout for {scene_id}: {e}")
        else:
            print(f"Warning: CSP layout not found for {scene_id}")
        
        # Check and process Direct method
        direct_final_dir = scene_dir / "direct" / "final"
        direct_layout_file = direct_final_dir / "layout.json"
        
        if direct_layout_file.exists():
            try:
                with open(direct_layout_file, 'r') as f:
                    scene_data["generated_layout_direct"] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read Direct layout for {scene_id}: {e}")
        else:
            print(f"Warning: Direct layout not found for {scene_id}")
        
        # If both methods failed, skip this scene
        if scene_data["generated_layout_csp"] is None and scene_data["generated_layout_direct"] is None:
            print(f"Skipping {scene_id}: No valid layouts found")
            continue
        
        # Read ground truth layout
        gt_layout_file = dataset_path / "dataset" / f"{scene_id}.json"
        if not gt_layout_file.exists():
            print(f"Error: Ground truth layout not found for {scene_id} at {gt_layout_file}")
            continue
        
        try:
            with open(gt_layout_file, 'r') as f:
                gt_data = json.load(f)
                
                # Combine object_descriptions with normalized_layout_bboxes.objects
                combined_layout = {}
                object_descriptions = gt_data.get("object_descriptions", [])
                layout_objects = gt_data.get("normalized_layout_bboxes", {}).get("objects", [])
                
                if len(object_descriptions) != len(layout_objects):
                    print(f"Warning: Mismatch between object_descriptions ({len(object_descriptions)}) and layout objects ({len(layout_objects)}) for {scene_id}")
                
                # Combine descriptions with layout info in dict of dicts structure
                for description, layout in zip(object_descriptions, layout_objects):
                    combined_layout[description] = layout
                
                # Store the gt_layout in the same structure as generated layouts
                scene_data["gt_layout"] = combined_layout
                
        except Exception as e:
            print(f"Error: Failed to read ground truth layout for {scene_id}: {e}")
            continue
        
        # Check for bbox OBJ file
        bbox_obj_file = dataset_path / "bbox" / f"{scene_id}.obj"
        if not bbox_obj_file.exists():
            print(f"Error: Bbox OBJ file not found for {scene_id} at {bbox_obj_file}")
            continue
        
        # Save scene data to job folder
        scene_output_file = job_folder / f"{scene_id}-layouts.json"
        try:
            with open(scene_output_file, 'w') as f:
                json.dump(scene_data, f, indent=2)
        except Exception as e:
            print(f"Error: Failed to save scene data for {scene_id}: {e}")
            continue
        
        # Copy bbox OBJ file to job folder
        bbox_output_file = job_folder / f"{scene_id}-bbox-gt.obj"
        try:
            shutil.copy2(bbox_obj_file, bbox_output_file)
        except Exception as e:
            print(f"Error: Failed to copy bbox OBJ for {scene_id}: {e}")
            continue
        
        processed_scenes.append(scene_id)
        print(f"Successfully processed {scene_id}")
    
    # Create metadata.json
    metadata = {
        "dataset_path": str(dataset_path),
        "exp_on_dataset_path": str(exp_on_dataset_path),
        "processed_scenes": processed_scenes,
        "total_scenes": len(processed_scenes)
    }
    
    metadata_file = job_folder / "metadata.json"  # this file is only touched here, subsequent operations should be read-only (NOTE: different to data subset's metadata)
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Created metadata file with {len(processed_scenes)} processed scenes")
    except Exception as e:
        print(f"Error: Failed to create metadata file: {e}")
    
    print(f"Evaluation preparation complete. Processed {len(processed_scenes)} scenes in {job_folder}")

if __name__ == '__main__':
    job_folder = Path("eval/evals/is_bedroom_100r33")
    dataset_path = Path("dataset/is_bedroom_100")
    exp_on_dataset_path = Path("exp/is_bedroom_100")
    
    prep_subset_eval(job_folder, dataset_path, exp_on_dataset_path)
