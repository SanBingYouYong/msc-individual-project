
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

from eval.bbox.utils import create_bounding_box, export_all_mesh_as_obj, create_normalized_bounding_boxes
from eval.utils import normalize_scene
from eval.bbox.blender_obj_overlap import calculate_iou_from_obj_files
from eval.prep import prep_subset_eval


'''
Given a processed subset eval dir, for each scene: 
1. read <scene_id>-layouts.json, and its generated_layout_<method> fields
2. use create_normalized_bounding_boxes to create normalized bounding boxes for each layout, and export as OBJ files named <scene_id>-bbox-<method>.obj
3. calculate IoU between each generated layout and the gt_layout
finally, record in a overlap.json file under the job folder, essentially a dict of dicts: {scene_id: {method: iou, ...}, ...}
'''
def evaluate_subset_layout_overlap(eval_folder: Path):
    """
    Evaluate layout overlap for each scene in the job folder and save results to overlap.json.
    Depends on running eval/prep.py
    
    Args:
        job_folder: Path to the evaluation job folder (e.g. eval/evals/is_bedroom_10)
    """
    
    # Find all scene JSON files in the job folder
    scene_json_files = list(eval_folder.glob("*-layouts.json"))
    
    if not scene_json_files:
        print(f"No scene JSON files found in {eval_folder}")
        return
    
    overlap_results = {}
    
    for scene_json_file in tqdm(scene_json_files, desc="Evaluating scenes"):
        scene_id = scene_json_file.stem.replace("-layouts", "")
        
        try:
            with open(scene_json_file, 'r') as f:
                scene_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read scene data for {scene_id}: {e}")
            continue
        
        gt_layout = scene_data.get("gt_layout")
        if not gt_layout:
            print(f"Warning: No ground truth layout for {scene_id}, skipping")
            continue
        
        overlap_results[scene_id] = {}
        
        for method in ["generated_layout_csp", "generated_layout_direct"]:
            generated_layout = scene_data.get(method)
            if not generated_layout:
                print(f"Warning: No generated layout for {scene_id} with method {method}, skipping")
                continue
            
            # Create normalized bounding boxes and export as OBJ
            method_suffix = method.replace("generated_layout_", "")
            obj_export_path = eval_folder / f"{scene_id}-bbox-{method_suffix}.obj"
            try:
                create_normalized_bounding_boxes(generated_layout, obj_export_path)
            except Exception as e:
                print(f"Warning: Failed to create bounding boxes for {scene_id} with method {method}: {e}")
                continue
            
            # Use existing ground truth OBJ file
            gt_obj_path = eval_folder / f"{scene_id}-bbox-gt.obj"
            if not gt_obj_path.exists():
                print(f"Warning: Ground truth OBJ file not found for {scene_id}, skipping")
                continue
            
            # Calculate IoU between generated layout and ground truth
            try:
                iou = calculate_iou_from_obj_files(
                    os.path.abspath(obj_export_path),
                    os.path.abspath(gt_obj_path)
                )['best_iou']
                if iou is None:
                    print(f"Warning: IoU calculation failed for {scene_id} with method {method}")
                    continue
                overlap_results[scene_id][method] = iou
            except Exception as e:
                print(f"Warning: Failed to calculate IoU for {scene_id} with method {method}: {e}")
                continue
    
    # calculate average of each method
    num_scenes = len(overlap_results)  # Get scene count before adding averages
    method_sums = {}
    method_counts = {}
    for scene_id, methods in overlap_results.items():
        for method, iou in methods.items():
            if method not in method_sums:
                method_sums[method] = 0.0
                method_counts[method] = 0
            method_sums[method] += iou
            method_counts[method] += 1
    for method in method_sums:
        avg_iou = method_sums[method] / method_counts[method] if method_counts[method] > 0 else 0.0
        print(f"Average IoU for {method}: {avg_iou:.4f} over {method_counts[method]} scenes")
        overlap_results["average_" + method] = avg_iou

    # Save overlap results to overlap.json
    overlap_file = eval_folder / "overlap.json"
    try:
        with open(overlap_file, 'w') as f:
            json.dump(overlap_results, f, indent=2)
        print(f"Saved overlap results to {overlap_file}")
    except Exception as e:
        print(f"Error: Failed to save overlap results: {e}")
        return
    
    print(f"Layout overlap evaluation complete for {num_scenes} scenes.")


# Example usage
if __name__ == "__main__":
    job_folder = Path("eval/evals/is_bedroom_100r33")
    dataset_path = Path("dataset/is_bedroom_100")
    exp_on_dataset_path = Path("exp/is_bedroom_100")
    
    # prep_subset_eval(job_folder, dataset_path, exp_on_dataset_path)
    evaluate_subset_layout_overlap(job_folder)