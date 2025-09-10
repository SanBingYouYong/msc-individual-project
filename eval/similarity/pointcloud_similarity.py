'''
To evaluate ulip2 similarity, we need to convert scenes to OBJ files, for both gt and generated scenes. 
For gt scenes, we follow the layout and find assets with object JID under 3D-future-models
For generated scenes, we enter the saved blender scene, export all objects as OBJ.
Maybe we need to join all objects first then export. This will create no material as the ulip2_encoder accepts only OBJ file not glb files, but it could be extended, but this is not an priority
'''
import os
import sys
sys.path.append(os.getcwd())
import json
from pathlib import Path
import bpy
import mathutils
import math
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from pydantic import BaseModel, Field
import numpy as np

from eval.utils import normalize_scene, rotate_and_normalize_scene
from eval.similarity.ulip2.query import compare_obj_files

'''
eval folder: metadata.json: 
{
  "dataset_path": "dataset/is_bedroom_10",
  "exp_on_dataset_path": "exp/is_bedroom_10",
  "processed_scenes": [
    "0380a4ca-bd26-4f49-9a5e-a00765aab879_MasterBedroom-35489",
    "65653285-2397-4f86-8012-17ab10f7b67a_SecondBedroom-34345",
    "bbd4d848-5cf2-4680-b508-bc13325d27b4_MasterBedroom-13224",
    "1cc33f5b-a9f0-4f6c-a859-25ccf28ddfab_MasterBedroom-275",
    "0e49912b-d9f3-4f1a-93e2-0245e6fb67c1_SecondBedroom-10798",
    "81461ff0-4f44-44df-9a8e-bb81a1c032ca_MasterBedroom-49492",
    "79597940-47fb-4944-92ed-fd7dc9da0762_MasterBedroom-9858",
    "195f3e27-09e0-4c6a-bb7b-795544e7e350_MasterBedroom-54633",
    "bf4d05fc-826d-4a7b-a938-373e42688dda_MasterBedroom-2080",
    "44903efe-45ea-42a0-954e-bacb917bd7dc_MasterBedroom-573"
  ],
  "total_scenes": 10
}
'''

def process_gt_scene(scene_id: str, dataset_path: str, eval_folder: str):
    '''
    given a scene id (from processed_scenes), read dataset_path/dataset/<scene_id>.json's 'normalized_layout_bboxes' to get gt layout, read 'object_jids' to locate assets under dataset/3D-FUTURE-model/<jid>/raw_model.obj; imports and put objects into the right place, then select all, join, normalize to unit cube, rotate 90 on x axis, export as obj without material as eval_folder/<scene_id>-scene_mesh-gt.obj
    '''
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Read scene data
    scene_json_path = Path(dataset_path) / "dataset" / f"{scene_id}.json"
    if not scene_json_path.exists():
        print(f"Scene file not found: {scene_json_path}")
        return
    
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    
    normalized_layout_bboxes = scene_data.get('normalized_layout_bboxes', {})
    object_jids = scene_data.get('object_jids', [])
    
    # Extract objects list from normalized_layout_bboxes
    bbox_objects = normalized_layout_bboxes.get('objects', [])
    
    if len(bbox_objects) != len(object_jids):
        print(f"Mismatch between bboxes ({len(bbox_objects)}) and jids ({len(object_jids)})")
        return
    
    # Import and place objects
    for bbox_obj, jid in zip(bbox_objects, object_jids):
        # NOTE: cannot use dataset_path here as it would be dataset/is_bedroom_10/3D... which is not right
        obj_path = Path('dataset') / "3D-FUTURE-model" / jid / "raw_model.obj"
        if not obj_path.exists():
            print(f"Object file not found: {obj_path}")
            continue
        
        # Import object
        bpy.ops.wm.obj_import(filepath=str(obj_path))
        
        # Get the imported object (active object after import)
        imported_obj = bpy.context.active_object
        if not imported_obj:
            print(f"Failed to import object: {obj_path}")
            continue
        
        # Extract transformation parameters
        location = bbox_obj.get('location', [0, 0, 0])
        min_coords = bbox_obj.get('min', [0, 0, 0])
        max_coords = bbox_obj.get('max', [0, 0, 0])
        orientation = bbox_obj.get('orientation', [0, 0, 0])
        scale_factor = bbox_obj.get('scale_factor', [1, 1, 1])
        
        # Calculate scale from min/max coordinates
        scale = [(max_coords[i] - min_coords[i]) for i in range(3)]
        
        # Set location
        imported_obj.location = location
        
        # Set scale (combining calculated scale with scale_factor)
        final_scale = [scale[i] * scale_factor[i] for i in range(3)]
        imported_obj.scale = final_scale
        
        # Set rotation (orientation is in radians)
        imported_obj.rotation_euler = orientation
    
    # Apply rotate-and-normalize method, then join and export
    rotate_and_normalize_scene()
    
    # Select all objects and join them
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        
        # Export as OBJ
        output_path = Path(eval_folder) / f"{scene_id}-scene_mesh-gt.obj"
        bpy.ops.wm.obj_export(filepath=str(output_path), export_materials=False)

def process_generated_scene(scene_id: str, exp_folder: str, eval_folder: str, method: str):
    '''
    given a scene id (from processed_scenes), bpy open exp_folder/<scene_id>/<method: csp or direct>/final/scene.blend (skip if not exist), select all, join, normalize to unit cube, (no need to rotate), export as obj without material as eval_folder/<scene_id>-scene_mesh-<method>.obj
    '''
    blend_path = Path(exp_folder) / scene_id / method / "final" / "scene.blend"
    if not blend_path.exists():
        print(f"Blend file not found: {blend_path}")
        return
    
    # 1. Open the blend file
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    
    # 2. Export the whole scene as OBJ without materials
    temp_obj_path = Path(eval_folder) / f"_{scene_id}_{method}.obj"
    bpy.ops.wm.obj_export(filepath=str(temp_obj_path), export_materials=False)
    
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 3. Import the OBJ file
    bpy.ops.wm.obj_import(filepath=str(temp_obj_path))
    
    # 4. Join all imported objects
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        
        # 5. Call normalize_scene
        normalize_scene()
        
        # 6. Export again as the final OBJ
        output_path = Path(eval_folder) / f"{scene_id}-scene_mesh-{method}.obj"
        bpy.ops.wm.obj_export(filepath=str(output_path), export_materials=False)
    
    # Clean up temporary file
    if temp_obj_path.exists():
        temp_obj_path.unlink()

'''
Now we loop through all scenes and export OBJ files first.
'''

def process_all(eval_folder: str):
    '''
    Given a job folder, e.g., eval/evals/is_bedroom_10, it reads metadata.json to get dataset_path and exp_on_dataset_path, then for each scene in processed_scenes, it processes gt scene and generated scenes (both csp and direct if exist) to export OBJ files.
    Depends on running eval/prep.py
    '''
    metadata_path = Path(eval_folder) / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    dataset_path = metadata.get('dataset_path')
    exp_on_dataset_path = metadata.get('exp_on_dataset_path')
    processed_scenes = metadata.get('processed_scenes', [])
    
    if not dataset_path or not exp_on_dataset_path:
        print("Missing dataset_path or exp_on_dataset_path in metadata")
        return
    
    print(f"Processing {len(processed_scenes)} scenes...")
    
    for scene_id in tqdm(processed_scenes, desc="Processing scenes"):
        print(f"\nProcessing scene: {scene_id}")
        
        # Process ground truth scene
        try:
            process_gt_scene(scene_id, dataset_path, eval_folder)
            print(f"  ✓ GT scene processed")
        except Exception as e:
            print(f"  ✗ Error processing GT scene: {e}")
        
        # Process generated scenes (both csp and direct methods)
        for method in ['csp', 'direct']:
            try:
                process_generated_scene(scene_id, exp_on_dataset_path, eval_folder, method)
                print(f"  ✓ {method} scene processed")
            except Exception as e:
                print(f"  ✗ Error processing {method} scene: {e}")

'''
Uses the compare_obj_files function to loop through all scenes and compare <scene_id>-scene_mesh-gt.obj and <scene_id>-scene_mesh-<method: csp or direct>.obj, saves the results in eval_folder/similarity_ulip2.json
'''
def ulip2_all(eval_folder: str):
    metadata_path = Path(eval_folder) / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    processed_scenes = metadata.get('processed_scenes', [])
    
    if not processed_scenes:
        print("No processed scenes found in metadata")
        return
    
    similarity_results = {}
    
    for scene_id in tqdm(processed_scenes, desc="Calculating ULIP2 similarities"):
        similarity_results[scene_id] = {}
        
        gt_obj_path = Path(eval_folder) / f"{scene_id}-scene_mesh-gt.obj"
        
        for method in ['csp', 'direct']:
            gen_obj_path = Path(eval_folder) / f"{scene_id}-scene_mesh-{method}.obj" # NOTE: sometimes the method here is fully named as generated_layout_csp or generated_layout_direct in prep.py, and in evals/*/*-layout.json but often after and here we use short names, usually in filenames
            
            if not gt_obj_path.exists():
                print(f"GT OBJ file not found: {gt_obj_path}")
                continue
            if not gen_obj_path.exists():
                # print(f"Generated OBJ file not found: {gen_obj_path}")  # this is too normal
                continue
            
            try:
                similarity_score = compare_obj_files(gt_obj_path, gen_obj_path)
                similarity_results[scene_id][method] = similarity_score
                # print(f"  ✓ {scene_id} [{method}] similarity: {similarity_score}")
            except Exception as e:
                print(f"  ✗ Error comparing {scene_id} [{method}]: {e}")
    
    # Calculate method averages
    method_scores = {'csp': [], 'direct': []}
    for scene_id, methods in similarity_results.items():
        for method in ['csp', 'direct']:
            score = methods.get(method)
            if score is not None:
                method_scores[method].append(score)

    method_averages = {}
    for method in ['csp', 'direct']:
        scores = method_scores[method]
        if scores:
            method_averages[method] = float(np.mean(scores))
        else:
            method_averages[method] = None

    similarity_results['method_averages'] = method_averages
    # Save results to JSON
    output_json_path = Path(eval_folder) / "similarity_ulip2.json"
    with open(output_json_path, 'w') as f:
        json.dump(similarity_results, f, indent=4)
    
    print(f"Similarity results saved to {output_json_path}")
    print("Method averages:", method_averages)


if __name__ == "__main__":
    job_folder = Path("eval/evals/is_bedroom_100r33")
    dataset_path = Path("dataset/is_bedroom_100")
    exp_on_dataset_path = Path("exp/is_bedroom_100")
    
    # process_gt_scene(
    #     scene_id="0e49912b-d9f3-4f1a-93e2-0245e6fb67c1_SecondBedroom-10798",
    #     dataset_path=dataset_path,
    #     eval_folder=job_folder
    # )
    # process_generated_scene(
    #     scene_id="0e49912b-d9f3-4f1a-93e2-0245e6fb67c1_SecondBedroom-10798",
    #     exp_folder=exp_on_dataset_path,
    #     eval_folder=job_folder,
    #     method="direct"
    # )
    
    process_all(str(job_folder))

    ulip2_all(str(job_folder))
    
