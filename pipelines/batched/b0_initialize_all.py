'''
This file accepts a dataset path and initializes a managed exp folder: 
- creates subdirs for each scene and a subdir/prep folder for subsequent batched decompose and model retrieval/generation
- saves scene description under subdir/_scene.txt
- saves object descriptions under subdir/_objects.txt
'''
import os
import sys
sys.path.append(os.getcwd())

import json
from pathlib import Path
from typing import List
import shutil

def initialize_all_scenes(dataset_path: Path,
                          exp_folder: Path):
    """
    Initializes the experiment folder by creating subdirectories for each scene and saving scene and object descriptions.
    
    Args:
        dataset_path (Path): Path to the dataset directory containing scene JSON files.
        exp_folder (Path): Path to the experiment folder where subdirectories will be created.
    """
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    exp_folder.mkdir(parents=True, exist_ok=True)
    json_files = list((dataset_path / 'dataset').glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path / 'dataset'}")
    
    for json_file in json_files:
        scene_id = json_file.stem
        scene_dir = exp_folder / scene_id
        prep_dir = scene_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)
        
        with json_file.open('r') as f:
            data = json.load(f)
            scene_description = data.get('scene_description', '').strip()
            object_descriptions = data.get('object_descriptions', [])
        
        if not scene_description:
            print(f"Warning: No scene_description found in {json_file}, skipping...")
            continue
        if not object_descriptions:
            print(f"Warning: No object_descriptions found in {json_file}, skipping...")
            continue
        
        scene_txt_path = scene_dir / "_scene.txt"
        with scene_txt_path.open('w') as f:
            f.write(scene_description)
        
        objects_txt_path = scene_dir / "_objects.json"
        with objects_txt_path.open('w') as f:
            json.dump([obj_desc.strip() for obj_desc in object_descriptions], f, indent=4)
    
    print(f"Initialized experiment folder at {exp_folder} with {len(json_files)} scenes.")
    
if __name__ == '__main__':
    initialize_all_scenes(
        dataset_path=Path("dataset/is_bedroom-30-4_6/"),
        exp_folder=Path("exp/is_bedroom-30-4_6/")
    )
