import os
import sys
sys.path.append(os.getcwd())
import json
from pathlib import Path
from typing import List
import shutil

def gather_scene_descriptions(dataset_path: Path = None) -> dict[str, str]:
    """
    Given a created subset path, read all dataset/*.json files and read their scene_description field and return as a dict of scene_id to scene_description.
    """
    if dataset_path is None:
        dataset_path = Path("dataset/InstructScene/threed_front_bedroom/")
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    json_files = list((dataset_path / 'dataset').glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path / 'dataset'}")
    
    scene_descriptions = {}
    for json_file in json_files:
        with json_file.open('r') as f:
            data = json.load(f)
            scene_id = json_file.stem
            description = data.get('scene_description', '').strip()
            if description:
                scene_descriptions[scene_id] = description
            else:
                print(f"Warning: No scene_description found in {json_file}")

    return scene_descriptions

def fix_subset_bbox_files(subset_path: Path, 
                         source_dataset_dir: Path = None) -> None:
    """
    Fix an existing subset by copying missing blender_layout_gt.obj files.
    Reads all dataset/*.json files in the subset and copies corresponding OBJ files 
    from the source dataset to the bbox/ directory.
    
    Args:
        subset_path: Path to the subset directory (should contain dataset/ folder)
        source_dataset_dir: Path to the original dataset directory
    """
    if source_dataset_dir is None:
        source_dataset_dir = Path("dataset/InstructScene/threed_front_bedroom/")
    
    if not subset_path.exists():
        raise ValueError(f"Subset path {subset_path} does not exist")
    
    dataset_dir = subset_path / 'dataset'
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    
    # Create bbox directory if it doesn't exist
    bbox_dir = subset_path / 'bbox'
    bbox_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/ensured bbox directory at {bbox_dir}")
    
    # Find all JSON files in the dataset directory
    json_files = list(dataset_dir.glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_dir}")
    
    print(f"Found {len(json_files)} JSON files in subset")
    
    copied_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        scene_id = json_file.stem
        
        # Find corresponding OBJ file in source dataset
        source_obj_path = source_dataset_dir / scene_id / 'blender_layout_gt.obj'
        target_obj_path = bbox_dir / f'{scene_id}.obj'
        
        if source_obj_path.exists():
            if not target_obj_path.exists():
                shutil.copy(source_obj_path, target_obj_path)
                copied_count += 1
                print(f"Copied {source_obj_path} -> {target_obj_path}")
            else:
                print(f"OBJ file already exists: {target_obj_path}")
                skipped_count += 1
        else:
            print(f"Warning: Source OBJ file not found: {source_obj_path}")
    
    print(f"Completed: {copied_count} files copied, {skipped_count} files already existed")
    
    # Update metadata if it exists
    metadata_path = subset_path / 'metadata.json'
    if metadata_path.exists():
        with metadata_path.open('r') as f:
            metadata = json.load(f)
        
        actual_copied_obj_files = len(list(bbox_dir.glob('*.obj')))
        metadata['actual_copied_obj_files'] = actual_copied_obj_files
        
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Updated metadata: {actual_copied_obj_files} total OBJ files")

if __name__ == '__main__':
    subset_path = Path("dataset/is_bedroom_3/")
    source_dataset_dir = Path("dataset/InstructScene/threed_front_bedroom/")
    fix_subset_bbox_files(subset_path, source_dataset_dir)
