'''
This file offers a method to create a subset of the InstructScene/threed_front_bedroom dataset
given a target count of scenes. It randomly samples scenes from the full dataset and copies only 
the scene_with_bbox.json to the new subset directory.
'''

import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import numpy as np
import shutil
import json

def create_subset(subset_length: int, 
                  subset_dir: Path,
                  source_dataset_dir: Path=None,
                  skip_source_subdirs: list=None):
    '''
    Create a subset of the InstructScene/threed_front_bedroom dataset.
    Args:
        subset_length: Number of scenes to include in the subset.
        subset_dir: Path to the output subset directory.
        source_dataset_dir: Path to the full InstructScene/threed_front_bedroom dataset.
                            If None, defaults to 'dataset/InstructScene/threed_front_bedroom/'.
        skip_source_subdirs: List of subdirectory names to skip from the source dataset.
                             If None, defaults to ["_test_blender_rendered_scene_256_topdown", "_train_blender_rendered_scene_256_topdown"].
    Returns:
        None. The function creates the subset directory with copied files.
    '''
    if source_dataset_dir is None:
        source_dataset_dir = Path('dataset/InstructScene/threed_front_bedroom/')
    if skip_source_subdirs is None:
        skip_source_subdirs = ["_test_blender_rendered_scene_256_topdown", "_train_blender_rendered_scene_256_topdown"]
    
    all_subdirs = [d for d in source_dataset_dir.iterdir() if d.is_dir() and d.name not in skip_source_subdirs]
    print(f"Found {len(all_subdirs)} subdirectories in {source_dataset_dir}, skipping {skip_source_subdirs}.")
    
    if subset_length > len(all_subdirs):
        raise ValueError(f"Requested subset length {subset_length} exceeds the total number of available subdirectories ({len(all_subdirs)}).")
    
    subset_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = subset_dir / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir = subset_dir / 'bbox'
    bbox_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created subset directory at {subset_dir}, dataset directory at {dataset_dir}, and bbox directory at {bbox_dir}.")

    # Create metadata JSON file
    metadata = {
        "source_dataset_dir": str(source_dataset_dir),
        "subset_length": subset_length,
        "skip_source_subdirs": skip_source_subdirs
    }
    metadata_path = subset_dir / 'metadata.json'
    with metadata_path.open('w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    
    sampled_subdirs = np.random.choice(all_subdirs, size=subset_length, replace=False)
    print(f"Randomly sampled {len(sampled_subdirs)} subdirectories for the subset.")
    
    for subdir in sampled_subdirs:
        # Copy scene_with_bbox.json
        source_json_path = subdir / 'scene_with_bbox.json'
        target_json_path = dataset_dir / f'{subdir.name}.json'
        if source_json_path.exists():
            shutil.copy(source_json_path, target_json_path)
        else:
            print(f"Warning: {source_json_path} does not exist, skipping.")
        
        # Copy blender_layout_gt.obj
        source_obj_path = subdir / 'blender_layout_gt.obj'
        target_obj_path = bbox_dir / f'{subdir.name}.obj'
        if source_obj_path.exists():
            shutil.copy(source_obj_path, target_obj_path)
        else:
            print(f"Warning: {source_obj_path} does not exist, skipping.")
        
    # record again the actual number of scenes copied
    actual_copied_scenes = len(list(dataset_dir.glob('*.json')))
    actual_copied_obj_files = len(list(bbox_dir.glob('*.obj')))
    metadata['actual_copied_scenes'] = actual_copied_scenes
    metadata['actual_copied_obj_files'] = actual_copied_obj_files
    with metadata_path.open('w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    
    print(f"Created metadata file at {metadata_path}.")

def create_subset_with_object_counts(subset_dir: Path,
                                     object_counts_and_number_of_scenes: dict,
                                     source_dataset_dir: Path=None,
                                     skip_source_subdirs: list=None):
    '''
    Creates a subset too, but adheres to specified object counts and number of scenes for each count.
    It also specifies in metadata.json the correspondence of each scene to its object count. e.g. counts: <count>: <list of scene ids that has that count of objects>
    Args:
        subset_dir: Path to the output subset directory.
        object_counts_and_number_of_scenes: Dict mapping object counts to number of scenes, e.g. {3: 50, 5: 30, 7: 20}.
        source_dataset_dir: Path to the full InstructScene/threed_front_bedroom dataset.
                            If None, defaults to 'dataset/InstructScene/threed_front_bedroom/'.
        skip_source_subdirs: List of subdirectory names to skip from the source dataset.
                             If None, defaults to ["_test_blender_rendered_scene_256_topdown", "_train_blender_rendered_scene_256_topdown"].
    Returns:
        None. The function creates the subset directory with copied files.
    '''
    if source_dataset_dir is None:
        source_dataset_dir = Path('dataset/InstructScene/threed_front_bedroom/')
    if skip_source_subdirs is None:
        skip_source_subdirs = ["_test_blender_rendered_scene_256_topdown", "_train_blender_rendered_scene_256_topdown"]
    
    all_subdirs = [d for d in source_dataset_dir.iterdir() if d.is_dir() and d.name not in skip_source_subdirs]
    print(f"Found {len(all_subdirs)} subdirectories in {source_dataset_dir}, skipping {skip_source_subdirs}.")
    
    # Create subset directory structure
    subset_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = subset_dir / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir = subset_dir / 'bbox'
    bbox_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created subset directory at {subset_dir}, dataset directory at {dataset_dir}, and bbox directory at {bbox_dir}.")

    # Group scenes by object count
    scenes_by_count = {}
    for subdir in all_subdirs:
        # Read scene_with_bbox.json and count length of field 'object_descriptions'
        json_file = subdir / "scene_with_bbox.json"
        if json_file.is_file():
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                object_count = len(data.get("object_descriptions", []))
                
                if object_count not in scenes_by_count:
                    scenes_by_count[object_count] = []
                scenes_by_count[object_count].append(subdir)
    
    print(f"Grouped scenes by object count. Found counts: {sorted(scenes_by_count.keys())}")
    
    # Sample scenes according to specified counts
    selected_scenes = []
    counts_metadata = {}
    
    # First, check if all requirements can be met
    for target_count, num_scenes in object_counts_and_number_of_scenes.items():
        if target_count not in scenes_by_count:
            distribution_str = "{" + ", ".join([f"{count}: {len(scenes)}" for count, scenes in sorted(scenes_by_count.items())]) + "}"
            raise ValueError(f"No scenes found with {target_count} objects. Available distribution: {distribution_str}")
        
        available_scenes = scenes_by_count[target_count]
        if len(available_scenes) < num_scenes:
            distribution_str = "{" + ", ".join([f"{count}: {len(scenes)}" for count, scenes in sorted(scenes_by_count.items())]) + "}"
            raise ValueError(f"Only {len(available_scenes)} scenes available with {target_count} objects, but {num_scenes} requested. Available distribution: {distribution_str}")
    
    # If all requirements can be met, proceed with sampling
    for target_count, num_scenes in object_counts_and_number_of_scenes.items():
        available_scenes = scenes_by_count[target_count]
        sampled_scenes = np.random.choice(available_scenes, size=num_scenes, replace=False)
        
        selected_scenes.extend(sampled_scenes)
        counts_metadata[target_count] = [scene.name for scene in sampled_scenes]
        print(f"Selected {len(sampled_scenes)} scenes with {target_count} objects.")
    
    print(f"Total selected scenes: {len(selected_scenes)}")
    
    # Copy files for selected scenes
    for subdir in selected_scenes:
        # Copy scene_with_bbox.json
        source_json_path = subdir / 'scene_with_bbox.json'
        target_json_path = dataset_dir / f'{subdir.name}.json'
        if source_json_path.exists():
            shutil.copy(source_json_path, target_json_path)
        else:
            print(f"Warning: {source_json_path} does not exist, skipping.")
        
        # Copy blender_layout_gt.obj
        source_obj_path = subdir / 'blender_layout_gt.obj'
        target_obj_path = bbox_dir / f'{subdir.name}.obj'
        if source_obj_path.exists():
            shutil.copy(source_obj_path, target_obj_path)
        else:
            print(f"Warning: {source_obj_path} does not exist, skipping.")
    
    # Create metadata JSON file
    actual_copied_scenes = len(list(dataset_dir.glob('*.json')))
    actual_copied_obj_files = len(list(bbox_dir.glob('*.obj')))
    
    metadata = {
        "source_dataset_dir": str(source_dataset_dir),
        "object_counts_and_number_of_scenes": object_counts_and_number_of_scenes,
        "skip_source_subdirs": skip_source_subdirs,
        "counts": counts_metadata,
        "actual_copied_scenes": actual_copied_scenes,
        "actual_copied_obj_files": actual_copied_obj_files
    }
    
    metadata_path = subset_dir / 'metadata.json'
    with metadata_path.open('w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    
    print(f"Created metadata file at {metadata_path}.")


if __name__ == '__main__':
    # subset_length = 3  # specify the desired number of scenes in the subset
    # subset_dir = Path('dataset/is_bedroom_3')  # specify the output subset directory
    # create_subset(subset_length, subset_dir)
    
    subset_dir = Path('dataset/is_bedroom-30-4_6')  # specify the output subset directory
    create_subset_with_object_counts(
        subset_dir,
        object_counts_and_number_of_scenes={4: 10, 6: 10}
    )
