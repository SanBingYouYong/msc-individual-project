'''
This file obtains all 3D models for a batch of decomposed scenes, either through text-to-3D or retrieve-shapenet. 
- in future it also could be retrieving 3D-future-models assets directly
'''
import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict
from tqdm import tqdm
import json

from agents.decomposer import DecomposedAsset, DecomposerOutput
from pipelines.common.prep_scene import _obtain_models_text_to_3d, _obtain_models_shapenet_retrieval, ScenePrepOutput
from pipelines.common.utils import save_pydantic_model_as_pkl, load_pydantic_model_from_pkl

class BatchObtainInput(BaseModel):
    """Input for batch obtaining 3D models for decomposed scenes."""
    b1_folder: Path = Field(..., description="Path to the experiment folder containing scene subdirectories and decomposed assets")
    obtain_method: Literal['text-to-3d', 'retrieve-shapenet'] = Field(
        'text-to-3d',
        description="Method to obtain 3D models: 'text-to-3d' for text-to-3D generation with trellis, 'retrieve-shapenet' for text-based ShapeNet model retrieval"
    )

def batch_obtain(input_data: BatchObtainInput) -> None:
    """
    Based on b1, for each scene, it reads prep/assets.json to get the asset list and descriptions, then obtains 3D models for all assets using the specified method.
    """
    if not input_data.b1_folder.exists():
        raise ValueError(f"Experiment folder {input_data.b1_folder} does not exist")
    
    scene_dirs = [d for d in input_data.b1_folder.iterdir() if d.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No scene subdirectories found in {input_data.b1_folder}")
    
    for scene_dir in tqdm(scene_dirs, desc="Obtaining 3D models for decomposed scenes"):
        prep_dir = scene_dir / "prep"
        if not prep_dir.exists():
            raise RuntimeError(f"Prep directory {prep_dir} does not exist, please run b0_initialize_all.py first")
        assets_json_path = prep_dir / "assets.json"
        if not assets_json_path.exists():
            print(f"Warning: No assets.json found in {prep_dir}, skipping...")
            continue
        
        with assets_json_path.open('r') as f:
            try:
                asset_list = json.load(f)
                if not isinstance(asset_list, list) or not all(isinstance(item, dict) and 'name' in item and 'description' in item and 'location' in item for item in asset_list):
                    raise ValueError
            except ValueError:
                print(f"Warning: Invalid assets.json format in {assets_json_path}, expected a list of dicts with 'name', 'description', and 'location', skipping...")
                continue
        
        obtained_models_dir = prep_dir / "models"
        obtained_models_dir.mkdir(exist_ok=True)
        
        if input_data.obtain_method == 'text-to-3d':
            obtained_files = _obtain_models_text_to_3d(
                asset_list=[DecomposedAsset(**asset) for asset in asset_list],
                obtained_models_dir=obtained_models_dir
            )
        elif input_data.obtain_method == 'retrieve-shapenet':
            obtained_files = _obtain_models_shapenet_retrieval(
                asset_list=[DecomposedAsset(**asset) for asset in asset_list],
                obtained_models_dir=obtained_models_dir
            )
        else:
            raise ValueError(f"Unknown obtain method: {input_data.obtain_method}")
        
        # Reads decomposer output pkl and save scene prep output pkl
        decomposer_output_path = prep_dir / "decomposer_output.pkl"
        if not decomposer_output_path.exists():
            raise RuntimeError(f"Decomposer output file {decomposer_output_path} does not exist, please run b1_decompose_all.py first")
        decomposer_output = load_pydantic_model_from_pkl(DecomposerOutput, decomposer_output_path)
        scene_prep_output = ScenePrepOutput(
            decomposer_output=decomposer_output,
            obtained_asset_files=obtained_files
        )
        scene_prep_output_path = prep_dir / "scene_prep_output.pkl"
        save_pydantic_model_as_pkl(scene_prep_output, scene_prep_output_path)
        # rename decomposer_output.pkl to start with _
        new_decomposer_output_path = prep_dir / "_decomposer_output.pkl"
        decomposer_output_path.rename(new_decomposer_output_path)
    
    print(f"Completed obtaining 3D models for all scenes in {input_data.b1_folder} using method '{input_data.obtain_method}'.")


if __name__ == '__main__':
    batch_obtain(BatchObtainInput(
        b1_folder=Path("exp/is_bedroom_100/"),
        obtain_method='retrieve-shapenet',
    ))

