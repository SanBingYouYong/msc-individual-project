'''
This pipeline calls decomposer on a batch of scenes.
'''
import os
import sys
sys.path.append(os.getcwd())

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal, Optional, Dict, List
from tqdm import tqdm
import json

from agents.decomposer_fixed_assets import DecomposerFAInput, DecomposerOutput, decompose_with_fixed_assets
from pipelines.common.utils import save_pydantic_model_as_pkl

class BatchDecomposeInput(BaseModel):
    """Input for batch decomposition of user requests."""
    b0_folder: Path = Field(..., description="Path to the experiment folder containing scene subdirectories")
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")

def batch_decompose(input_data: BatchDecomposeInput) -> None:
    """
    Decomposes user requests for all scenes in the specified experiment folder.
    
    Args:
        input_data (BatchDecomposeInput): Input parameters for batch decomposition.
    
    Returns:
        Dict[str, DecomposerOutput]: A dictionary mapping scene IDs to their decomposer outputs.
    """
    if not input_data.b0_folder.exists():
        raise ValueError(f"Experiment folder {input_data.b0_folder} does not exist")
    
    scene_dirs = [d for d in input_data.b0_folder.iterdir() if d.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No scene subdirectories found in {input_data.b0_folder}")
    
    for scene_dir in tqdm(scene_dirs, desc="Decomposing scenes"):
        prep_dir = scene_dir / "prep"
        if not prep_dir.exists():
            raise RuntimeError(f"Prep directory {prep_dir} does not exist, please run b0_initialize_all.py first")
        scene_txt_path = scene_dir / "_scene.txt"
        if not scene_txt_path.exists():
            print(f"Warning: No _scene.txt found in {scene_dir}, skipping...")
            continue
        objects_json_path = scene_dir / "_objects.json"
        if not objects_json_path.exists():
            print(f"Warning: No _objects.json found in {scene_dir}, skipping...")
            continue
        
        with scene_txt_path.open('r') as f:
            user_request = f.read().strip()
        
        with objects_json_path.open('r') as f:
            asset_descriptions = json.load(f)
        if not isinstance(asset_descriptions, list) or not all(isinstance(desc, str) for desc in asset_descriptions):
            print(f"Warning: Invalid _objects.json format in {scene_dir}, expected a list of strings, skipping...")
            continue
        
        if not user_request:
            print(f"Warning: Empty user request in {scene_txt_path}, skipping...")
            continue
        
        decomposer_input = DecomposerFAInput(
            user_request=user_request,
            asset_descriptions=asset_descriptions,
            provider=input_data.provider,
            model=input_data.model,
            temperature=input_data.temperature
        )
        
        decomposer_output = decompose_with_fixed_assets(decomposer_input)
        
        asset_list = decomposer_output.assets

        # Save decomposer output 
        decomposer_output_stage_path = prep_dir / "decomposer_output.pkl"
        save_pydantic_model_as_pkl(decomposer_output, decomposer_output_stage_path)

        # Save assets to assets.json
        assets_json_path = prep_dir / "assets.json"
        with open(assets_json_path, "w") as f:
            json.dump([asset.model_dump() for asset in asset_list], f, indent=4)

        # Save formatted_assets_str to assets.md
        assets_md_path = prep_dir / "assets.md"
        with open(assets_md_path, "w") as f:
            f.write(decomposer_output.formatted_asset_str)
        

if __name__ == '__main__':
    batch_decompose(
        BatchDecomposeInput(
            b0_folder=Path("exp/is_bedroom-30-4_6/"),
            provider="ollama",
            model="gemma3",
            temperature=0
        )
    )
