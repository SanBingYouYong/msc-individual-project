# calls decomposer to get asset list and feeds it to either text-to-3d generation or shapenet retrieval pipeline
import os
from pathlib import Path
from typing import List, Dict
import sys
import os
import json
import shutil
sys.path.append(os.getcwd())

from agents.decomposer import decompose, DecomposerInput, DecomposerOutput, DecomposedAsset
from tools.text_to_3d import GenerationRequest, generate_all_models
from tools.retrieve_shapenet import retrieve_all_models, RetrievalRequest
from pipelines.common.utils import save_pydantic_model_as_pkl

from pydantic import BaseModel, Field
from yaspin import yaspin
from typing import Optional, Dict, Literal

class ScenePrepInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    job_folder: Path = Field(..., description="Job folder path where the scene preparation will be processed; expected: job-folder/prep")
    obtain_method: Literal['text-to-3d', 'retrieve-shapenet'] = Field(
        'text-to-3d',
        description="Method to obtain 3D models: 'text-to-3d' for text-to-3D generation with trellis, 'retrieve-shapenet' for text-based ShapeNet model retrieval"
    )
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")

class ScenePrepOutput(BaseModel):
    decomposer_output: DecomposerOutput = Field(..., description="Output from the decomposer containing asset list and descriptions")
    obtained_asset_files: Dict[str, Path] = Field(..., description="Mapping of asset names to their generated model file paths")

def _obtain_models_text_to_3d(asset_list: List[DecomposedAsset], obtained_models_dir: Path) -> Dict[str, Path]:
    """
    Generate 3D models using text-to-3D generation for a list of assets.
    
    Args:
        asset_list: List of assets with descriptions
        obtained_models_dir: Directory to save the generated models
        
    Returns:
        Dict mapping asset names to their generated model file paths
    """
    tasks = [
        GenerationRequest(description=asset.description, glb_path=obtained_models_dir / f"{asset.name}.glb")
        for asset in asset_list
    ]
    print(f"Generating {len(tasks)} models using text-to-3D based on decomposed assets...")
    generate_all_models(tasks)
    return {asset.name: task.glb_path for asset, task in zip(asset_list, tasks)}

def _obtain_models_shapenet_retrieval(asset_list: List[DecomposedAsset], obtained_models_dir: Path) -> Dict[str, Path]:
    """
    Retrieve 3D models from ShapeNet database for a list of assets.
    
    Args:
        asset_list: List of assets with descriptions
        obtained_models_dir: Directory to save the retrieved models
        
    Returns:
        Dict mapping asset names to their retrieved model file paths
    """
    tasks = [
        RetrievalRequest(query=asset.description, glb_path=obtained_models_dir / f"{asset.name}.glb")
        for asset in asset_list
    ]
    print(f"Retrieving {len(tasks)} models from ShapeNet based on decomposed assets...")
    retrieve_all_models(tasks)
    return {asset.name: task.glb_path for asset, task in zip(asset_list, tasks)}

def decompose_and_obtain_assets(scene_prep_input: ScenePrepInput) -> ScenePrepOutput:
    # Ensure the job folder exists
    scene_prep_input.job_folder.mkdir(parents=True, exist_ok=True)
    
    # Create a stage subfolder for temporary storage
    stage_folder = scene_prep_input.job_folder / "stage"
    stage_folder.mkdir(parents=True, exist_ok=True)
    
    # Create generated_models subfolder
    obtained_models_dir = scene_prep_input.job_folder / "models"
    obtained_models_dir.mkdir(parents=True, exist_ok=True)

    # Decompose the user request into a list of assets
    decomposer_input = DecomposerInput(
        user_request=scene_prep_input.user_request,
        provider=scene_prep_input.provider,
        model=scene_prep_input.model,
        temperature=scene_prep_input.temperature
    )
    with yaspin(text="Decomposing user request...", color="green", timer=True) as spinner:
        decomposer_output = decompose(decomposer_input)
        spinner.ok("✔")
    asset_list = decomposer_output.assets

    # Temporarily save decomposer output in stage folder
    decomposer_output_stage_path = stage_folder / "decomposer_output.pkl"
    save_pydantic_model_as_pkl(decomposer_output, decomposer_output_stage_path)

    # Save assets to assets.json
    assets_json_path = scene_prep_input.job_folder / "assets.json"
    with open(assets_json_path, "w") as f:
        json.dump([asset.model_dump() for asset in asset_list], f, indent=4)

    # Save formatted_assets_str to assets.md
    assets_md_path = scene_prep_input.job_folder / "assets.md"
    with open(assets_md_path, "w") as f:
        f.write(decomposer_output.formatted_asset_str)

    # Obtain models for each asset based on the specified method
    if scene_prep_input.obtain_method == 'text-to-3d':
        obtained_files = _obtain_models_text_to_3d(asset_list, obtained_models_dir)
    elif scene_prep_input.obtain_method == 'retrieve-shapenet':
        obtained_files = _obtain_models_shapenet_retrieval(asset_list, obtained_models_dir)
    else:
        raise ValueError(f"Unknown obtain_method: {scene_prep_input.obtain_method}. Must be 'text-to-3d' or 'retrieve-shapenet'.")

    scene_prep_output = ScenePrepOutput(
        decomposer_output=decomposer_output,
        obtained_asset_files=obtained_files
    )
    
    # Save scene prep output as pkl file (contains decomposer output)
    scene_prep_output_path = scene_prep_input.job_folder / "scene_prep_output.pkl"
    save_pydantic_model_as_pkl(scene_prep_output, scene_prep_output_path)

    # Remove the stage folder since we no longer need the temporary decomposer output
    shutil.rmtree(stage_folder)

    return scene_prep_output


if __name__ == '__main__':
    user_request = "A cozy living room with a fireplace, a sofa, and a coffee table."
    job_folder = Path("exp/test_prep_scene-generate1")

    # Example 1: Using text-to-3D generation (default)
    scene_prep_input = ScenePrepInput(
        user_request=user_request, 
        job_folder=job_folder,
        obtain_method='text-to-3d'
    )

    # Example 2: Using ShapeNet retrieval (uncomment to test)
    # scene_prep_input = ScenePrepInput(
    #     user_request=user_request, 
    #     job_folder=job_folder,
    #     obtain_method='retrieve-shapenet'
    # )

    # Generate/retrieve models based on the user request
    scene_prep_output = decompose_and_obtain_assets(scene_prep_input)


