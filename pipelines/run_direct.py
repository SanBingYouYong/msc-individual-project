"""
Direct Pipeline: Complete 3D Scene Generation and Optimization Pipeline

This pipeline accepts a user input scene description and orchestrates the complete
3D scene synthesis process through three main stages:
1. Scene Preparation (decomposition + 3D model generation)
2. Direct Loop (direct JSON layout planning and optimization)
3. Final scene creation and rendering

The pipeline creates and optimizes 3D scenes through iterative direct layout planning.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.append(os.getcwd())

from pydantic import BaseModel, Field
from colorama import Fore, init
from typing import Literal
from yaspin import yaspin

# Import pipeline components
from pipelines.common.prep_scene import ScenePrepInput, ScenePrepOutput, decompose_and_obtain_assets
from pipelines.direct.direct import DirectLoopInput, DirectLoopOutput, direct_loop
from pipelines.common.utils import save_pydantic_model_as_pkl

# Initialize colorama
init()


class DirectPipelineInput(BaseModel):
    """Input for the complete Direct pipeline."""
    user_request: str = Field(..., description="User request describing the desired 3D scene")
    job_folder: Optional[Path] = Field(None, description="Job folder path where the scene will be processed. If None, will be auto-generated")
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the direct loop")
    obtain_method: Literal['text-to-3d', 'retrieve-shapenet'] = Field(
        'text-to-3d',
        description="Method to obtain 3D models: 'text-to-3d' for text-to-3D generation with trellis, 'retrieve-shapenet' for text-based ShapeNet model retrieval"
    )
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")


class DirectPipelineOutput(BaseModel):
    """Output from the complete Direct pipeline."""
    user_request: str = Field(..., description="Original user request")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from scene preparation stage")
    direct_loop_output: DirectLoopOutput = Field(..., description="Output from direct loop optimization")
    job_folder: Path = Field(..., description="Path to the job folder containing all outputs")
    final_scene_path: Path = Field(..., description="Path to the final optimized Blender scene file")
    final_rendered_images: list[Path] = Field(..., description="Paths to the final rendered scene images")
    is_satisfactory: bool = Field(..., description="Whether the final scene layout was deemed satisfactory")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")


def run_direct_pipeline(input_data: DirectPipelineInput) -> DirectPipelineOutput:
    """
    Runs the complete Direct pipeline for 3D scene synthesis.
    
    Process:
    1. Scene Preparation: Decompose user request → Generate 3D models
    2. Direct Loop: Plan scene directly → Generate layout JSON → Optimize through iterations
    3. Final Output: Create optimized scene and render final images
    
    Args:
        input_data: Pipeline input configuration
        
    Returns:
        Complete pipeline output with scene files and metadata
    """
    start_time = datetime.now()
    
    # Generate job folder if not provided
    if input_data.job_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_data.job_folder = Path("exp") / f"direct_pipeline_{timestamp}"
    
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Direct 3D Scene Synthesis Pipeline{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.YELLOW}User Request: {input_data.user_request}{Fore.RESET}")
    print(f"{Fore.YELLOW}Job Folder: {input_data.job_folder}{Fore.RESET}")
    print(f"{Fore.YELLOW}Provider: {input_data.provider}, Model: {input_data.model}{Fore.RESET}")
    print()

    # Ensure job folder exists
    input_data.job_folder.mkdir(parents=True, exist_ok=True)

    # Stage 1: Scene Preparation (Decomposition + 3D Model Generation)
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    print(f"{Fore.BLUE}Stage 1: Scene Preparation{Fore.RESET}")
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    
    prep_folder = input_data.job_folder / "prep"
    scene_prep_input = ScenePrepInput(
        user_request=input_data.user_request,
        job_folder=prep_folder,
        obtain_method=input_data.obtain_method,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    
    print("Running scene preparation (decomposition + 3D model generation)...")
    scene_prep_output = decompose_and_obtain_assets(scene_prep_input)
    
    print(f"{Fore.GREEN}✔ Scene preparation completed successfully{Fore.RESET}")
    print(f"{Fore.GREEN}  - Decomposed into {len(scene_prep_output.decomposer_output.assets)} assets{Fore.RESET}")
    print(f"{Fore.GREEN}  - Generated {len(scene_prep_output.obtained_asset_files)} 3D models{Fore.RESET}")
    print()

    # Stage 2: Direct Loop (Layout Planning + Optimization)
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    print(f"{Fore.BLUE}Stage 2: Direct Loop Optimization{Fore.RESET}")
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    
    direct_folder = input_data.job_folder / "direct"
    direct_input = DirectLoopInput(
        user_request=input_data.user_request,
        decomposer_output=scene_prep_output.decomposer_output,
        scene_prep_output=scene_prep_output,
        job_folder=direct_folder,
        max_update_iterations=input_data.max_update_iterations,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    
    print("Running direct loop optimization...")
    direct_output = direct_loop(direct_input)
    
    print(f"{Fore.GREEN}✔ Direct loop optimization completed{Fore.RESET}")
    print(f"{Fore.GREEN}  - Layout satisfactory: {direct_output.is_satisfactory}{Fore.RESET}")
    print(f"{Fore.GREEN}  - Final scene: {direct_output.final_scene_path}{Fore.RESET}")
    print()

    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Stage 3: Create Final Pipeline Output
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    print(f"{Fore.BLUE}Stage 3: Final Output Generation{Fore.RESET}")
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    
    pipeline_output = DirectPipelineOutput(
        user_request=input_data.user_request,
        scene_prep_output=scene_prep_output,
        direct_loop_output=direct_output,
        job_folder=input_data.job_folder,
        final_scene_path=direct_output.final_scene_path,
        final_rendered_images=direct_output.final_rendered_images,
        is_satisfactory=direct_output.is_satisfactory,
        processing_time_seconds=processing_time
    )
    
    # Save complete pipeline output
    final_output_path = input_data.job_folder / "direct_pipeline_output.pkl"
    save_pydantic_model_as_pkl(pipeline_output, final_output_path)
    
    print(f"{Fore.GREEN}✔ Pipeline output saved to: {final_output_path}{Fore.RESET}")
    print()

    # Final Summary
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Pipeline Completed Successfully!{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}Processing Time: {processing_time:.2f} seconds{Fore.RESET}")
    print(f"{Fore.GREEN}Job Folder: {input_data.job_folder}{Fore.RESET}")
    print(f"{Fore.GREEN}Final Scene: {direct_output.final_scene_path}{Fore.RESET}")
    print(f"{Fore.GREEN}Final Renders: {len(direct_output.final_rendered_images)} images{Fore.RESET}")
    print(f"{Fore.GREEN}Layout Satisfactory: {direct_output.is_satisfactory}{Fore.RESET}")
    
    # Print asset summary
    assets = scene_prep_output.decomposer_output.assets
    print(f"{Fore.YELLOW}Generated Assets:{Fore.RESET}")
    for asset in assets:
        print(f"  • {asset.name}: {asset.description}")
    
    return pipeline_output


def main():
    """Example usage of the Direct pipeline."""
    # Example scene descriptions
    example_scenes = [
        "A cozy living room with a fireplace, a sofa, and a coffee table",
        "Three boxes of different sizes, stacked on top of each other",
        "A modern kitchen with an island, refrigerator, and dining table",
        "A bedroom with a bed, nightstand, dresser, and floor lamp",
        "A garden scene with trees, flowers, a bench, and a fountain",
        "A park scene where there is a tree, a bench and a street lamp."
    ]
    
    # Use command line argument if provided, otherwise use default
    if len(sys.argv) > 1:
        user_request = sys.argv[1]
    else:
        user_request = example_scenes[-1]  # Default to cozy living room
    
    print(f"{Fore.MAGENTA}Running Direct Pipeline with scene: '{user_request}'{Fore.RESET}")
    print()
    
    # Create pipeline input
    pipeline_input = DirectPipelineInput(
        user_request=user_request,
        max_update_iterations=3,
        obtain_method='text-to-3d',  # Change to 'retrieve-shapenet' for ShapeNet retrieval
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    
    try:
        # Run the pipeline
        output = run_direct_pipeline(pipeline_input)
        
        print(f"\n{Fore.CYAN}Pipeline completed successfully!{Fore.RESET}")
        print(f"{Fore.CYAN}Check the results in: {output.job_folder}{Fore.RESET}")
        
    except Exception as e:
        print(f"\n{Fore.RED}Pipeline failed with error: {str(e)}{Fore.RESET}")
        raise


if __name__ == "__main__":
    main()
