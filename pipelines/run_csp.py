"""
CSP Pipeline: Complete 3D Scene Generation and Optimization Pipeline

This pipeline accepts a user input scene description and orchestrates the complete
3D scene synthesis process through three main stages:
1. Scene Preparation (decomposition + 3D model generation)
2. CSP Inner Loop (constraint satisfaction and optimization)
3. Final scene creation and rendering

The pipeline creates and optimizes 3D scenes through iterative constraint satisfaction.
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
from pipelines.csp.csp_inner_loop import CSPInnerLoopInput, CSPInnerLoopOutput, csp_inner_loop
from pipelines.common.utils import save_pydantic_model_as_pkl

# Initialize colorama
init()


class CSPPipelineInput(BaseModel):
    """Input for the complete CSP pipeline."""
    user_request: str = Field(..., description="User request describing the desired 3D scene")
    job_folder: Optional[Path] = Field(None, description="Job folder path where the scene will be processed. If None, will be auto-generated")
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the CSP inner loop")
    obtain_method: Literal['text-to-3d', 'retrieve-shapenet'] = Field(
        'text-to-3d',
        description="Method to obtain 3D models: 'text-to-3d' for text-to-3D generation with trellis, 'retrieve-shapenet' for text-based ShapeNet model retrieval"
    )
    layout_verify_iters: int = Field(3, description="Maximum number of iterations for layout verification; this is shared between init and updates")
    code_fix_iters: int = Field(3, description="Maximum number of iterations for EACH error fixing during code execution; this is shared between init and updates")
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")


class CSPPipelineOutput(BaseModel):
    """Output from the complete CSP pipeline."""
    user_request: str = Field(..., description="Original user request")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from scene preparation stage")
    csp_inner_loop_output: CSPInnerLoopOutput = Field(..., description="Output from CSP inner loop optimization")
    job_folder: Path = Field(..., description="Path to the job folder containing all outputs")
    final_scene_path: Path = Field(..., description="Path to the final optimized Blender scene file")
    final_rendered_images: list[Path] = Field(..., description="Paths to the final rendered scene images")
    is_satisfactory: bool = Field(..., description="Whether the final scene layout was deemed satisfactory")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")


def run_csp_pipeline(input_data: CSPPipelineInput) -> CSPPipelineOutput:
    """
    Runs the complete CSP pipeline for 3D scene synthesis.
    
    Process:
    1. Scene Preparation: Decompose user request → Generate 3D models
    2. CSP Inner Loop: Plan scene → Generate layout code → Optimize through iterations
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
        input_data.job_folder = Path("exp") / f"csp_pipeline_{timestamp}"
    
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}CSP 3D Scene Synthesis Pipeline{Fore.RESET}")
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

    # Stage 2: CSP Inner Loop (Layout Planning + Optimization)
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    print(f"{Fore.BLUE}Stage 2: CSP Inner Loop Optimization{Fore.RESET}")
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    
    csp_folder = input_data.job_folder / "csp"
    csp_input = CSPInnerLoopInput(
        user_request=input_data.user_request,
        decomposer_output=scene_prep_output.decomposer_output,
        scene_prep_output=scene_prep_output,
        job_folder=csp_folder,
        max_update_iterations=input_data.max_update_iterations,
        layout_verify_iters=input_data.layout_verify_iters,
        code_fix_iters=input_data.code_fix_iters,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    
    print("Running CSP inner loop optimization...")
    csp_output = csp_inner_loop(csp_input)
    
    print(f"{Fore.GREEN}✔ CSP inner loop optimization completed{Fore.RESET}")
    print(f"{Fore.GREEN}  - Layout satisfactory: {csp_output.is_satisfactory}{Fore.RESET}")
    print(f"{Fore.GREEN}  - Final scene: {csp_output.final_scene_path}{Fore.RESET}")
    print()

    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Stage 3: Create Final Pipeline Output
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    print(f"{Fore.BLUE}Stage 3: Final Output Generation{Fore.RESET}")
    print(f"{Fore.BLUE}{'='*40}{Fore.RESET}")
    
    pipeline_output = CSPPipelineOutput(
        user_request=input_data.user_request,
        scene_prep_output=scene_prep_output,
        csp_inner_loop_output=csp_output,
        job_folder=input_data.job_folder,
        final_scene_path=csp_output.final_scene_path,
        final_rendered_images=csp_output.final_rendered_images,
        is_satisfactory=csp_output.is_satisfactory,
        processing_time_seconds=processing_time
    )
    
    # Save complete pipeline output
    final_output_path = input_data.job_folder / "csp_pipeline_output.pkl"
    save_pydantic_model_as_pkl(pipeline_output, final_output_path)
    
    print(f"{Fore.GREEN}✔ Pipeline output saved to: {final_output_path}{Fore.RESET}")
    print()

    # Final Summary
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.CYAN}Pipeline Completed Successfully!{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
    print(f"{Fore.GREEN}Processing Time: {processing_time:.2f} seconds{Fore.RESET}")
    print(f"{Fore.GREEN}Job Folder: {input_data.job_folder}{Fore.RESET}")
    print(f"{Fore.GREEN}Final Scene: {csp_output.final_scene_path}{Fore.RESET}")
    print(f"{Fore.GREEN}Final Renders: {len(csp_output.final_rendered_images)} images{Fore.RESET}")
    print(f"{Fore.GREEN}Layout Satisfactory: {csp_output.is_satisfactory}{Fore.RESET}")
    
    # Print asset summary
    assets = scene_prep_output.decomposer_output.assets
    print(f"{Fore.YELLOW}Generated Assets:{Fore.RESET}")
    for asset in assets:
        print(f"  • {asset.name}: {asset.description}")
    
    return pipeline_output


def main():
    """Example usage of the CSP pipeline."""
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
    
    print(f"{Fore.MAGENTA}Running CSP Pipeline with scene: '{user_request}'{Fore.RESET}")
    print()
    
    # Create pipeline input
    pipeline_input = CSPPipelineInput(
        user_request=user_request,
        max_update_iterations=10,
        obtain_method='text-to-3d',  # Change to 'text-to-3d' for text-to-3D generation
        provider="google",
        model="gemini-2.5-flash",
        temperature=0,
    )
    
    try:
        # Run the pipeline
        output = run_csp_pipeline(pipeline_input)
        
        print(f"\n{Fore.CYAN}Pipeline completed successfully!{Fore.RESET}")
        print(f"{Fore.CYAN}Check the results in: {output.job_folder}{Fore.RESET}")
        
    except Exception as e:
        print(f"\n{Fore.RED}Pipeline failed with error: {str(e)}{Fore.RESET}")
        raise


if __name__ == "__main__":
    main()
