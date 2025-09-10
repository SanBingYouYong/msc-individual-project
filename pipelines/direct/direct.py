import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from pipelines.direct.direct_init import DirectInitInput, DirectInitOutput, direct_init
from pipelines.direct.direct_updates import DirectUpdatesInput, DirectUpdatesOutput, direct_updates
from pipelines.common.create_and_render import CreateAndRenderInput, CreateAndRenderOutput, create_and_render
from agents.decomposer import DecomposerOutput
from pipelines.common.prep_scene import ScenePrepOutput
from pipelines.common.utils import save_pydantic_model_as_pkl

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
from colorama import Fore
import json


class DirectLoopInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    decomposer_output: DecomposerOutput = Field(..., description="Output from the decomposer containing asset list and descriptions")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from the scene preparation pipeline containing generated model files")
    job_folder: Path = Field(..., description="Path to the job folder where the scene will be processed")
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the inner loop")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")


class DirectLoopOutput(BaseModel):
    init_output: DirectInitOutput = Field(..., description="Output from the initialization phase")
    updates_output: DirectUpdatesOutput = Field(..., description="Output from the updates phase")
    final_layout_json_str: str = Field(..., description="Final layout JSON string from the complete direct loop process")
    final_layout_path: Path = Field(..., description="Path to the final layout JSON file")
    final_scene_path: Path = Field(..., description="Path to the final Blender scene file")
    final_rendered_images: List[Path] = Field(..., description="Paths to the final rendered images")
    is_satisfactory: bool = Field(..., description="Whether the final layout was deemed satisfactory")


def direct_loop(input_data: DirectLoopInput) -> DirectLoopOutput:
    """
    Complete Direct loop pipeline that combines initialization and updates sequentially.
    
    This pipeline:
    1. Runs the direct initialization (planning, scene creation, rendering)
    2. Runs the direct updates (review iterations with potential layout improvements)
    3. Returns the final output with the best available results
    
    Args:
        input_data: DirectLoopInput containing all necessary inputs for the complete loop
        
    Returns:
        DirectLoopOutput containing all outputs from both init and update phases
    """
    print(f"{Fore.CYAN}Starting Direct Loop Pipeline{Fore.RESET}")
    print(f"{Fore.CYAN}Job folder: {input_data.job_folder}{Fore.RESET}")
    
    # Phase 1: Initialization
    print(f"{Fore.BLUE}Running Direct Initialization...{Fore.RESET}")
    init_folder = input_data.job_folder / "direct_init"
    init_input = DirectInitInput(
        user_request=input_data.user_request,
        decomposer_output=input_data.decomposer_output,
        scene_prep_output=input_data.scene_prep_output,
        job_folder=init_folder,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    init_output = direct_init(init_input)
    print(f"{Fore.GREEN}✔ Direct Initialization completed successfully{Fore.RESET}")

    # Phase 2: Updates
    print(f"{Fore.BLUE}Running Direct Updates...{Fore.RESET}")
    
    updates_input = DirectUpdatesInput(
        original_task_instruction=init_output.planner_output.sent_query,
        last_layout_json_str=init_output.planner_output.scene_layout,
        last_rendered_images=init_output.scene_output.render_scene_output.rendered_images,
        decomposer_output=input_data.decomposer_output,
        scene_prep_output=input_data.scene_prep_output,
        job_folder=input_data.job_folder,
        max_iterations=input_data.max_update_iterations,
        starting_index=1,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    updates_output = direct_updates(updates_input)
    print(f"{Fore.GREEN}✔ Direct Updates completed successfully{Fore.RESET}")

    # Phase 3: Generate Final Results
    print(f"{Fore.BLUE}Generating final scene and outputs...{Fore.RESET}")
    
    # Determine which results to use: updates final_output or init results
    if updates_output.final_output is None:
        # Use initialization results
        final_layout_json_str = init_output.planner_output.scene_layout
        is_satisfactory = False
        print(f"{Fore.YELLOW}Using initialization results (no successful updates or initial results are satisfactory){Fore.RESET}")
    else:
        # Use final update results (guaranteed to be valid)
        final_layout_json_str = updates_output.final_output.reviewer_output.updated_layout_json_str
        # Check if the layout was deemed satisfactory by the last update's reviewer
        is_satisfactory = updates_output.updates[-1].reviewer_output.satisfactory if updates_output.updates else False
        print(f"{Fore.GREEN}Using final update results{Fore.RESET}")
    
    # Create final folder and save final layout
    final_folder = input_data.job_folder / "final"
    final_folder.mkdir(parents=True, exist_ok=True)
    
    # Save final layout
    final_layout_path = final_folder / "layout.json"
    try:
        # Parse and format the JSON string
        layout_data = json.loads(final_layout_json_str)
        with open(final_layout_path, "w") as f:
            json.dump(layout_data, f, indent=4)
        print(f"{Fore.GREEN}Final layout saved to {final_layout_path}{Fore.RESET}")
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}Warning: Invalid JSON layout. Saving raw string: {e}{Fore.RESET}")
        with open(final_layout_path, "w") as f:
            f.write(final_layout_json_str)
    
    # Create and Render final scene
    print(f"{Fore.BLUE}Creating and rendering final scene...{Fore.RESET}")
    final_scene_path = final_folder / "scene.blend"
    create_and_render_input = CreateAndRenderInput(
        layout_path=final_layout_path,
        scene_prep_output=input_data.scene_prep_output,
        output_blend_path=final_scene_path,
        render_folder=final_folder / "renders",
        image_prefix="final_render"
    )
    
    try:
        create_and_render_output = create_and_render(create_and_render_input)
        final_rendered_images = create_and_render_output.render_scene_output.rendered_images
        print(f"{Fore.GREEN}Final scene created and rendered successfully{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}Error creating final scene: {e}{Fore.RESET}")
        # Use the last successful render from updates or init
        if updates_output.final_output and updates_output.final_output.scene_output:
            final_rendered_images = updates_output.final_output.scene_output.render_scene_output.rendered_images
            final_scene_path = updates_output.final_output.scene_output.create_scene_output.blend_file_path
        else:
            final_rendered_images = init_output.scene_output.render_scene_output.rendered_images
            final_scene_path = init_output.scene_output.create_scene_output.blend_file_path
        print(f"{Fore.YELLOW}Using last successful scene and renders{Fore.RESET}")
    
    # Create final output
    direct_output = DirectLoopOutput(
        init_output=init_output,
        updates_output=updates_output,
        final_layout_json_str=final_layout_json_str,
        final_layout_path=final_layout_path,
        final_scene_path=final_scene_path,
        final_rendered_images=final_rendered_images,
        is_satisfactory=is_satisfactory
    )
    
    # Save the complete output
    final_output_path = input_data.job_folder / "final" / "direct_loop_output.pkl"
    save_pydantic_model_as_pkl(direct_output, final_output_path)
    
    print(f"{Fore.GREEN}Direct Loop Pipeline completed successfully!{Fore.RESET}")
    print(f"{Fore.GREEN}Final results: {final_layout_path}{Fore.RESET}")
    print(f"{Fore.GREEN}Layout satisfactory: {is_satisfactory}{Fore.RESET}")
    print(f"{Fore.GREEN}Complete output saved to: {final_output_path}{Fore.RESET}")
    
    return direct_output


if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_scene_desc, example_decomposer_output, example_sceneprep_output
    
    input_data = DirectLoopInput(
        user_request=example_scene_desc,
        decomposer_output=example_decomposer_output,
        scene_prep_output=example_sceneprep_output,
        job_folder=Path("exp") / "test_direct_loop3",
        max_update_iterations=3,
        provider="openai",
        model="4omini",
        temperature=1
    )
    
    output = direct_loop(input_data)
    print(f"Direct Loop completed successfully!")
    print(f"Final layout satisfactory: {output.is_satisfactory}")
    print(f"Final scene path: {output.final_scene_path}")
