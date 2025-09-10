import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())


from pipelines.csp.csp_inner_loop_init import InnerLoopInitInput, InnerLoopInitOutput, inner_loop_init
from pipelines.csp.csp_inner_loop_updates import UpdatesInput, UpdatesOutput, updates
from pipelines.common.create_and_render import CreateAndRenderInput, CreateAndRenderOutput, create_and_render
from agents.decomposer import DecomposerOutput
from pipelines.common.prep_scene import ScenePrepOutput
from pipelines.common.utils import save_pydantic_model_as_pkl

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
from colorama import Fore
import json


class CSPInnerLoopInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    # NOTE: this field is included in scene_prep_output as well, but we keep it here for easy access and modularity in case we wish to change scene prep output later
    decomposer_output: DecomposerOutput = Field(..., description="Output from the decomposer containing asset list and descriptions")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from the scene preparation pipeline containing generated model files")
    job_folder: Path = Field(..., description="Path to the job folder where the scene will be processed")
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the inner loop")
    layout_verify_iters: int = Field(3, description="Maximum number of iterations for layout verification; this is shared between init and updates")
    code_fix_iters: int = Field(3, description="Maximum number of iterations for EACH error fixing during code execution; this is shared between init and updates")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")


class CSPInnerLoopOutput(BaseModel):
    init_output: InnerLoopInitOutput = Field(..., description="Output from the initialization phase")
    updates_output: UpdatesOutput = Field(..., description="Output from the updates phase")
    final_code: str = Field(..., description="Final verified code from the complete inner loop process")
    final_layout_path: Path = Field(..., description="Path to the final layout JSON file")
    final_scene_path: Path = Field(..., description="Path to the final Blender scene file")
    final_rendered_images: List[Path] = Field(..., description="Paths to the final rendered images")
    is_satisfactory: bool = Field(..., description="Whether the final layout was deemed satisfactory")


def csp_inner_loop(input_data: CSPInnerLoopInput) -> CSPInnerLoopOutput:
    """
    Complete CSP inner loop pipeline that combines initialization and updates sequentially.
    
    This pipeline:
    1. Runs the inner loop initialization (planning, coding, layout verification, scene creation, rendering)
    2. Runs the inner loop updates (review iterations with potential layout improvements)
    3. Returns the final output with the best available results
    
    Args:
        input_data: CSPInnerLoopInput containing all necessary inputs for the complete inner loop
        
    Returns:
        CSPInnerLoopOutput containing all outputs from both init and update phases
    """
    print(f"{Fore.CYAN}Starting CSP Inner Loop Pipeline{Fore.RESET}")
    print(f"{Fore.CYAN}Job folder: {input_data.job_folder}{Fore.RESET}")
    
    # Phase 1: Initialization
    print(f"{Fore.BLUE}Running Inner Loop Initialization...{Fore.RESET}")
    init_folder = input_data.job_folder / "csp_init"
    init_input = InnerLoopInitInput(
        user_request=input_data.user_request,
        decomposer_output=input_data.decomposer_output,
        scene_prep_output=input_data.scene_prep_output,
        job_folder=init_folder,
        layout_verify_iters=input_data.layout_verify_iters,
        code_fix_iters=input_data.code_fix_iters,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    init_output = inner_loop_init(init_input)
    print(f"{Fore.GREEN}✔ Inner Loop Initialization completed successfully{Fore.RESET}")

    # Phase 2: Updates
    print(f"{Fore.BLUE}Running Inner Loop Updates...{Fore.RESET}")
    
    updates_input = UpdatesInput(
        original_task_instruction=init_output.coder_output.sent_query,
        last_code=init_output.layout_verification_output.final_code,
        last_rendered_images=init_output.scene_output.render_scene_output.rendered_images,
        decomposer_output=input_data.decomposer_output,
        scene_prep_output=input_data.scene_prep_output,
        job_folder=input_data.job_folder,
        max_iterations=input_data.max_update_iterations,
        layout_verify_iters=input_data.layout_verify_iters,
        code_fix_iters=input_data.code_fix_iters,
        starting_index=1,
        provider=input_data.provider,
        model=input_data.model,
        temperature=input_data.temperature
    )
    updates_output = updates(updates_input)
    print(f"{Fore.GREEN}✔ Inner Loop Updates completed successfully{Fore.RESET}")

    # Phase 3: Generate Final Results
    print(f"{Fore.BLUE}Generating final scene and outputs...{Fore.RESET}")
    
    # Determine which results to use: updates final_output or init results
    if updates_output.final_output is None:
        # Use initialization results
        final_layout_verification = init_output.layout_verification_output
        is_satisfactory = False  # No updates were satisfactory
        print(f"{Fore.YELLOW}Using initialization results (no successful updates or initial results are satisfactory){Fore.RESET}")
    else:
        # Use final update results (guaranteed to be valid)
        final_layout_verification = updates_output.final_output.layout_verification_output
        # we cannot use final output's review as the review by that time was not yet satisfactory - it should have been deemed satisfactory by next iteration's review which is only returned in the final list of updates
        # so we check for the review of the last update
        is_satisfactory = updates_output.updates[-1].reviewer_output.satisfactory
        print(f"{Fore.GREEN}Using final update results{Fore.RESET}")
    
    # Create final folder and save final code and layout
    final_folder = input_data.job_folder / "final"
    final_folder.mkdir(parents=True, exist_ok=True)
    
    # Save final code
    final_code = final_layout_verification.final_code
    final_code_path = final_folder / "code.py"
    with open(final_code_path, "w") as f:
        f.write(final_code)
    print(f"{Fore.GREEN}Final code saved to {final_code_path}{Fore.RESET}")
    
    # Save final layout
    final_layout_path = final_folder / "layout.json"
    with open(final_layout_path, "w") as f:
        json.dump(final_layout_verification.verified_layout, f, indent=4)
    print(f"{Fore.GREEN}Final layout saved to {final_layout_path}{Fore.RESET}")
    
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
    create_and_render_output = create_and_render(create_and_render_input)
    final_rendered_images = create_and_render_output.render_scene_output.rendered_images
    print(f"{Fore.GREEN}Final scene created and rendered successfully{Fore.RESET}")
    
    # Create final output
    csp_output = CSPInnerLoopOutput(
        init_output=init_output,
        updates_output=updates_output,
        final_code=final_code,
        final_layout_path=final_layout_path,
        final_scene_path=final_scene_path,
        final_rendered_images=final_rendered_images,
        is_satisfactory=is_satisfactory
    )
    
    # Save the complete output
    final_output_path = input_data.job_folder / "final" / "csp_inner_loop_output.pkl"
    save_pydantic_model_as_pkl(csp_output, final_output_path)
    
    print(f"{Fore.GREEN}CSP Inner Loop Pipeline completed successfully!{Fore.RESET}")
    print(f"{Fore.GREEN}Final results: {final_layout_path}{Fore.RESET}")
    print(f"{Fore.GREEN}Layout satisfactory: {is_satisfactory}{Fore.RESET}")
    print(f"{Fore.GREEN}Complete output saved to: {final_output_path}{Fore.RESET}")
    
    return csp_output


if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_scene_desc, example_decomposer_output, example_sceneprep_output
    
    input_data = CSPInnerLoopInput(
        user_request=example_scene_desc,
        decomposer_output=example_decomposer_output,
        scene_prep_output=example_sceneprep_output,
        job_folder=Path("exp") / "test_refactor_iters",
        max_update_iterations=3,
        provider="ollama",
        model="gemma3",
    )
    
    output = csp_inner_loop(input_data)
    print(f"CSP Inner Loop completed successfully!")
    print(f"Final layout satisfactory: {output.is_satisfactory}")
    print(f"Final scene path: {output.final_scene_path}")
