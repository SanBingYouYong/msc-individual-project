import sys
import os
sys.path.append(os.getcwd())

from agents import csp_1planner, csp_2coder
from agents.decomposer import DecomposerOutput
from pipelines.common.prep_scene import ScenePrepOutput
from pipelines.csp.layout_verification import LayoutVerificationInput, LayoutVerificationOutput, layout_verification
from pipelines.common.create_and_render import CreateAndRenderInput, CreateAndRenderOutput, create_and_render
from llms.llms import helper
from pipelines.common.utils import save_pydantic_model_as_pkl

from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
from colorama import Fore
import json
from yaspin import yaspin

class InnerLoopInitInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    decomposer_output: DecomposerOutput = Field(..., description="Output from the decomposer containing asset list and descriptions")
    scene_prep_output: ScenePrepOutput = Field(..., description="Output from the scene preparation pipeline containing generated model files")
    job_folder: Path = Field(..., description="Path to the job folder where the scene will be processed; expected: job-folder/csp_init")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")
    layout_verify_iters: int = Field(3, description="Maximum number of iterations for layout verification")
    code_fix_iters: int = Field(3, description="Maximum number of iterations for EACH error fixing during code execution")

class InnerLoopInitOutput(BaseModel):
    planner_output: csp_1planner.PlannerOutput = Field(..., description="Output from the planner containing the relational bipartite graph")
    coder_output: csp_2coder.CoderOutput = Field(..., description="Output from the coder containing the scoring code")
    layout_verification_output: LayoutVerificationOutput = Field(..., description="Output from the layout verification process")
    scene_output: CreateAndRenderOutput = Field(..., description="Output from the scene creation and rendering process")
    
def inner_loop_init(input_data: InnerLoopInitInput):
    """Initializes the inner loop for the CSP process by planning the scene layout and generating scoring functions."""
    # Ensure the job folder exists
    input_data.job_folder.mkdir(parents=True, exist_ok=True)
    # create a subfolder for pipeline input outputs
    stage_folder = input_data.job_folder / "stage"
    stage_folder.mkdir(parents=True, exist_ok=True)
    staged_files: List[Path] = []

    with yaspin(text=f"Inner Loop Init: Planner...", color="green", timer=True) as spinner:
        planner_input = csp_1planner.PlannerInput(
            user_request=input_data.user_request,
            assets_str=input_data.decomposer_output.formatted_asset_str,
            provider=input_data.provider,
            model=input_data.model,
            temperature=input_data.temperature)
        planner_output = csp_1planner.plan(planner_input)
        staged_files.append(save_pydantic_model_as_pkl(planner_output, stage_folder / "planner_output.pkl"))
        with open(input_data.job_folder / "graph.txt", "w") as f:
            f.write(planner_output.graph)
        spinner.ok("✔")
    
    with yaspin(text=f"Inner Loop Init: Coder...", color="green", timer=True) as spinner:
        coder_input = csp_2coder.CoderInput(
            user_request=input_data.user_request,
            assets_str=input_data.decomposer_output.formatted_asset_str,
            graph=planner_output.graph,
            provider=input_data.provider,
            model=input_data.model,
            temperature=input_data.temperature)
        coder_output = csp_2coder.code(coder_input)
        staged_files.append(save_pydantic_model_as_pkl(coder_output, stage_folder / "coder_output.pkl"))
        with open(input_data.job_folder / "code.py", "w") as f:
            f.write(coder_output.code)
        spinner.ok("✔")

    with yaspin(text=f"Inner Loop Init: Code Execution and Layout Verification...", color="green", timer=True) as spinner:
        context_provider_history = helper.load_history(coder_output.session_id, as_history_messages=True)
        original_task_instruction = context_provider_history[0]['content'][0]['text']
        layout_verification_input = LayoutVerificationInput(
            code=coder_output.code,
            decomposer_output=input_data.decomposer_output,
            original_task_instruction=original_task_instruction,
            working_dir=input_data.job_folder / "temp",
            max_iterations=input_data.layout_verify_iters,
            error_fixing_max_iter=input_data.code_fix_iters,
            provider=input_data.provider,
            model=input_data.model,
            temperature=input_data.temperature,
        )
        layout_verification_output = layout_verification(layout_verification_input)
        staged_files.append(save_pydantic_model_as_pkl(layout_verification_output, stage_folder / "layout_verification_output.pkl"))
        spinner.ok("✔")

    if not layout_verification_output.is_success:
        raise RuntimeError("Layout verification failed during the inner loop initialization.")
    
    # save the final code to a py file
    final_code = layout_verification_output.final_code
    with open(input_data.job_folder / "code.py", "w") as f:
        f.write(final_code)
    print(f"{Fore.GREEN}Final code saved to {input_data.job_folder / 'code.py'}{Fore.RESET}")
    # save verified layout to a json file
    if not layout_verification_output.verified_layout:
        raise RuntimeError("!!!No verified layout returned after passing the code execution and layout verification.")
    layout_path = input_data.job_folder / "layout.json"
    with open(layout_path, "w") as f:
        json.dump(layout_verification_output.verified_layout, f, indent=4)
    print(f"{Fore.GREEN}Layout JSON saved to {layout_path}{Fore.RESET}")

    with yaspin(text="Inner Loop Init: Creating and Rendering scene according to layout...", color="green", timer=True) as spinner:
        create_and_render_input = CreateAndRenderInput(
            layout_path=layout_path,
            scene_prep_output=input_data.scene_prep_output,
            output_blend_path=input_data.job_folder / "scene.blend",
            render_folder=input_data.job_folder / "renders",
            image_prefix="init_render"
        )
        create_and_render_output = create_and_render(create_and_render_input)
        staged_files.append(save_pydantic_model_as_pkl(create_and_render_output, stage_folder / "create_and_render_output.pkl"))
        spinner.ok("✔")
    
    inner_loop_init_output = InnerLoopInitOutput(
        planner_output=planner_output,
        coder_output=coder_output,
        layout_verification_output=layout_verification_output,
        scene_output=create_and_render_output
    )

    # remove the stage folder and save the final big output
    final_output_path = input_data.job_folder / "inner_loop_init_output.pkl"
    save_pydantic_model_as_pkl(inner_loop_init_output, final_output_path)
    # remove staged pkls
    for staged_file in staged_files:
        if staged_file.exists():
            staged_file.unlink()
    try:
        stage_folder.rmdir()
    except OSError:
        print(f"Warning: Could not remove stage folder {stage_folder}. It may not be empty.")

    # # move everything into init folder
    # init_folder = input_data.job_folder / "csp_init"
    # init_folder.mkdir(parents=True, exist_ok=True)

    # # move renders, temp, code.py, graph.txt, pkl, layout.json and scene.blend
    # items = [
    #     input_data.job_folder / "renders",
    #     input_data.job_folder / "temp",
    #     input_data.job_folder / "code.py",
    #     input_data.job_folder / "graph.txt",
    #     input_data.job_folder / "layout.json",
    #     input_data.job_folder / "scene.blend",
    #     final_output_path
    # ]
    # for item in items:
    #     if item.exists():
    #         item.rename(init_folder / item.name)

    return inner_loop_init_output
    


if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_coder_query, example_decomposer_output, example_sceneprep_output
    input_data = InnerLoopInitInput(
        user_request=example_coder_query,
        decomposer_output=example_decomposer_output,
        scene_prep_output=example_sceneprep_output,
        job_folder=Path("exp") / "test_inner_loop_init_gemini10",
        provider="google",
        model="gemini-2.0-flash-lite",
    )
    output = inner_loop_init(input_data)
    print(f"Inner loop initialization completed successfully!")
