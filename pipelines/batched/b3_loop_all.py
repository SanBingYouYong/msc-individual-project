'''
This file calls update loops (either csp or direct) on a batch of scenes with decomposed and obtained assets.
'''
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Union
from tqdm import tqdm
import json
from time import time
from colorama import Fore, init
init()

from pipelines.common.prep_scene import ScenePrepOutput
from agents.decomposer import DecomposedAsset, DecomposerOutput
from pipelines.common.utils import save_pydantic_model_as_pkl, load_pydantic_model_from_pkl

from pipelines.csp.csp_inner_loop import CSPInnerLoopInput, CSPInnerLoopOutput, csp_inner_loop
from pipelines.direct.direct import DirectLoopInput, DirectLoopOutput, direct_loop
from pipelines.run_csp import CSPPipelineOutput
from pipelines.run_direct import DirectPipelineOutput


class BatchLoopInput(BaseModel):
    """Input for batch processing of update loops on decomposed scenes."""
    b2_folder: Path = Field(..., description="Path to the experiment folder containing scene subdirectories with decomposed and obtained assets")
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the inner loop")
    loop_method: Literal['csp', 'direct'] = Field('direct', description="Method for the inner loop; 'csp' for constraint satisfaction, 'direct' for direct optimization")
    # shared param
    max_update_iterations: int = Field(3, description="Maximum number of update iterations for the inner loop")
    # csp-specific params
    layout_verify_iters: int = Field(3, description="Maximum number of iterations for layout verification; only used if loop_method is 'csp'")
    code_fix_iters: int = Field(3, description="Maximum number of iterations for EACH error fixing during code execution; only used if loop_method is 'csp'")
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")

def prepare_input(scene_dir: Path, batch_loop_input: BatchLoopInput) -> CSPInnerLoopInput:
    prep_dir = scene_dir / "prep"
    if not prep_dir.exists():
        raise RuntimeError(f"Prep directory {prep_dir} does not exist, please run b0_initialize_all.py first")
    scene_txt_path = scene_dir / "_scene.txt"
    if not scene_txt_path.exists():
        raise RuntimeError(f"No _scene.txt found in {scene_dir}, cannot proceed")
    with scene_txt_path.open('r') as f:
        user_request = f.read().strip()
    
    scene_prep_output_path = prep_dir / "scene_prep_output.pkl"
    if not scene_prep_output_path.exists():
        raise RuntimeError(f"No scene_prep_output.pkl found in {prep_dir}, cannot proceed")
    scene_prep_output: ScenePrepOutput = load_pydantic_model_from_pkl(ScenePrepOutput, scene_prep_output_path)
    decomposer_output = scene_prep_output.decomposer_output
    
    if batch_loop_input.loop_method == 'csp':
        job_folder = scene_dir / "csp"
        return CSPInnerLoopInput(
            user_request=user_request,
            decomposer_output=decomposer_output,
            scene_prep_output=scene_prep_output,
            job_folder=job_folder,
            max_update_iterations=batch_loop_input.max_update_iterations,
            layout_verify_iters=batch_loop_input.layout_verify_iters,
            code_fix_iters=batch_loop_input.code_fix_iters,
            provider=batch_loop_input.provider,
            model=batch_loop_input.model,
            temperature=batch_loop_input.temperature
        )
    elif batch_loop_input.loop_method == 'direct':
        job_folder = scene_dir / "direct"
        return DirectLoopInput(
            user_request=user_request,
            decomposer_output=decomposer_output,
            scene_prep_output=scene_prep_output,
            job_folder=job_folder,
            max_update_iterations=batch_loop_input.max_update_iterations,
            provider=batch_loop_input.provider,
            model=batch_loop_input.model,
            temperature=batch_loop_input.temperature
        )
    else:
        raise ValueError(f"Unsupported loop_method: {batch_loop_input.loop_method}")

def batch_loop(input_data: BatchLoopInput) -> None:
    """
    For each scene in the specified experiment folder, runs the chosen inner loop method (CSP or direct) using the decomposed and obtained assets.
    """
    if not input_data.b2_folder.exists():
        raise ValueError(f"Experiment folder {input_data.b2_folder} does not exist")
    
    scene_dirs = [d for d in input_data.b2_folder.iterdir() if d.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No scene subdirectories found in {input_data.b2_folder}")
    
    errored_scenes = []
    for scene_dir in tqdm(scene_dirs, desc="Running inner loops on decomposed scenes"):
        try:
            loop_input = prepare_input(scene_dir, input_data)
            start_time = time()
            if input_data.loop_method == 'csp':
                loop_output: CSPInnerLoopOutput = csp_inner_loop(loop_input)
                output_path = loop_input.job_folder / "csp_inner_loop_output.pkl"
                save_pydantic_model_as_pkl(loop_output, output_path)
                end_time = time()
                processing_time = end_time - start_time
                pipeline_output = CSPPipelineOutput(
                    user_request=loop_input.user_request,
                    scene_prep_output=loop_input.scene_prep_output,
                    csp_inner_loop_output=loop_output,
                    job_folder=loop_input.job_folder,
                    final_scene_path=loop_output.final_scene_path,
                    final_rendered_images=loop_output.final_rendered_images,
                    is_satisfactory=loop_output.is_satisfactory,
                    processing_time_seconds=processing_time
                )
                pipeline_output_path = loop_input.job_folder / "csp_pipeline_output.pkl"
                save_pydantic_model_as_pkl(pipeline_output, pipeline_output_path)
            elif input_data.loop_method == 'direct':
                loop_output: DirectLoopOutput = direct_loop(loop_input)
                output_path = loop_input.job_folder / "direct_loop_output.pkl"
                save_pydantic_model_as_pkl(loop_output, output_path)
                end_time = time()
                processing_time = end_time - start_time
                pipeline_output = DirectPipelineOutput(
                    user_request=loop_input.user_request,
                    scene_prep_output=loop_input.scene_prep_output,
                    direct_loop_output=loop_output,
                    job_folder=loop_input.job_folder,
                    final_scene_path=loop_output.final_scene_path,
                    final_rendered_images=loop_output.final_rendered_images,
                    is_satisfactory=loop_output.is_satisfactory,
                    processing_time_seconds=processing_time
                )
                pipeline_output_path = loop_input.job_folder / "direct_pipeline_output.pkl"
                save_pydantic_model_as_pkl(pipeline_output, pipeline_output_path)
            else:
                raise ValueError(f"Unsupported loop_method: {input_data.loop_method}")
            print(f"{Fore.GREEN}✔ Completed processing scene {scene_dir.name}{Fore.RESET}")
        except Exception as e:
            print(f"Error processing scene {scene_dir.name}: {e}")
            errored_scenes.append(scene_dir.name)
            continue
    if errored_scenes:
        print(f"{Fore.RED}Errors occurred in the following scenes: {', '.join(errored_scenes)}{Fore.RESET}")
    else:
        print(f"{Fore.GREEN}Batch loop processing completed for all scenes in {input_data.b2_folder}{Fore.RESET}")



if __name__ == '__main__':
    batch_loop(
        BatchLoopInput(# NOTE: we can reuse job folders for csp/ and direct/
            b2_folder=Path("exp/is_bedroom_100/"),
            max_update_iterations=3,
            loop_method='direct',
            layout_verify_iters=5,
            code_fix_iters=5,
            provider="ollama",
            model="gemma3",
            temperature=0.7
        )
    )
    batch_loop(
        BatchLoopInput(
            b2_folder=Path("exp/is_bedroom_100/"),
            max_update_iterations=3,
            loop_method='csp',
            layout_verify_iters=5,
            code_fix_iters=5,
            provider="ollama",
            model="gemma3",
            temperature=0.7
        )
    )
        
        