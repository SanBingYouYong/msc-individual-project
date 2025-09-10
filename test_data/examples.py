import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
from agents.decomposer import DecomposerOutput, DecomposedAsset, format_asset_list
from pipelines.common.prep_scene import ScenePrepOutput
from pipelines.common.create_and_render import CreateAndRenderOutput
import json
from typing import List
from uuid import uuid4

with open("test_data/example_decomposed_assets.jsonl", "r") as f:
    _example_decompose = f.read().strip()
def _parse_decompose(text: str) -> List[DecomposedAsset]:
    assets = []
    for line in text.strip().split('\n'):
        if line.strip():
            data = json.loads(line)
            asset = DecomposedAsset(**data)
            assets.append(asset)
    return assets

example_scene_desc = "A cozy living room with a fireplace, a sofa, and a coffee table."
_example_decomposed_assets = _parse_decompose(_example_decompose)
_example_formatted_asset_str = format_asset_list(_example_decomposed_assets)
example_decomposer_output = DecomposerOutput(
    assets=_example_decomposed_assets, 
    formatted_asset_str=_example_formatted_asset_str,
    session_id="example-fictional-decomposer-session")
with open("test_data/example_graph.txt", "r") as f:
    example_graph = f.read().strip()
example_sceneprep_output = ScenePrepOutput(
    decomposer_output=example_decomposer_output,
    obtained_asset_files={
        asset.name: Path(f"test_data/generated_models/{asset.name}.glb")
        for asset in _example_decomposed_assets
    }
)

with open("test_data/example_layout.json", "r") as f:
    example_layout = json.load(f)

with open("test_data/example_code.py", "r") as f:
    example_code = f.read()

from llms.llms import helper
example_csp_2coder_history = helper.load_history("example_csp_2coder_history", as_history_messages=True)

example_scene_blend_file = Path("test_data/coder-layout_scene.blend")

from agents.csp_1planner import PlannerOutput
example_planner_output = PlannerOutput(
    graph=example_graph,
    session_id="example-planner-session"
)
from agents.csp_2coder import CoderOutput
with open("test_data/example_coder_query.txt", "r") as f:
    example_coder_query = f.read().strip()
example_coder_output = CoderOutput(
    code=example_code,
    sent_query=example_coder_query,
    session_id="example-coder-session",
    history=example_csp_2coder_history
)
from pipelines.csp.layout_verification import LayoutVerificationOutput
example_layout_verification_output = LayoutVerificationOutput(
    is_success=True,
    final_code=example_code,
    verified_layout=example_layout,
    iterations_used=1,
    total_error_fixes=0
)

from pipelines.common.basic.create_scene import CreateSceneInput, CreateSceneOutput, create_scene
example_create_scene_output = CreateSceneOutput(
    blend_file_path=Path("test_data/coder-layout_scene.blend")
)

from pipelines.common.basic.render_scene import RenderSceneInput, RenderSceneOutput, render_scene
example_render_output = RenderSceneOutput(
    rendered_images=[
        Path(f"test_data/rendered_images/example_render-{axis}.png")
        for axis in ["x", "y", "z"]
    ]
)
example_scene_output = CreateAndRenderOutput(
    create_scene_output=example_create_scene_output,
    render_scene_output=example_render_output
)

from pipelines.csp.csp_inner_loop_init import InnerLoopInitOutput
example_inner_loop_init_output = InnerLoopInitOutput(
    planner_output=example_planner_output,
    coder_output=example_coder_output,
    layout_verification_output=example_layout_verification_output,
    scene_output=example_scene_output
)

with open("test_data/example_direct_planner_query.txt", "r") as f:
    example_direct_planner_query = f.read().strip()

if __name__ == '__main__':
    print(
        os.path.exists(
            example_sceneprep_output.obtained_asset_files[
                _example_decomposed_assets[0].name
            ]
        )
    )

