import os
import sys
sys.path.append(os.getcwd())

from llms.llms import helper
from datetime import datetime
import json

from pydantic import BaseModel, Field
from typing import Tuple
from agents.decomposer import DecomposerOutput

# Prompt for direct scene layout generation
_prompt = f"""
You are tasked with creating a detailed 3D scene layout in JSON format based on the provided description and asset list. Your goal is to directly specify the position, rotation, and scale of each asset in the scene.

Please think step-by-step:
1. Review the scene description and the list of assets.
2. Determine the spatial layout and arrangement of objects in the scene.
3. For each asset, specify its position (x, y, z coordinates), rotation (x, y, z angles in degrees), and scale (x, y, z scaling factors).
4. Consider realistic spatial relationships like objects on surfaces, proper spacing, natural orientations, and logical groupings.
5. Use a coordinate system where Z is up, and position objects appropriately relative to a ground plane at Z=0.

Output the layout in the following JSON format, ensuring to include all assets from the provided list using their exact names as keys:

```json
{{
    "asset_name": {{
        "location": [x, y, z],
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "orientation": [pitch, yaw, roll],
    }},
    ...
}}
```

Guidelines:
- values in the tuples are floats
- Position coordinates should be realistic (e.g., furniture on ground, decorations at appropriate heights)
- min and max should define the axis-aligned bounding box (AABB) of the assets
- Rotation angles are in degrees (0-360)
- Consider the natural flow and accessibility of the space
- Ensure objects don't overlap inappropriately
- Group related objects logically

You may start with any reasoning but in the end ensure to provide the JSON data wrapped in a single markdown code block like above. 
"""

class DirectPlannerInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    assets_str: str = Field(..., description="List of decomposed 3D assets in md string format")
    session_id: str | None = Field(None, description="Session ID for the LLM query")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class DirectPlannerOutput(BaseModel):
    sent_query: str = Field(..., description="The query sent to the LLM for generating the scene layout")
    scene_layout: str = Field(..., description="Scene layout in JSON format")
    session_id: str = Field(..., description="Session ID for the LLM query")

def plan(input_data: DirectPlannerInput) -> DirectPlannerOutput:
    """Plans the 3D scene layout directly in JSON format based on the user request and asset list."""
    query_prompt = _prompt + f"\n\nUser Request: {input_data.user_request}\n\nAsset List:\n{input_data.assets_str}\n\n"
    session_id = input_data.session_id or f"direct_planner-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = helper.query(
        provider=input_data.provider,
        model=input_data.model,
        user_prompt=query_prompt,
        save_path=session_id,
        temperature=input_data.temperature,
    )
    scene_layout = extract_scene_layout(response)
    return DirectPlannerOutput(sent_query=query_prompt, scene_layout=scene_layout, session_id=session_id)

def extract_scene_layout(response: str) -> str:
    """Extracts the JSON scene layout from the response."""
    json_str = helper.extract_code_block(response, 'json')
    if not json_str:
        raise ValueError("No valid JSON layout found in the response.")
    return json_str.strip()

if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_scene_desc, example_decomposer_output
    input_data = DirectPlannerInput(
        user_request=example_scene_desc,
        assets_str=example_decomposer_output.formatted_asset_str,
        provider="ollama",
        model="gemma3"
    )
    output = plan(input_data)
    print(output.scene_layout)
