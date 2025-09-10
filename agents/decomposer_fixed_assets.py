import sys
import os
sys.path.append(os.getcwd())

from llms.llms import helper

from datetime import datetime
import json
from pathlib import Path
from typing import Tuple, List
from pydantic import BaseModel, Field

# original scene craft seems quite coarse in concating their final sub-scenes, maybe we can try to improve it, e.g. describe sub-scene relationships too
# or we don't decompose into sub-scenes at all
# solution: here we obtain a list of assets (all of them) and their descriptions, we decide later if we want to use sub-scenes - the asset list length can be a signal too
# alternatively we may simply make it decompose into different steps of modeling the scene, so to lower the workload for each step - previous steps' results will be frozen
_prompt = """
You are an expert in 3D scene synthesis. Given a user request and a list of user-provided asset descriptions, you need to understand the scene and describe the location of 3D assets involved in the scene, so other 3D modeling specialists can retrieve suitable 3D models and put them into right places. Please think step-by-step: 
- what is the type, scale and style of the scene? 
- what is the best way to specify the mutual relationships between different objects, e.g. relative positions, orientations, etc.?
- use all of the user-provided assets and only those assets, no more no less - when there are repeated object descriptions, it means there are multiple instances of them and you need to handle them separately, and when an object description describes multiple objects, you still need to treat them as one object - please ensure that the number of objects you list matches the number of user-provided object descriptions
- when you provide your answer, alongside location descriptions, make sure to use exact same object descriptions as provided by the user, and you need to name them with unique names suitable to be used as file names, so no special characters, spaces, etc.

Upon receiving the user request, you may start by answering the above questions, ask and answer additional questions by yourself if needed, and then provide the full asset list with the following JSONL format, with each line being a distinct asset to be 3D modeled and put into the scene:
```jsonl
{"name": "unique_suitable_name", "description": "asset_description (as user provided)", "location": "asset_location"}
... (wrap all JSONL lines in the same markdown code block)
```

User 3D scene synthesis request:
"""

class DecomposerFAInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    asset_descriptions: List[str] = Field(..., description="List of object descriptions provided by the user")
    session_id: str | None = Field(None, description="Session ID for the LLM query")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

from agents.decomposer import DecomposedAsset, DecomposerOutput

def decompose_with_fixed_assets(input_data: DecomposerFAInput) -> DecomposerOutput:
    """Decomposes a user request into a list of 3D assets and their descriptions."""
    query_prompt = (
        _prompt + input_data.user_request + "\n\nUser-provided object descriptions:\n" +
        "\n".join(f"- {desc}" for desc in input_data.asset_descriptions) + "\n\n"
    )
    session_id = input_data.session_id or f"decomposerFA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = helper.query(
        provider=input_data.provider,
        model=input_data.model,
        user_prompt=query_prompt,
        save_path=session_id,
        temperature=input_data.temperature,
    )
    jsonl_str = helper.extract_code_block(response, 'jsonl')
    jsonl_data = [json.loads(line) for line in jsonl_str.splitlines() if line.strip()]
    assets = [DecomposedAsset(**asset) for asset in jsonl_data]
    formatted_asset_str = format_asset_list(assets)
    return DecomposerOutput(assets=assets, formatted_asset_str=formatted_asset_str, session_id=session_id)

def format_asset_list(assets: List[DecomposedAsset]) -> str:
    """Formats the list of decomposed assets into a string representation."""
    return "\n".join([
        f"- {asset.name}:\n  - Description: {asset.description}\n  - Location: {asset.location}"
        for asset in assets
    ])

if __name__ == '__main__':
    user_request = "A cozy living room with a fireplace, a sofa, and a coffee table."
    asset_list = [
        "a modern fireplace with a wooden mantel",
        "a large, comfortable sofa with cushions",
        "a rectangular coffee table made of glass",
        "a floor lamp with a warm light",
        "a rug with a geometric pattern"
    ]
    input_data = DecomposerFAInput(user_request=user_request, asset_descriptions=asset_list)
    output = decompose_with_fixed_assets(input_data)
    print(output.model_dump())
    


