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
You are an expert in 3D scene synthesis. Given a user request, you need to identify a list of 3D assets involved in the scene and provide descriptions for them, as well as where they are in the scene, so other 3D modeling specialists can produce suitable 3D models and put them into right places. Please think step-by-step: 
- what is the type, scale and style of the scene? 
- what is a suitable granularity of assets? 
- apart from concrete objects, what other elements, for example lighting, are needed - for those, you don't need to include them in the asset list, but you should describe them in the description of related assets
- specifically, when you are describing an attachment of another asset as an individual asset, you should make it clear in the description that it is an attachment, and so as to tell the 3D modeler to not model the main asset again
- when you name the assets, make sure they are unique and also suitable to be used as file names, so no special characters, spaces, etc.

Upon receiving the user request, you may start by answering the above questions, ask and answer additional questions by yourself if needed, and then provide the asset list with the following JSONL format, with each line being a distinct asset to be 3D modeled and put into the scene:
```jsonl
{"name": "asset_name", "description": "asset_description", "location": "asset_location"}
... (wrap all JSONL lines in the same markdown code block)
```

User 3D scene synthesis request: 
"""

class DecomposerInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    session_id: str | None = Field(None, description="Session ID for the LLM query")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class DecomposedAsset(BaseModel):
    name: str = Field(..., description="Name of the 3D asset")
    description: str = Field(..., description="Description of the 3D asset")
    location: str = Field(..., description="Location of the 3D asset in the scene")

class DecomposerOutput(BaseModel):
    assets: list[DecomposedAsset] = Field(..., description="List of decomposed 3D assets")
    formatted_asset_str: str = Field(..., description="Formatted string representation of the asset list")
    session_id: str = Field(..., description="Session ID for the LLM query")

def decompose(input_data: DecomposerInput) -> DecomposerOutput:
    """Decomposes a user request into a list of 3D assets and their descriptions."""
    query_prompt = _prompt + input_data.user_request + "\n\n"
    session_id = input_data.session_id or f"decomposer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    input_data = DecomposerInput(user_request=user_request)
    output = decompose(input_data)
    print(output.model_dump())
    


