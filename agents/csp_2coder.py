import os
import sys
sys.path.append(os.getcwd())

from llms.llms import helper
from datetime import datetime

from pydantic import BaseModel, Field
from typing import Tuple

# layout matrix taken from scenecraft appendix
# @dataclass 
# class Layout: 
#     location: Tuple[float, float, float] 
#     min: Tuple[float, float, float] 
#     max: Tuple[float, float, float] 
#     orientation: Tuple[float, float, float] # Euler angles (pitch, yaw, roll)

# TODO: blender object to AABB and position, rotation
# the layout matrix works for concrete objects, but what about light directions and terrain stuff? 
# should we pass in Blender object references? Would it require a debugging process? 
# _prompt = """
# You are an expert of writing ORTOOlS code to express constraints in terms of scoring functions. Given a relational bipartite graph 'G(s) = (A, R, E)' where:
# - 'A' represents the set of assets.
# - 'R' represents the set of relations as nodes.
# - 'E' represents the edges connecting a relation node to a subset of assets 'E(r)' in the scene that satisfies this relation.
# Please write corresponding python methods for each relation type that takes in assets parameters and returns a score based on how well they satisfy the relation. The scoring function should return a higher score when the relation is satisfied and a lower score when it is not. For each input asset, the method will receive a layout matrix defined by the following dataclass:
# class Layout:
#     location: Tuple[float, float, float]  # location of the asset in 3D space
#     min: Tuple[float, float, float]  # minimum corner of the AABB bounding box
#     max: Tuple[float, float, float]  # maximum corner of the AABB bounding box
#     orientation: Tuple[float, float, float]  # Euler angles (pitch, yaw, roll)

# Return your methods in a single markdown code block and make sure each method can be imported and used directly:
# ```python
# # your code
# ```
# """
_prompt = """
You are an expert in writing Optuna code and Python scripts to find optimal 3D scene layouts. 

Given a relational bipartite graph 'G(s) = (A, R, E)' where:
- 'A' represents the set of assets.
- 'R' represents the set of relations as nodes.
- 'E' represents the edges connecting a relation node to a subset of assets 'E(r)' in the scene that satisfies this relation.

Please first write Python methods for each relation type that takes in assets parameters and returns a score based on how well they satisfy the relation. The scoring function should return a higher score when the relation is satisfied and a lower score when it is not. For each input asset, the method will receive a layout matrix defined by the following dataclass:

```python
class Layout:
    location: Tuple[float, float, float]  # location of the asset in 3D space
    min: Tuple[float, float, float]  # minimum corner of the AABB bounding box
    max: Tuple[float, float, float]  # maximum corner of the AABB bounding box
    orientation: Tuple[float, float, float]  # Euler angles (pitch, yaw, roll) in radians
```

Then, use these methods as scoring functions in Optuna to express constraints for the scene layout optimization problem by adding them to the solver. After running the solver, save the optimal layout found in a "layout.json" file in the same directory as the script (get script location explicitly with `os.path.dirname(os.path.realpath(__file__))`), mapping exact asset names from given asset list to their corresponding layout matrices. When you use `optuna.create_study`, please set `store=None` to avoid creating any other files. The up-axis is the Z-axis, front is X-axis and Y-axis points to right in a [x, y, z] coordinate system.

Return your whole Python script in a single markdown code block and make sure it can be run directly:
```python
# your code
```
"""

class CoderInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    assets_str: str = Field(..., description="List of decomposed 3D assets in md string format")
    graph: str = Field(..., description="Relational bipartite graph representing the scene layout")
    session_id: str | None = Field(None, description="Session ID for the LLM query")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class CoderOutput(BaseModel):
    sent_query: str = Field(..., description="The query sent to the LLM for generating code")
    code: str = Field(..., description="Generated Optuna code for scoring functions")
    session_id: str = Field(..., description="Session ID for the LLM query")

def code(input_data: CoderInput) -> CoderOutput:
    """Generates Optuna code for scoring functions based on the relational bipartite graph."""
    query_prompt = _prompt + f"\n\nUser Request: {input_data.user_request}\n\n\nGraph:\nAsset List:\n{input_data.assets_str}\n{input_data.graph}\n\n"
    session_id = input_data.session_id or f"csp_2coder-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = helper.query(
        provider=input_data.provider,
        model=input_data.model,
        user_prompt=query_prompt,
        save_path=session_id,
        temperature=input_data.temperature,
    )
    generated_code = helper.extract_code_block(response, 'python')
    return CoderOutput(sent_query=query_prompt, code=generated_code, session_id=session_id)


if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_coder_query, example_decomposer_output, example_graph
    input_data = CoderInput(
        user_request=example_coder_query,
        assets_str=example_decomposer_output.formatted_asset_str,
        graph=example_graph,
        provider="ollama",
        model="gemma3"
    )
    output = code(input_data)
    print(output.code)
    