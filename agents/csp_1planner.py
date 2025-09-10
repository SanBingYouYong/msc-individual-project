import os
import sys
sys.path.append(os.getcwd())

from llms.llms import helper
from datetime import datetime

from pydantic import BaseModel, Field
from typing import Tuple
from agents.decomposer import DecomposerOutput

# asked gpt to finish the incomplete provided prompt
_prompt = f"""
You are tasked with constructing a relational bipartite graph for a 3D scene based on the provided description and asset list. Your goal is to model the spatial and contextual layout of the scene. Please think step-by-step: 
1. Review the scene description and the list of assets.
2. Determine the spatial and contextual relationships needed to accurately represent the scene's layout. Consider relationships like proximity, alignment, parallelism, containment, support, and others.
3. Construct the relational bipartite graph 'G(s) = (A, R, E)' where:
    - 'A' represents the set of assets (this is already given).
    - 'R' represents the set of relations as nodes.
    - 'E' represents the edges connecting a relation node to a subset of assets 'E(r)' in the scene that satisfies this relation.
4. For each identified relationship, create a relation node and link it to the appropriate assets through edges in the graph. For repeated relations of different assets, create multiple edges using the same relation node.

Output your findings in the following structured format: 

- mark the start of your structured output with a new line: "### Relational Bipartite Graph".

- List of relation nodes 'R' with their types and descriptions.
    - relation type: description.

- Edges 'E' that link assets to their corresponding relation nodes.
    - (Relation, list of assets that satisfy this relation): description of how the assets relate to each other.

This process will guide the arrangement of assets in the 3D scene, ensuring they are positioned, scaled, and oriented correctly according to the described intent. Be precise and exhaustive in capturing all meaningful spatial and contextual relationships necessary for accurate layout planning.
"""

class PlannerInput(BaseModel):
    user_request: str = Field(..., description="User request for the 3D scene synthesis")
    assets_str: str = Field(..., description="List of decomposed 3D assets in md string format")
    session_id: str | None = Field(None, description="Session ID for the LLM query")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class PlannerOutput(BaseModel):
    graph: str = Field(..., description="Relational bipartite graph representing the scene layout")
    session_id: str = Field(..., description="Session ID for the LLM query")

def plan(input_data: PlannerInput) -> PlannerOutput:
    """Plans the spatial and contextual layout of a 3D scene based on the user request and asset list."""
    query_prompt = _prompt + f"\n\nUser Request: {input_data.user_request}\n\nAsset List:\n{input_data.assets_str}\n\n"
    session_id = input_data.session_id or f"csp_1planner-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = helper.query(
        provider=input_data.provider,
        model=input_data.model,
        user_prompt=query_prompt,
        save_path=session_id,
        temperature=input_data.temperature,
    )
    graph = extract_relational_graph(response)
    return PlannerOutput(graph=graph, session_id=session_id)

def extract_relational_graph(response: str) -> str:
    """Extracts the relational bipartite graph from the response."""
    # Assuming the response is structured as described in the prompt
    lines = response.splitlines()
    graph_lines = []
    found_graph = False
    for line in lines:
        if line.startswith("### Relational Bipartite Graph"):
            found_graph = True
            continue
        if found_graph:
            graph_lines.append(line.strip())
    return "\n".join(graph_lines) if found_graph else response


if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_scene_desc, example_decomposer_output
    input_data = PlannerInput(
        user_request=example_scene_desc,
        assets_str=example_decomposer_output.formatted_asset_str,
        provider="ollama",
        model="gemma3"
    )
    output = plan(input_data)
    print(output.graph)



