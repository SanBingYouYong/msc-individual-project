import os
import sys
sys.path.append(os.getcwd())

from llms.llms import helper
from datetime import datetime
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, Field

class ReviewerInput(BaseModel):
    original_task_instruction: str = Field(..., description="Original task instruction provided by the user")
    code_executed: str = Field(..., description="The code that has been executed by the coder agent, to be concat as chat history context")
    image_paths: list[Path] = Field(..., description="Paths to the rendered images of the layout from different axes (x, y, z)")
    session_id: str | None = Field(None, description="Session ID for the review process, if not provided a new one will be generated")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class ReviewerOutput(BaseModel):
    updated_code: str = Field(..., description="Updated code for the scoring functions based on the review")
    satisfactory: bool = Field(..., description="Indicates if the layout is satisfactory or not")
    session_id: str = Field(..., description="Session ID for the review process")


# we supply a manual history with the final code executed as assistant response.
_prompt = """
Your code has been successfully executed and the optimal layout found under the constraints are provided below, also visualized in the attached image renderings. Please think step-by-step to identify any constraints that are lacking or not correctly satisfied, and revise the code to improve the scoring functions for the relations. 

If the python code needs to be updated, please provide it again in full in a single markdown code block, ensuring that it can be run directly:
```python
# your updated code
# if no problems are found and the layout is satisfactory, simply include one line of comment "# layout is good" in this code block
```
"""

def review(input_data: ReviewerInput) -> ReviewerOutput:
    """Reviews the executed code and rendered layout, providing updated code if improvements are needed."""
    # Create the query prompt with the original task and code context
    # query_prompt = _prompt + f"\n\nOriginal Task: {input_data.original_task_instruction}\n\nExecuted Code:\n```python\n{input_data.code_executed}\n```\n\nPlease review the layout based on the rendered images and provide improvements if needed."
    history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_data.original_task_instruction
                }
            ]
        },
        {
            "role": "assistant",
            "content": f"```python\n{input_data.code_executed}\n```"
        }
    ]
    
    # Generate session ID if not provided
    session_id = input_data.session_id or f"csp_3reviewer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Convert Path objects to strings for image paths
    image_path_strings = [str(path.resolve()) for path in input_data.image_paths]
    
    # Query the LLM with the rendered images
    response = helper.query(
        provider=input_data.provider,
        model=input_data.model,
        user_prompt=_prompt,
        history_messages=history,
        image_paths=image_path_strings,
        save_path=session_id,
        temperature=input_data.temperature,
    )
    
    # Extract updated code from the response
    updated_code, satisfactory = extract_updated_code(response)
    
    return ReviewerOutput(updated_code=updated_code,  # note: when satisfactory, this should be discarded
                          satisfactory=satisfactory,
                          session_id=session_id)

def extract_updated_code(response: str) -> Tuple[str, bool]:
    """Extracts the updated code from the LLM response, also checks for the layout satisfaction."""
    try:
        extracted_code = helper.extract_code_block(response, 'python')
        satisfactory = "layout is good" in extracted_code
        return extracted_code, satisfactory
    except Exception as e:
        # check for the phrase "layout is good" in the response
        print(f"Warning: Could not extract code block from LLM response: {e}")
        if "layout is good" in response:
            return "# layout is good", True
        else:
            print("Using full response as code (this might not work)...")
            return response, False

if __name__ == '__main__':
    # Example usage
    from test_data.examples import example_coder_query, example_code, example_render_output
    input_data = ReviewerInput(
        original_task_instruction=example_coder_query,
        code_executed=example_code,
        image_paths=example_render_output.rendered_images,
        provider="google",
        model="gemini-2.0-flash-lite",
        temperature=0
    )
    output = review(input_data)
    from rich import print
    print(output.model_dump())
