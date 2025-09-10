import sys
import os
import subprocess
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.getcwd())

from llms.llms import helper
from agents.error_fixer import fix_code, ErrorFixerInput
from agents.decomposer import DecomposerOutput
from pipelines.csp.code_execution import execute_code, ExecuteOrFixInput

from pydantic import BaseModel, Field
from typing import List

class LayoutVerificationInput(BaseModel):
    code: str = Field(..., description="Code to be executed for layout verification")
    decomposer_output: DecomposerOutput = Field(..., description="Output from the Decomposer agent containing asset information")
    original_task_instruction: str = Field(..., description="Original task instruction for context")
    working_dir: Path = Field(Path("temp"), description="Working directory to execute the code in; expected: job-folder/temp")
    max_iterations: int = Field(3, description="Maximum number of attempts to fix the code if errors occur")
    layout_fix_session_id: Optional[str] = Field(None, description="Session ID for the error fixing query that aims to fix an invalid layout, if applicable")
    error_fixing_max_iter: int = Field(3, description="parameter to pass to code execution pipeline to limit error fixing rounds")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class LayoutVerificationOutput(BaseModel):
    is_success: bool = Field(..., description="Whether the code executed successfully and passed verification")
    final_code: str = Field(..., description="Final version of the code after execution and possible fixes")
    verified_layout: Optional[Dict[str, Any]] = Field(None, description="Verified layout dictionary if successful")
    iterations_used: int = Field(..., description="Number of iterations used for fixing invalid layout (0 if successful on first try)")
    total_error_fixes: int = Field(..., description="Total number of error fixes applied during the process")

def layout_verification(input_data: LayoutVerificationInput) -> LayoutVerificationOutput:
    """
    Execute code and verify layout output, fixing issues if necessary.
    
    Args:
        input_data: LayoutVerificationInput containing code, decomposer output, and configuration
        
    Returns:
        LayoutVerificationOutput with verification results
    """
    layout_fix_session_id = input_data.layout_fix_session_id or f"layout_verification-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    current_code = input_data.code
    
    # Initialize chat history with original task instruction and code
    chat_history = [
        {
            "role": "user", 
            "content": [{"type": "text", "text": input_data.original_task_instruction}]
        }
    ]
    
    layout_verification_iterations = 0
    total_error_fixes = 0
    
    for iteration in range(input_data.max_iterations):
        print(f"Layout verification attempt {iteration + 1}/{input_data.max_iterations}")
        
        # Step 1: Execute the code using code_execution pipeline
        execute_input = ExecuteOrFixInput(
            code_snippet=current_code,
            task_instruction=input_data.original_task_instruction,
            working_dir=input_data.working_dir,
            max_iterations=input_data.error_fixing_max_iter,
            session_id=None,  # Let code execution create its own session
            provider=input_data.provider,
            model=input_data.model,
            temperature=input_data.temperature
        )
        
        execute_result = execute_code(execute_input)
        current_code = execute_result.final_code
        total_error_fixes += execute_result.iterations_used
        
        # Update chat history with executed code
        chat_history.append(
            {
                "role": "assistant",
                "content": f"```python\n{current_code}\n```"  # include the updated code in the chat history
            }  # let ErrorFixer add error message on behalf of the user later
        )
        # save history (overwrite it everytime with new executed code)
        # side effect: even if first run is good, we will still produce this chat history, but is fine, comparing to what to save in later stages... 
        helper._save_history(
            save_path=layout_fix_session_id,
            history_to_save=chat_history,
        )
        
        # If code execution failed, return failure
        if not execute_result.is_success:
            print("Code execution failed even after error fixing attempts.")
            return LayoutVerificationOutput(
                is_success=False,
                final_code=current_code,
                verified_layout=None,
                iterations_used=layout_verification_iterations,
                total_error_fixes=total_error_fixes
            )
        
        print("Code executed, verifying layout output...")
        
        # Step 2: Execute the code as a Python script to verify layout.json output
        # TODO: maybe we can try copying the code generated layout.json directly to avoid running the CSP again
        verification_result = _verify_layout_output(current_code, input_data.working_dir, input_data.decomposer_output)
        
        if verification_result["is_valid"]:
            print("Layout verification successful!")
            return LayoutVerificationOutput(
                is_success=True,
                final_code=current_code,
                verified_layout=verification_result["layout"],
                iterations_used=layout_verification_iterations,
                total_error_fixes=total_error_fixes
            )
        
        # Step 3: Layout verification failed, increment iteration counter
        layout_verification_iterations += 1
        
        # If this is the last iteration, return failure
        if iteration == input_data.max_iterations - 1:
            print(f"Maximum layout verification attempts ({input_data.max_iterations}) reached.")
            return LayoutVerificationOutput(
                is_success=False,
                final_code=current_code,
                verified_layout=None,
                iterations_used=layout_verification_iterations,
                total_error_fixes=total_error_fixes
            )
        
        # Step 4: Use ErrorFixer to fix layout issues
        print("Layout verification failed. Attempting to fix...")
        error_message = verification_result["error_message"]
        
        try:
            error_fixer_input = ErrorFixerInput(
                context_provider_history=chat_history,  # altho it supports incremental history, but here we need to tweak the history with updated code, so we maintain a copy ourselves, and claim it to be new attempts everytime
                error_message=error_message,
                attempt_number=1,
                session_id=None,  # by using the same error fixer session ID, we can overwrite it everytime with newer modified history  # UPDATE: no it overwrites. we have modified history so we have to tell it it's a new one everytime and manully maintain a chat history
                provider=input_data.provider,
                model=input_data.model,
                temperature=input_data.temperature
            )
            
            fix_result = fix_code(error_fixer_input)
            current_code = fix_result.fixed_code
            
            # Add only a fake user message containing only the error message
            chat_history.append({
                "role": "user",
                "content": error_message
            })
            
            print("Obtained updated code for layout issues. Retrying verification...")
            
        except Exception as fix_error:
            print(f"Error while trying to fix layout issues: {fix_error}")
            return LayoutVerificationOutput(
                is_success=False,
                final_code=current_code,
                verified_layout=None,
                iterations_used=layout_verification_iterations,
                total_error_fixes=total_error_fixes
            )
    
    # Should not reach here, but just in case
    return LayoutVerificationOutput(
        is_success=False,
        final_code=current_code,
        verified_layout=None,
        iterations_used=layout_verification_iterations,
        total_error_fixes=total_error_fixes
    )

def _verify_layout_output(code: str, working_dir: Path, decomposer_output: DecomposerOutput) -> Dict[str, Any]:
    """
    Execute code as Python script and verify the layout.json output.
    
    Args:
        code: Python code to execute
        working_dir: Working directory to execute the code in
        decomposer_output: Output from decomposer containing expected assets
        
    Returns:
        Dictionary with verification results
    """
    
    # Ensure working directory exists
    os.makedirs(working_dir, exist_ok=True)
    
    # Create temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', prefix='_layout_verify-', suffix='.py', 
                                   dir=working_dir, delete=False) as temp_file:
        temp_file.write(code)
        code_path = temp_file.name
    
    try:
        # Execute the code
        result = subprocess.run(
            [sys.executable, code_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Check if layout.json exists next to the script
        code_dir = os.path.dirname(code_path)
        layout_path = os.path.join(code_dir, "layout.json")
        
        if not os.path.exists(layout_path):
            return {
                "is_valid": False,
                "error_message": f"Layout verification failed: layout.json file was not created next to the script. Expected path: {layout_path}. Please ensure your code creates a layout.json file in the same directory as the script."
            }
        
        # Load and validate layout.json
        try:
            with open(layout_path, 'r') as f:
                layout_data = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "is_valid": False,
                "error_message": f"Layout verification failed: layout.json is not valid JSON. Error: {e}. Please ensure the layout.json file contains valid JSON data."
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"Layout verification failed: Could not read layout.json. Error: {e}. Please ensure the layout.json file is readable and properly formatted."
            }
        
        # Validate layout structure
        validation_result = _validate_layout_structure(layout_data, decomposer_output)
        
        if validation_result["is_valid"]:
            return {
                "is_valid": True,
                "layout": layout_data
            }
        else:
            return {
                "is_valid": False,
                "error_message": validation_result["error_message"]
            }
            
    except subprocess.CalledProcessError as e:
        error_details = e.stderr if e.stderr else "No error details available"
        return {
            "is_valid": False,
            "error_message": f"Layout verification failed: Code execution error during verification. {error_details}. Please fix the code execution issues."
        }
    except subprocess.TimeoutExpired:
        return {
            "is_valid": False,
            "error_message": "Layout verification failed: Code execution timed out during verification. Please optimize your code to run within 60 seconds."
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_message": f"Layout verification failed: Unexpected error during verification: {e}. Please review your code for potential issues."
        }
    finally:
        # Clean up temporary file
        if os.path.exists(code_path):
            os.unlink(code_path)

def _validate_layout_structure(layout_data: Dict[str, Any], decomposer_output: DecomposerOutput) -> Dict[str, Any]:
    """
    Validate that the layout.json has the expected structure and contains required assets.
    
    Args:
        layout_data: Loaded layout JSON data
        decomposer_output: Output from decomposer containing expected assets
        
    Returns:
        Dictionary with validation results
    """
    # Check if layout_data is a dictionary
    if not isinstance(layout_data, dict):
        return {
            "is_valid": False,
            "error_message": "Layout verification failed: layout.json must be a dictionary (object) at the top level, not a list or other type."
        }
    
    # Get expected asset names from decomposer output
    expected_assets = set(asset.name for asset in decomposer_output.assets)
    layout_assets = set(layout_data.keys())
    
    # Check for missing assets
    missing_assets = expected_assets - layout_assets
    if missing_assets:
        return {
            "is_valid": False,
            "error_message": f"Layout verification failed: Missing assets in layout.json: {', '.join(missing_assets)}. Expected all assets from decomposer output: {', '.join(expected_assets)}"
        }
    
    # Validate each asset's layout parameters
    required_fields = {"location", "min", "max", "orientation"}
    
    for asset_name, asset_layout in layout_data.items():
        # Check if asset layout is a dictionary
        if not isinstance(asset_layout, dict):
            return {
                "is_valid": False,
                "error_message": f"Layout verification failed: Asset '{asset_name}' layout must be a dictionary, not {type(asset_layout).__name__}."
            }
        
        # Check for required fields
        missing_fields = required_fields - set(asset_layout.keys())
        if missing_fields:
            return {
                "is_valid": False,
                "error_message": f"Layout verification failed: Asset '{asset_name}' is missing required fields: {', '.join(missing_fields)}. Each asset must have: location, min, max, orientation."
            }
        
        # Validate field types and values
        for field in required_fields:
            field_value = asset_layout[field]
            
            # Each field should be a list/array of 3 numbers
            if not isinstance(field_value, (list, tuple)):
                return {
                    "is_valid": False,
                    "error_message": f"Layout verification failed: Asset '{asset_name}' field '{field}' must be a list/array, not {type(field_value).__name__}."
                }
            
            if len(field_value) != 3:
                return {
                    "is_valid": False,
                    "error_message": f"Layout verification failed: Asset '{asset_name}' field '{field}' must have exactly 3 values (x, y, z), got {len(field_value)} values."
                }
            
            # Check if all values are numbers
            for i, val in enumerate(field_value):
                if not isinstance(val, (int, float)):
                    return {
                        "is_valid": False,
                        "error_message": f"Layout verification failed: Asset '{asset_name}' field '{field}' value at index {i} must be a number, got {type(val).__name__}."
                    }
    
    return {"is_valid": True}

# Example usage and testing
if __name__ == "__main__":
    from agents.decomposer import DecomposedAsset
    
    # Create a mock decomposer output for testing
    mock_assets = [
        DecomposedAsset(name="Fireplace", description="A fireplace", location="center"),
        DecomposedAsset(name="Sofa", description="A sofa", location="living area"),
        DecomposedAsset(name="Coffee Table", description="A coffee table", location="in front of sofa")
    ]
    
    mock_decomposer_output = DecomposerOutput(
        assets=mock_assets,
        formatted_asset_str="test assets",
        session_id="test-session"
    )
    
    # Test code that should work
    test_code_good = '''
import os
import json

def main():
    # Generate layout data
    layout_data = {
        "Fireplace": {
            "location": [0.40892755233689043, 0.6, -0.2769126440651295],
            "min": [-0.34107244766310957, 0.0, -0.5269126440651295],
            "max": [1.1589275523368905, 1.2, -0.0269126440651295],
            "orientation": [0.0, -0.5064501215422398, 0.0]
        },
        "Sofa": {
            "location": [3.4482267086549316, 0.4, -4.74612205929713],
            "min": [2.4482267086549316, 0.0, -5.19612205929713],
            "max": [4.448226708654932, 0.8, -4.296122059297129],
            "orientation": [0.0, -3.135223839957194, 0.0]
        },
        "Coffee Table": {
            "location": [-0.7663731235021038, 0.2, -3.763695006312834],
            "min": [-1.2663731235021038, 0.0, -4.013695006312834],
            "max": [-0.26637312350210385, 0.4, -3.513695006312834],
            "orientation": [0.0, 1.420608244407495, 0.0]
        }
    }
    
    # Save layout to file next to script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    layout_path = os.path.join(script_dir, "layout.json")
    
    with open(layout_path, "w") as f:
        json.dump(layout_data, f, indent=2)
    
    print(f"Layout saved to {layout_path}")

if __name__ == "__main__":
    main()
'''
    
    # Test the pipeline
    test_input = LayoutVerificationInput(
        code=test_code_good,
        decomposer_output=mock_decomposer_output,
        original_task_instruction="Create a layout for a living room with fireplace, sofa, and coffee table.",
        working_dir=Path("temp"),
        max_iterations=3,
        error_fixing_max_iter=2,
        provider="google",
        model="gemini-2.5-flash-lite",
        temperature=0
    )
    
    print("Testing layout verification pipeline...")
    result = layout_verification(test_input)
    
    print(f"\nResult:")
    print(f"Success: {result.is_success}")
    print(f"Layout verification iterations: {result.iterations_used}")
    print(f"Total error fixes: {result.total_error_fixes}")
    print(f"Final code length: {len(result.final_code)} characters")
    
    if result.verified_layout:
        print(f"Verified layout contains {len(result.verified_layout)} assets:")
        for asset_name in result.verified_layout.keys():
            print(f"  - {asset_name}")
    
    if result.is_success:
        print("Pipeline successfully verified the layout!")
    else:
        print("Pipeline could not verify the layout within maximum iterations.")

