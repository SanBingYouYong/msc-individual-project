import sys
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path to import modules
sys.path.append(os.getcwd())

from agents.error_fixer import fix_code, ErrorFixerInput
import re


from pydantic import BaseModel, Field

class ExecuteOrFixInput(BaseModel):
    code_snippet: str = Field(..., description="Code snippet to be executed")
    task_instruction: str = Field(..., description="Original task instruction that was used to generate the code snippet")
    working_dir: Path = Field(Path("temp"), description="Working directory to execute the code in; expected: job-folder/temp")
    max_iterations: int = Field(3, description="Maximum number of attempts to fix the code if errors occur")
    session_id: Optional[str] = Field(None, description="Session ID for the error fixing query, if applicable")
    provider: str = Field("google", description="LLM provider to use for the query")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for the query")
    temperature: float = Field(0, description="Temperature for the LLM response generation")

class ExecuteOrFixOutput(BaseModel):
    final_code: str = Field(..., description="Final version of the code after execution and possible fixes")
    is_success: bool = Field(..., description="Whether the code executed successfully without errors")
    session_id: Optional[str] = Field(None, description="Session ID for the error fixing query, if applicable")
    iterations_used: int = Field(..., description="Number of iterations used for error fixing (0 if successful on first try)")

def execute_code(eof_in: ExecuteOrFixInput) -> ExecuteOrFixOutput:
    """
    Execute the provided code snippet and fix it if errors occur - all the error fixing happens in the same session, with initial chat history created from input data.
    
    Args:
        eof_in: ExecuteOrFixInput containing the code snippet, task instruction, and configuration
        
    Returns:
        ExecuteOrFixOutput with final code, success status, and session ID
    """
    # Create a unique session ID if not provided, for error fixer agent to use through-out
    error_fixer_session_id = eof_in.session_id or f"execute_code-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    current_code = eof_in.code_snippet
    
    # Create initial chat history with task instruction as user message and code as assistant response
    chat_history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": eof_in.task_instruction}]
        },
        {
            "role": "assistant",
            "content": f"```python\n{eof_in.code_snippet}\n```"  # include the code snippet in the chat history
        }
    ]
    
    for attempt in range(eof_in.max_iterations):
        print(f"Code execution attempt {attempt + 1}/{eof_in.max_iterations}")
        
        # Create temporary file for the code
        # Ensure the working directory exists
        os.makedirs(eof_in.working_dir, exist_ok=True)
        # Create temporary file for the code in the specified working directory
        with tempfile.NamedTemporaryFile(mode='w', prefix='_eof-', suffix='.py', dir=eof_in.working_dir, delete=False) as temp_file:
            temp_file.write(current_code)
            code_path = temp_file.name
        
        try:
            # Try to execute the code
            result = subprocess.run(
                [sys.executable, code_path], 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=60  # 60 second timeout
            )
            # Clean up temp file
            # os.unlink(code_path)
            return ExecuteOrFixOutput(
                final_code=current_code,
                is_success=True,
                session_id=error_fixer_session_id if attempt > 0 else None,
                iterations_used=attempt
            )
            
        except subprocess.CalledProcessError as e:
            # Extract and format error message
            error_message = _format_error_message(e)
            
            # Clean up temp file
            # os.unlink(code_path)
            
            # If this is the last attempt, return the current code with failure status
            if attempt == eof_in.max_iterations - 1:
                print(f"Maximum attempts ({eof_in.max_iterations}) reached. Returning last version with unhandled error message.")
                return ExecuteOrFixOutput(
                    final_code=current_code,
                    is_success=False,
                    session_id=error_fixer_session_id if attempt > 0 else None,
                    iterations_used=attempt + 1
                )
            
            # Try to fix the code using error_fixer agent
            print(f"Attempting to fix error in code: {error_message}")
            try:
                error_fixer_input = ErrorFixerInput(
                    context_provider_history=chat_history,
                    error_message=error_message,
                    attempt_number=attempt + 1,
                    session_id=error_fixer_session_id,
                    provider=eof_in.provider,
                    model=eof_in.model,
                    temperature=eof_in.temperature
                )
                
                fix_result = fix_code(error_fixer_input)
                current_code = fix_result.fixed_code
                error_fixer_session_id = fix_result.session_id
                
            except Exception as fix_error:
                print(f"Error while trying to fix code: {fix_error}")
                return ExecuteOrFixOutput(
                    final_code=current_code,
                    is_success=False,
                    session_id=error_fixer_session_id,
                    iterations_used=attempt + 1
                )
                
        except subprocess.TimeoutExpired:
            print("Code execution timed out (60 seconds).")
            # os.unlink(code_path)
            
            if attempt == eof_in.max_iterations - 1:
                return ExecuteOrFixOutput(
                    final_code=current_code,
                    is_success=False,
                    session_id=error_fixer_session_id,
                    iterations_used=attempt + 1
                )
                
            # Try to fix timeout issue using error_fixer agent
            error_message = "Code execution timed out after 60 seconds. This might be due to infinite loops, blocking operations, or computationally expensive operations."
            try:
                error_fixer_input = ErrorFixerInput(
                    context_provider_history=chat_history,
                    error_message=error_message,
                    attempt_number=attempt + 1,
                    session_id=error_fixer_session_id,
                    provider=eof_in.provider,
                    model=eof_in.model,
                    temperature=eof_in.temperature
                )
                
                fix_result = fix_code(error_fixer_input)
                current_code = fix_result.fixed_code
                error_fixer_session_id = fix_result.session_id
                
            except Exception as fix_error:
                print(f"Error while trying to fix timeout: {fix_error}")
                return ExecuteOrFixOutput(
                    final_code=current_code,
                    is_success=False,
                    session_id=error_fixer_session_id,
                    iterations_used=attempt + 1
                )
                
        except FileNotFoundError:
            print(f"Python interpreter not found at: {sys.executable}")
            return ExecuteOrFixOutput(
                final_code=current_code,
                is_success=False,
                session_id=None,
                iterations_used=attempt
            )
        except Exception as e:
            print(f"Unexpected error during execution: {e}")
            # os.unlink(code_path)
            return ExecuteOrFixOutput(
                final_code=current_code,
                is_success=False,
                session_id=None,
                iterations_used=attempt
            )
    
    # Should not reach here, but just in case
    return ExecuteOrFixOutput(
        final_code=current_code,
        is_success=False,
        session_id=error_fixer_session_id,
        iterations_used=eof_in.max_iterations
    )

def _format_error_message(error: subprocess.CalledProcessError) -> str:
    """Format error message from subprocess to be more readable and focused."""
    if error.stderr:
        lines = error.stderr.splitlines()
        formatted_lines = []
        
        # Look for the error indicator line with ^^^
        for i in range(len(lines) - 1, -1, -1):
            if "^^^" in lines[i]:
                # Include some context around the error
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                for line in lines[start:end]:
                    # Keep only file base name and line number for readability
                    line = re.sub(r"File '([^']+)', line (\d+)", lambda m: f"{os.path.basename(m.group(1))}, {m.group(2)}", line)
                    formatted_lines.append(line)
                break
        
        # If no ^^^ found, just return the last few lines of stderr
        if not formatted_lines:
            formatted_lines = lines[-10:] if len(lines) > 10 else lines
            
        return "\n".join(formatted_lines)
    else:
        return "No detailed error message available."

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample code snippet that has intentional errors
    test_code_with_error = '''
import os
import json
from typing import Tuple

class Layout:
    def __init__(self, location: Tuple[float, float, float], min: Tuple[float, float, float], 
                 max: Tuple[float, float, float], orientation: Tuple[float, float, float]):
        self.location = location
        self.min = min
        self.max = max
        self.orientation = orientation

def main():
    # This will cause a NameError because 'undefined_variable' is not defined
    print(undefined_variable)
    
    # Save layout (this won't be reached due to error above)
    layout_data = {
        "box1": {
            "location": [0, 0, 0],
            "min": [-1, -1, -1],
            "max": [1, 1, 1],
            "orientation": [0, 0, 0]
        }
    }
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    layout_path = os.path.join(script_dir, "layout.json")
    
    with open(layout_path, "w") as f:
        json.dump(layout_data, f, indent=2)
    
    print(f"Layout saved to {layout_path}")

if __name__ == "__main__":
    main()
'''
    
    # Create test input
    test_input = ExecuteOrFixInput(
        code_snippet=test_code_with_error,
        task_instruction="Please always write bugged code to test this error fixing pipeline.",
        working_dir=Path("temp"),
        max_iterations=5,
        provider='ollama',
        model='qwen3',
    )
    
    print("Testing execute_or_fix pipeline...")
    result = execute_code(test_input)
    
    print(f"\nFinal result:")
    print(f"Success: {result.is_success}")
    print(f"Final code length: {len(result.final_code)} characters")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Error fixer session ID: {result.session_id}")
    
    if result.is_success:
        print("Pipeline successfully fixed and executed the code!")
    else:
        print("Pipeline could not fix the code within the maximum iterations.")
