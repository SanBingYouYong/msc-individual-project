import sys
import os
sys.path.append(os.getcwd())
import pickle
from pydantic import BaseModel
from typing import Union
from pathlib import Path

def save_pydantic_model_as_pkl(model: BaseModel, path: Union[str, Path]) -> Path:
    """
    Dumps a Pydantic BaseModel instance to a pickle file at the specified path.

    Args:
        model (BaseModel): The Pydantic model instance.
        path (Union[str, Path]): The file path to save the pickle file.
    """
    data = model.model_dump()
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path

def load_pydantic_model_from_pkl(model_class: BaseModel, path: Union[str, Path]) -> BaseModel:
    """
    Loads a Pydantic BaseModel instance from a pickle file at the specified path.

    Args:
        model_class (BaseModel): The Pydantic model class to instantiate.
        path (Union[str, Path]): The file path to load the pickle file from.

    Returns:
        BaseModel: An instance of the Pydantic model class.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return model_class.model_validate(data)

# trying to suppress bpy outputs that flush to console: https://blender.stackexchange.com/a/270201
from contextlib import contextmanager

# this does not work with import and blend file operations... but works for rendering
@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class QueryConfig(BaseModel):
    """Configuration for LLM queries."""
    provider: str = Field("google", description="LLM provider to use for queries")
    model: str = Field("gemini-2.5-flash-lite", description="LLM model to use for queries")
    temperature: float = Field(0, description="Temperature for LLM response generation")

if __name__ == '__main__':
    from test_data.examples import example_inner_loop_init_output
    example_path = Path("test_data/example_inner_loop_init_output.pkl")
    save_pydantic_model_as_pkl(example_inner_loop_init_output, example_path)
    print(f"Saved example InnerLoopInitOutput to {example_path}")
    from pipelines.csp.csp_inner_loop_init import InnerLoopInitOutput
    loaded_output = load_pydantic_model_from_pkl(
        InnerLoopInitOutput, example_path
    )
    print(f"Loaded InnerLoopInitOutput: {loaded_output.model_dump().keys()}")

