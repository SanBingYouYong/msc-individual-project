import requests
import os
from pathlib import Path

def compare_obj_files(file1_path: Path, file2_path: Path) -> float:
    """
    Compares two OBJ files using an external service and returns the similarity score.

    Args:
        file1_path: The path to the first OBJ file.
        file2_path: The path to the second OBJ file.

    Returns:
        A float representing the cosine similarity score between the two files.
    
    Raises:
        requests.exceptions.RequestException: If the request to the service fails.
        KeyError: If the 'similarity' key is not in the response.
    """
    # Define the service endpoint URL
    url = "http://localhost:8003/compare_obj"
    
    # Check if the files exist
    if not file1_path.exists() or not file1_path.is_file():
        raise FileNotFoundError(f"File not found: {file1_path}")
    if not file2_path.exists() or not file2_path.is_file():
        raise FileNotFoundError(f"File not found: {file2_path}")

    # Prepare the files for the multipart/form-data request
    files = {
        'file1': (file1_path.name, open(file1_path, 'rb'), 'application/octet-stream'),
        'file2': (file2_path.name, open(file2_path, 'rb'), 'application/octet-stream')
    }
    
    try:
        # Make the POST request to the service
        response = requests.post(url, files=files, timeout=60)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        # Return the similarity score
        return result["similarity"]
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        raise
    except KeyError:
        print("The 'similarity' key was not found in the API response.")
        raise
    finally:
        # Close the opened files
        if 'file1' in files and files['file1'][1]:
            files['file1'][1].close()
        if 'file2' in files and files['file2'][1]:
            files['file2'][1].close()

# Example usage (you would replace these with your actual file paths)
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    try:
        # Call the function with Path objects for test1.obj and test2.obj
        similarity_score = compare_obj_files(script_dir / "scene_mesh-direct.obj", script_dir / "bbox-direct.obj")
        print(f"The similarity score is: {similarity_score}")
    except (FileNotFoundError, requests.exceptions.RequestException, KeyError) as e:
        print(f"Failed to get similarity score: {e}")
