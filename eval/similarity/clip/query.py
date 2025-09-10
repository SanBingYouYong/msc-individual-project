import requests
import os
import mimetypes # Import the library
from typing import List, Dict, Any

def query_similarity_service(
    text_prompt: str, 
    image_paths: List[str], 
    url: str = "http://localhost:8002/similarity"
) -> Dict[str, Any]:
    # ... (docstring is the same) ...
    payload = {"text": text_prompt}
    files_to_upload = []
    opened_files = []

    try:
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found at: {path}")
            
            file_handle = open(path, 'rb')
            opened_files.append(file_handle)
            
            filename = os.path.basename(path)
            
            # --- MODIFICATION START ---
            # Guess the MIME type of the file based on its extension
            content_type, _ = mimetypes.guess_type(path)
            if content_type is None:
                # Provide a default fallback if the type can't be guessed
                content_type = "application/octet-stream"
            
            # Use the full tuple format to explicitly provide the content type
            files_to_upload.append(("images", (filename, file_handle, content_type)))
            # --- MODIFICATION END ---

        # print(f"Sending {len(files_to_upload)} images to {url} with prompt: '{text_prompt}'")
        response = requests.post(url, data=payload, files=files_to_upload)
        response.raise_for_status()
        return response.json()
    
    finally:
        for f in opened_files:
            f.close()

if __name__ == "__main__":
    # Example usage
    text_prompt = "A chair."
    image_paths = [
        "chair.jpg",
        "number.png"
    ]
    
    try:
        results = query_similarity_service(text_prompt, image_paths)
        print("Similarity Results:", results)
    except Exception as e:
        print(f"Error querying similarity service: {e}")