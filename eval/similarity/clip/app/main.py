import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from typing import List
from .model import compute_similarity, model

# Initialize the FastAPI app
app = FastAPI(title="CLIP Similarity API")

@app.on_event("startup")
async def startup_event():
    """Check if the model is loaded at startup."""
    if model is None:
        raise RuntimeError("Failed to load CLIP model. Check logs for details.")
    print("Application startup complete. Model is ready.")


@app.get("/")
def read_root():
    """Root endpoint to check if the service is running."""
    return {"status": "CLIP similarity service is running."}


@app.post("/similarity")
async def get_similarity_batch(
    text: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Endpoint to compute similarity between one or more images and a single text prompt.
    
    Accepts a multipart/form-data request with a 'text' field and one or more 'images' fields.
    """
    pil_images = []
    filenames = []

    # Process all uploaded image files
    for image_file in images:
        # if not image_file.content_type.startswith("image/"):
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail=f"File '{image_file.filename}' is not a valid image."
            )
        try:
            # Read image bytes, open with Pillow, and add to list
            image_bytes = await image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
            filenames.append(image_file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing file {image_file.filename}: {str(e)}"
            )

    if not pil_images:
        raise HTTPException(status_code=400, detail="No valid images were provided.")

    try:
        # Compute scores in a single batch for maximum efficiency
        scores = compute_similarity(pil_images, text)
        
        # Ensure scores is always a list for consistent zipping
        if not isinstance(scores, list):
            scores = [scores]

        # Combine filenames with their corresponding scores
        results = [
            {"filename": name, "similarity_score": score}
            for name, score in zip(filenames, scores)
        ]

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during similarity computation: {str(e)}"
        )