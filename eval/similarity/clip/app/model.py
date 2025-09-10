import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union

# Determine the device to use (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CLIP model is using device: {device}")

# Specify the pre-trained CLIP model to use
MODEL_NAME = "openai/clip-vit-base-patch32"

# Load the model and processor from Hugging Face
try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("CLIP model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    model = None
    processor = None

def compute_similarity(
    images: Union[Image.Image, List[Image.Image]],
    text: str
) -> Union[float, List[float]]:
    """
    Computes similarity score(s) between image(s) and a text prompt.
    This function handles both a single image and a list (batch) of images.

    Args:
        images (Union[Image.Image, List[Image.Image]]): A single Pillow Image or a list of Pillow Images.
        text (str): The input text prompt.

    Returns:
        Union[float, List[float]]: A single similarity score or a list of scores.
    """
    if model is None or processor is None:
        raise RuntimeError("Model is not available.")

    # The processor can handle both a single image and a list of images
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    ).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # The logits_per_image gives the similarity score(s)
    logits_per_image = outputs.logits_per_image.squeeze()

    # .tolist() correctly converts a 0-dim tensor to a float
    # and a 1-dim tensor to a list of floats.
    return logits_per_image.tolist()