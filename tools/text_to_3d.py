#!/usr/bin/env python3
"""
A script to generate a 3D model from a text description using the TRELLIS API.

This script sends a prompt to the TRELLIS Text-to-3D API, waits for the
generation to complete, and then downloads the resulting .glb file.

Usage:
    python generate_3d.py --prompt "a futuristic spaceship" --output "spaceship.glb"
"""

import requests
import time
import argparse
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from yaspin import yaspin

# --- Configuration ---
# You can adjust these default values
DEFAULT_API_URL = "http://localhost:8000"
# Timeout for the generation request in seconds (e.g., 10 minutes)
# Adjust this based on your hardware and the complexity of the models.
GENERATION_TIMEOUT = 600

def generate_3d_model(prompt: str, output_path: Path, api_url: str=DEFAULT_API_URL, verbose: bool=False):
    """
    Generates a 3D model from a text prompt and saves it to a file.

    Args:
        prompt (str): The text description of the model to generate.
        output_path (Path): The path to save the generated .glb file.
        api_url (str): The base URL of the TRELLIS API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    if verbose:
        print(f"🎨 Generating 3D model from: '{prompt}'")
        print(f"📍 Output location: {output_path.resolve()}")
        print(f"⏱️  Please wait, this may take several minutes...")
        print()

    # 1. --- Send Generation Request ---
    # This payload is based on the test script, optimized for speed.
    # We request a 'mesh' format, which should produce a .glb file.
    payload = {
        "prompt": prompt,
        "seed": 42,  # Using a fixed seed for reproducibility
        "formats": ["mesh"],
        "ss_steps": 12,  # A reasonable number of steps
        "slat_steps": 12,
        "generate_video": False,
        "texture_size": 1024 # A decent texture quality
    }

    generation_result = None
    try:
        start_time = time.time()
        if verbose:
            print("🚀 Sending request to API...")
        response = requests.post(
            f"{api_url}/generate",
            json=payload,
            timeout=GENERATION_TIMEOUT
        )
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        generation_result = response.json()
        end_time = time.time()

        if verbose:
            print("✅ Generation completed successfully!")
            print(f"📊 Job ID: {generation_result.get('job_id', 'N/A')}")
            print(f"📊 Status: {generation_result.get('status', 'Unknown')}")
            print(f"⏱️  API Time: {generation_result.get('generation_time_seconds', 0):.2f}s")
            print(f"⏱️  Total Time: {end_time - start_time:.2f}s")
            print(f"📁 Files Generated: {list(generation_result.get('files', {}).keys())}")
            print()


    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Generation timed out after {GENERATION_TIMEOUT} seconds.")
        print("💡 This can happen on slower systems or with complex prompts.")
        print("💡 Consider increasing the GENERATION_TIMEOUT value in the script.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: API request failed: {e}")
        sys.exit(1)

    # 2. --- Check for Success and Find File ---
    if not generation_result or generation_result.get('status') != 'success' or not generation_result.get('files'):
        print("❌ ERROR: API did not return a successful result or any files.")
        print("🔍 Response:", generation_result)
        sys.exit(1)

    # Find the .glb file in the response. The test script suggests the key is 'mesh'
    # and the value is the relative URL.
    mesh_file_url = None
    if 'mesh' in generation_result['files']:
         mesh_file_url = generation_result['files']['mesh']
    else:
        # Fallback to find any file ending with .glb
        for file_type, url in generation_result['files'].items():
            if url.endswith('.glb'):
                mesh_file_url = url
                break

    if not mesh_file_url:
        print("❌ ERROR: Could not find a .glb file in the API response.")
        sys.exit(1)

    # 3. --- Download the Generated File ---
    filename = mesh_file_url.split('/')[-1]
    job_id = generation_result['job_id']
    download_url = f"{api_url}/files/{job_id}/{filename}"

    if verbose:
        print(f"📥 Downloading generated file...")
        print(f"📁 Filename: {filename}")
        print(f"🔗 URL: {download_url}")
        print()

    try:
        file_response = requests.get(download_url, timeout=120)
        file_response.raise_for_status()

        content_length = len(file_response.content)
        if content_length == 0:
            print("❌ ERROR: Downloaded file is empty.")
            sys.exit(1)

        if verbose:
            print(f"📦 Downloaded {content_length / 1024:.2f} KB successfully!")

        # Save the file to the specified output path
        output_path.write_bytes(file_response.content)
        if verbose:
            print(f"✅ SUCCESS! Model saved to: {output_path.resolve()}")

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Failed to download the generated file: {e}")
        sys.exit(1)

class GenerationRequest(BaseModel):
    description: str = Field(..., description="Text description of the 3D model to generate.")
    glb_path: Path = Field(..., description="Path to save the generated .glb file.")


def generate_all_models(tasks: list[GenerationRequest], api_url: str=DEFAULT_API_URL, verbose: bool=False):
    """
    Generates multiple 3D models from a list of tasks.

    Args:
        tasks (list[GenerationRequest]): List of generation requests.
        api_url (str): The base URL of the TRELLIS API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"🎯 Task {i}/{len(tasks)}: {task.description}")
            print(f"{'='*50}")
        with yaspin(text=f"Generating model for task {i}/{len(tasks)}: {task.description}", color="green", timer=True) as spinner:
            generate_3d_model(task.description, task.glb_path, api_url, verbose)
            spinner.ok("✔")


def main():
    """Main function to parse arguments and run the generation."""
    parser = argparse.ArgumentParser(
        description="Generate a 3D model from text using the TRELLIS API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the 3D model to create."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output .glb file."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="Base URL of the TRELLIS API server."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress messages."
    )

    args = parser.parse_args()
    api_url = args.api_url.rstrip('/')

    # Ensure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generate_3d_model(args.prompt, args.output, api_url, args.verbose)


if __name__ == "__main__":
    main()
