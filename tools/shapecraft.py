#!/usr/bin/env python3
"""
A script to generate 3D models from text descriptions using the ShapeCraft API.

This script sends a shape description to the ShapeCraft API, waits for the
generation to complete, downloads the results, and converts the OBJ to GLB.

Usage:
    python shapecraft.py --prompt "a simple chair" --output "chair.glb"
"""

import requests
import time
import argparse
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from pydantic import BaseModel, Field
from yaspin import yaspin
import bpy

# --- Configuration ---
# You can adjust these default values
DEFAULT_API_URL = "http://localhost:8004"
# Timeout for the generation request in seconds (e.g., 30 minutes)
# Adjust this based on your hardware and the complexity of the models.
GENERATION_TIMEOUT = 1800

def convert_obj_to_glb(obj_path: Path, glb_path: Path, verbose: bool = False):
    """
    Converts an OBJ file to GLB using Blender's bpy module.
    
    Args:
        obj_path (Path): Path to the input OBJ file.
        glb_path (Path): Path to save the output GLB file.
        verbose (bool): Whether to print detailed progress messages.
    """
    if verbose:
        print(f"🔄 Converting {obj_path.name} to GLB format...")
    
    try:
        # read empty scene
        bpy.ops.wm.read_homefile(empty=True)  # Reset to default scene
        
        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=str(obj_path))
        
        # Export as GLB
        bpy.ops.export_scene.gltf(
            filepath=str(glb_path),
            export_format='GLB',
            use_selection=False
        )
        
        if verbose:
            print(f"✅ Successfully converted to GLB: {glb_path.name}")
            
    except Exception as e:
        print(f"❌ ERROR: Failed to convert OBJ to GLB: {e}")
        sys.exit(1)

def generate_shapecraft_model(prompt: str, output_path: Path, api_url: str = DEFAULT_API_URL, verbose: bool = False):
    """
    Generates a 3D model from a text prompt using ShapeCraft and saves it as GLB.

    Args:
        prompt (str): The text description of the model to generate.
        output_path (Path): The path to save the generated .glb file.
        api_url (str): The base URL of the ShapeCraft API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    if verbose:
        print(f"🎨 Generating 3D model from: '{prompt}'")
        print(f"📍 Output location: {output_path.resolve()}")
        print(f"⏱️  Please wait, this may take several minutes...")
        print()

    # 1. --- Send Generation Request ---
    payload = {
        "shape_description": prompt
    }

    generation_result = None
    try:
        start_time = time.time()
        if verbose:
            print("🚀 Sending request to API...")
        response = requests.post(
            f"{api_url}/generate_shape",
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
            print(f"⏱️  Total Time: {end_time - start_time:.2f}s")
            print()

    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Generation timed out after {GENERATION_TIMEOUT} seconds.")
        print("💡 This can happen on slower systems or with complex prompts.")
        print("💡 Consider increasing the GENERATION_TIMEOUT value in the script.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: API request failed: {e}")
        sys.exit(1)

    # 2. --- Check for Success and Get Job ID ---
    if not generation_result or not generation_result.get('job_id'):
        print("❌ ERROR: API did not return a job ID.")
        print("🔍 Response:", generation_result)
        sys.exit(1)

    job_id = generation_result['job_id']

    # 3. --- Download the Generated Files ---
    download_url = f"{api_url}/download/{job_id}"
    
    if verbose:
        print(f"📥 Downloading generated files...")
        print(f"🔗 URL: {download_url}")
        print()

    try:
        file_response = requests.get(download_url, timeout=300)
        file_response.raise_for_status()

        content_length = len(file_response.content)
        if content_length == 0:
            print("❌ ERROR: Downloaded file is empty.")
            sys.exit(1)

        if verbose:
            print(f"📦 Downloaded {content_length / 1024:.2f} KB successfully!")

        # 4. --- Extract and Process Files ---
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / f"{job_id}.zip"
            
            # Save the zip file
            zip_path.write_bytes(file_response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # Find the global-*.obj file
            obj_files = list(temp_path.glob("**/global-*.obj"))
            if not obj_files:
                print("❌ ERROR: Could not find global-*.obj file in the downloaded archive.")
                sys.exit(1)
            
            obj_path = obj_files[0]
            
            # Find the corresponding .py file (shape program)
            py_files = list(temp_path.glob("**/*.py"))
            py_file = py_files[0] if py_files else None
            
            if verbose:
                print(f"📁 Found OBJ file: {obj_path.name}")
                if py_file:
                    print(f"📁 Found shape program: {py_file.name}")
            
            # Convert OBJ to GLB
            convert_obj_to_glb(obj_path, output_path, verbose)
            
            # Save the shape program alongside the GLB file
            if py_file:
                py_output_path = output_path.with_suffix('.py')
                shutil.copy2(py_file, py_output_path)
                
                if verbose:
                    print(f"✅ Shape program saved to: {py_output_path}")

        if verbose:
            print(f"✅ SUCCESS! Model saved to: {output_path.resolve()}")

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Failed to download the generated files: {e}")
        sys.exit(1)
    except zipfile.BadZipFile as e:
        print(f"❌ ERROR: Downloaded file is not a valid zip archive: {e}")
        sys.exit(1)

class GenerationRequest(BaseModel):
    description: str = Field(..., description="Text description of the 3D model to generate.")
    glb_path: Path = Field(..., description="Path to save the generated .glb file.")

def generate_all_models(tasks: list[GenerationRequest], api_url: str = DEFAULT_API_URL, verbose: bool = False):
    """
    Generates multiple 3D models from a list of tasks.

    Args:
        tasks (list[GenerationRequest]): List of generation requests.
        api_url (str): The base URL of the ShapeCraft API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"🎯 Task {i}/{len(tasks)}: {task.description}")
            print(f"{'='*50}")
        with yaspin(text=f"Generating model for task {i}/{len(tasks)}: {task.description}", color="cyan", timer=True) as spinner:
            generate_shapecraft_model(task.description, task.glb_path, api_url, verbose)
            spinner.ok("✔")

def main():
    """Main function to parse arguments and run the generation."""
    parser = argparse.ArgumentParser(
        description="Generate a 3D model from text using the ShapeCraft API.",
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
        help="Base URL of the ShapeCraft API server."
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

    generate_shapecraft_model(args.prompt, args.output, api_url, args.verbose)

if __name__ == "__main__":
    main()
