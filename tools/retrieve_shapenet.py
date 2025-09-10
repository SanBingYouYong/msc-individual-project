#!/usr/bin/env python3
"""
A script to retrieve 3D models from a locally hosted ShapeNet service.

This script searches for 3D models by text description and downloads the
best matching .glb files from a ShapeNet database.

Usage:
    python retrieve_shapenet.py --query "a modern chair" --output "chair.glb"
"""

import requests
import argparse
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from yaspin import yaspin

# --- Configuration ---
# You can adjust these default values
DEFAULT_API_URL = "http://localhost:8001"
# Timeout for requests in seconds
REQUEST_TIMEOUT = 120

def retrieve_shapenet_model(query: str, output_path: Path, api_url: str = DEFAULT_API_URL, verbose: bool = False):
    """
    Retrieves a 3D model from ShapeNet by text query and saves it to a file.

    Args:
        query (str): The text description to search for.
        output_path (Path): The path to save the retrieved .glb file.
        api_url (str): The base URL of the ShapeNet API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    if verbose:
        print(f"🔍 Searching ShapeNet for: '{query}'")
        print(f"📍 Output location: {output_path.resolve()}")
        print()

    try:
        if verbose:
            print("🚀 Sending search request to API...")
        
        # Search and download GLB in one request
        response = requests.get(
            f"{api_url}/search-glb",
            params={"query": query, "limit": 1},
            timeout=REQUEST_TIMEOUT
        )
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Check if we got content
        content_length = len(response.content)
        if content_length == 0:
            print("❌ ERROR: No model found or downloaded file is empty.")
            sys.exit(1)

        # Extract metadata from headers
        shape_id = response.headers.get('X-Shape-ID', 'Unknown')
        score = response.headers.get('X-Score', 'Unknown')
        
        if verbose:
            print("✅ Model retrieved successfully!")
            print(f"📊 Shape ID: {shape_id}")
            print(f"📊 Match Score: {score}")
            print(f"📦 Downloaded {content_length / 1024:.2f} KB")
            print()

        # Save the file to the specified output path
        output_path.write_bytes(response.content)
        
        if verbose:
            print(f"✅ SUCCESS! Model saved to: {output_path.resolve()}")

    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Request timed out after {REQUEST_TIMEOUT} seconds.")
        print("💡 The ShapeNet service might be slow or unavailable.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: API request failed: {e}")
        sys.exit(1)

class RetrievalRequest(BaseModel):
    query: str = Field(..., description="Text query to search for in ShapeNet.")
    glb_path: Path = Field(..., description="Path to save the retrieved .glb file.")

def retrieve_all_models(tasks: list[RetrievalRequest], api_url: str = DEFAULT_API_URL, verbose: bool = False):
    """
    Retrieves multiple 3D models from a list of tasks.

    Args:
        tasks (list[RetrievalRequest]): List of retrieval requests.
        api_url (str): The base URL of the ShapeNet API.
        verbose (bool): Whether to print detailed progress messages. Default is False.
    """
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"🎯 Task {i}/{len(tasks)}: {task.query}")
            print(f"{'='*50}")
        
        with yaspin(text=f"Retrieving model for task {i}/{len(tasks)}: {task.query}", color="blue", timer=True) as spinner:
            retrieve_shapenet_model(task.query, task.glb_path, api_url, verbose)
            spinner.ok("✔")

def main():
    """Main function to parse arguments and run the retrieval."""
    parser = argparse.ArgumentParser(
        description="Retrieve a 3D model from ShapeNet by text query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text description to search for in ShapeNet."
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
        help="Base URL of the ShapeNet API server."
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

    retrieve_shapenet_model(args.query, args.output, api_url, args.verbose)

if __name__ == "__main__":
    main()
