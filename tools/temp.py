#!/usr/bin/env python3
"""
Test script for group retrieval functionality of retrieve_shapenet.py
"""

import sys
from pathlib import Path

# Add the tools directory to the path so we can import retrieve_shapenet
sys.path.append(str(Path(__file__).parent))

from retrieve_shapenet import retrieve_all_models, RetrievalRequest

def test_group_retrieval():
    """Test the batch retrieval functionality."""
    
    # Create output directory
    output_dir = Path("ttt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define a list of retrieval tasks
    tasks = [
        RetrievalRequest(
            query="a modern chair",
            glb_path=output_dir / "chair.glb"
        ),
        RetrievalRequest(
            query="wooden table",
            glb_path=output_dir / "table.glb"
        ),
        RetrievalRequest(
            query="office desk",
            glb_path=output_dir / "desk.glb"
        ),
        RetrievalRequest(
            query="bookshelf",
            glb_path=output_dir / "bookshelf.glb"
        ),
        RetrievalRequest(
            query="floor lamp",
            glb_path=output_dir / "lamp.glb"
        )
    ]
    
    print("🚀 Starting batch retrieval test...")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print(f"📊 Number of models to retrieve: {len(tasks)}")
    print()
    
    # Test batch retrieval with verbose output
    try:
        retrieve_all_models(tasks, verbose=True)
        print("\n✅ Batch retrieval test completed successfully!")
        
        # Check which files were created
        print("\n📋 Results:")
        for task in tasks:
            if task.glb_path.exists():
                size_kb = task.glb_path.stat().st_size / 1024
                print(f"  ✅ {task.glb_path.name}: {size_kb:.2f} KB")
            else:
                print(f"  ❌ {task.glb_path.name}: Not found")
                
    except Exception as e:
        print(f"❌ Batch retrieval test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_group_retrieval()
    sys.exit(0 if success else 1)
