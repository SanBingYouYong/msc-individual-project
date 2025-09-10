"""
the eval/evals/is_bedroom-30_3_5_7 folder is created by prep.py, and then eval/bbox/compare_overlap.py, eval/similarity/visual_similarity.py and eval/similarity/pointcloud_similarity.py are run on it. 
Now, dataset/is_bedroom-30_3_5_7/metadata.json contains information on the number of objects in each scene, and grouped scenes by the object counts under 'counts' field, as a dict of <int object_count, list of scene_ids>.
We need to evaluate these metrics again stratified by object counts, and save the results to eval/evals/is_bedroom-30_3_5_7/stratified_eval.json. 
To do so: 
- read dataset/is_bedroom-30_3_5_7/metadata.json to get the scene groups by object counts
- read existing overall evaluations from similarity_clip.json, similarity_ulip2.json, and overlap.json under eval/evals/is_bedroom-30_3_5_7/, all of them are dict of dicts: {scene_id: {method: score, ...}, ...} (contains extra fields like averages, ignore those)
- match the scene ids in each group with the overall evaluations, and calculate the average metrics for each group, for each group, record the following metrics:
    - average bbox IoU for each method (csp, direct), obtained from overlap.json
    - average visual similarity for each method (csp, direct), obtained from similarity_clip.json
    - average point cloud similarity for each method (csp, direct), obtained from similarity_ulip2.json
You need to check the existence of each method's score for each scene, if not exists, ignore that scene for that method's average calculation.
Save the final results to eval/evals/is_bedroom-30_3_5_7/stratified_eval.json, as a dict of dicts: {object_count: {metric: {method: average_score, ...}, ...}, ...}
"""

import json
import os
from collections import defaultdict

def load_json(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_stratified_metrics(dataset_dir, eval_dir):
    """Calculate stratified evaluation metrics by object count. Accepts dataset and eval directories as inputs."""
    
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    similarity_clip_path = os.path.join(eval_dir, "similarity_clip.json")
    similarity_ulip2_path = os.path.join(eval_dir, "similarity_ulip2.json") 
    overlap_path = os.path.join(eval_dir, "overlap.json")
    output_path = os.path.join(eval_dir, "stratified_eval.json")
    
    # Load data
    metadata = load_json(metadata_path)
    similarity_clip = load_json(similarity_clip_path)
    similarity_ulip2 = load_json(similarity_ulip2_path)
    overlap = load_json(overlap_path)
    
    # Get scene groups by object count
    scene_groups = metadata["counts"]
    
    # Initialize results structure
    results = {}
    
    # Process each object count group
    for obj_count, scene_ids in scene_groups.items():
        obj_count = int(obj_count)  # Ensure it's an integer
        results[obj_count] = {
            "bbox_iou": {"csp": [], "direct": []},
            "visual_similarity": {"csp": [], "direct": []}, 
            "pointcloud_similarity": {"csp": [], "direct": []}
        }
        
        # Collect scores for each scene in this group
        for scene_id in scene_ids:
            
            # Bbox IoU from overlap.json
            if scene_id in overlap:
                scene_overlap = overlap[scene_id]
                if "generated_layout_csp" in scene_overlap:
                    results[obj_count]["bbox_iou"]["csp"].append(scene_overlap["generated_layout_csp"])
                if "generated_layout_direct" in scene_overlap:
                    results[obj_count]["bbox_iou"]["direct"].append(scene_overlap["generated_layout_direct"])
            
            # Visual similarity from similarity_clip.json
            if scene_id in similarity_clip:
                scene_clip = similarity_clip[scene_id]
                if "csp_relative_score" in scene_clip and scene_clip["csp_relative_score"] is not None:
                    results[obj_count]["visual_similarity"]["csp"].append(scene_clip["csp_relative_score"])
                if "direct_relative_score" in scene_clip and scene_clip["direct_relative_score"] is not None:
                    results[obj_count]["visual_similarity"]["direct"].append(scene_clip["direct_relative_score"])
            
            # Point cloud similarity from similarity_ulip2.json
            if scene_id in similarity_ulip2:
                scene_ulip2 = similarity_ulip2[scene_id]
                if "csp" in scene_ulip2:
                    results[obj_count]["pointcloud_similarity"]["csp"].append(scene_ulip2["csp"])
                if "direct" in scene_ulip2:
                    results[obj_count]["pointcloud_similarity"]["direct"].append(scene_ulip2["direct"])
        
        # Calculate averages for each metric and method
        final_results = {}
        for metric in ["bbox_iou", "visual_similarity", "pointcloud_similarity"]:
            final_results[metric] = {}
            for method in ["csp", "direct"]:
                scores = results[obj_count][metric][method]
                if scores:  # Only calculate average if there are scores
                    final_results[metric][method] = sum(scores) / len(scores)
                else:
                    final_results[metric][method] = None
        
        results[obj_count] = final_results
    
    # Save results
    save_json(results, output_path)
    print(f"Stratified evaluation results saved to {output_path}")
    
    # Print summary
    print("\nSummary of stratified evaluation:")
    for obj_count in sorted(results.keys()):
        print(f"\nObject count: {obj_count}")
        for metric in ["bbox_iou", "visual_similarity", "pointcloud_similarity"]:
            print(f"  {metric}:")
            for method in ["csp", "direct"]:
                score = results[obj_count][metric][method]
                if score is not None:
                    print(f"    {method}: {score:.4f}")
                else:
                    print(f"    {method}: No data")

if __name__ == "__main__":
    calculate_stratified_metrics(
        dataset_dir="dataset/is_bedroom-30-4_6",
        eval_dir="eval/evals/is_bedroom-30-4_6"
    )
