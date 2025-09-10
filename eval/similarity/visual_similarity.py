'''
Similar to bbox evaluations, this file uses CLIP score to compute semantic similarity between scene description and rendered images of final produced scenes, comparing to original scene renderings's similarity. 
- NOTE: some scene descriptions may exceed the word length limit of CLIP model, we may consider using BLIP2 instead for longer text inputs. - only a small fraction, clip for now.
    we can try using the shapenet-db's open clip model (used in ulip2) for similarity queries.

3D scene -> 2/4/6th images -> gemma3 captioning = scene descriptions
scene description -> our pipeline -> 3D scene -> images rendering or point clouds conversion

visual similarity score: clip/blip -> img(scene_generated) / img(scene_gt)
3D similarity score: ulip -> pointcloud(scene_generated) / pointcloud(scene_gt)

'''

import os
import sys
sys.path.append(os.getcwd())
import json
from pathlib import Path

from eval.similarity.clip.query import query_similarity_service

'''
Given a job folder (e.g. eval/evals/is_bedroom_10), a dataset path (e.g. dataset/is_bedroom_10/), a path to a exp on dataset folder (e.g. exp/is_bedroom_10) containing each subdir named after scene_id: 
For each scene_id subdir, there should be both a csp/ and direct/ folder for both tested methods, for both: 
1. checks the existence of final/ folder and <scene_id>/<csp/direct>/final/renders/*.png images
    if not exists, early terminate
2. use the scene id to locate the ground truth scene folder to locate subset_path/dataset/<scene_id>.json (as well as subset_path/metadata.json):
    a) read scene description from the scene json file in the 'scene_description' field
    b) also read 'image_paths' field for a list of gt scene rendered images, keep only their filenames as the recorded path has been shifted
    c) read metadata.json's 'source_dataset_dir' field to locate the source dataset path, concat that with <scene_id>/blender_rendered_scene_256/<image_filename> to locate the actual gt rendered images
The actual method to compute similarity is still being developed, so for now, gather the above information and save a job_folder/_clip_prep.json file that contains:
{
    "scene_id": {
        "scene_description": "...",
        "gt_image_paths": ["...", "...", ...],
        "csp_generated_image_paths": ["...", "...", ...],
        "direct_generated_image_paths": ["...", "...", ...]
    },
    ...
}
Those early terminated scenes will have corresponding empty lists for the generated image paths.

Depends on running eval/prep.py
'''
def visual_similarity_prep(job_folder: Path, 
                           dataset_path: Path, 
                           exp_on_dataset_path: Path) -> None:
    if not job_folder.exists():
        raise ValueError(f"Job folder {job_folder} does not exist")
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    if not exp_on_dataset_path.exists():
        raise ValueError(f"Exp on dataset path {exp_on_dataset_path} does not exist")
    
    metadata_path = dataset_path / 'metadata.json'
    if not metadata_path.exists():
        raise ValueError(f"Metadata file {metadata_path} does not exist")
    with metadata_path.open('r') as f:
        metadata = json.load(f)
    source_dataset_dir = metadata.get('source_dataset_dir', None)
    if source_dataset_dir is None:
        raise ValueError(f"No source_dataset_dir field found in {metadata_path}")
    source_dataset_dir = Path(source_dataset_dir)
    if not source_dataset_dir.exists():
        raise ValueError(f"Source dataset dir {source_dataset_dir} does not exist")
    
    scene_json_files = list((dataset_path / 'dataset').glob('*.json'))
    if not scene_json_files:
        raise ValueError(f"No JSON files found in {dataset_path / 'dataset'}")
    
    similarity_prep = {}
    for scene_json_file in scene_json_files:
        scene_id = scene_json_file.stem
        with scene_json_file.open('r') as f:
            scene_data = json.load(f)
        
        scene_description = scene_data.get('scene_description', '').strip()
        image_paths = scene_data.get('image_paths', [])
        image_filenames = [Path(p).name for p in image_paths]
        
        gt_image_paths = []
        for img_filename in image_filenames:
            gt_img_path = source_dataset_dir / scene_id / 'blender_rendered_scene_256' / img_filename
            if gt_img_path.exists():
                gt_image_paths.append(str(gt_img_path))
            else:
                print(f"Warning: GT image path {gt_img_path} does not exist, skipping.")
        
        csp_generated_image_paths = []
        direct_generated_image_paths = []
        
        for method in ['csp', 'direct']:
            final_renders_dir = exp_on_dataset_path / scene_id / method / 'final' / 'renders'
            if not final_renders_dir.exists():
                print(f"Warning: Final renders directory {final_renders_dir} does not exist for scene {scene_id} method {method}, skipping.")
                continue
            render_images = list(final_renders_dir.glob('*.png'))
            if not render_images:
                print(f"Warning: No render images found in {final_renders_dir} for scene {scene_id} method {method}, skipping.")
                continue
            render_image_paths = [str(p) for p in render_images]
            if method == 'csp':
                csp_generated_image_paths = render_image_paths
            else:
                direct_generated_image_paths = render_image_paths

        similarity_prep[scene_id] = {
            'scene_description': scene_description,
            'gt_image_paths': gt_image_paths,
            'csp_generated_image_paths': csp_generated_image_paths,
            'direct_generated_image_paths': direct_generated_image_paths
        }
        
    output_path = job_folder / '_clip_prep.json'
    with output_path.open('w') as f:
        json.dump(similarity_prep, f, indent=4)
    print(f"Saved similarity prep data to {output_path}")

'''
Reads the job_folder/_clip_prep.json file. 
For each scene_id entry, we need to send queries using the query_similarity_service function to compute similarity scores between scene description and gt images, csp generated images, direct generated images respectively. If any of the image paths list is empty, we skip that query and return None for that score.
For csp or direct method, we need to calculate their score relative to gt score, i.e. ratio = method_score / gt_score
Finally, save a job_folder/similarity_clip.json file that contains:
{
    "scene_id": {
        "gt_score": ...,
        "csp_score": ...,
        "csp_relative_score": ...,
        "direct_score": ...,
        "direct_relative_score": ...
    },
    ...
}
In addition, calculate each method's average relative score across all scenes, save in the same json file under the key "average_<method>_relative_score".
'''
def clip_all(job_folder: Path) -> None:
    """
    Compute CLIP similarity scores for all scenes in the job folder.
    
    Args:
        job_folder: Path to the job folder containing _clip_prep.json
    """
    if not job_folder.exists():
        raise ValueError(f"Job folder {job_folder} does not exist")
    
    prep_file = job_folder / '_clip_prep.json'
    if not prep_file.exists():
        raise ValueError(f"Similarity prep file {prep_file} does not exist")
    
    with prep_file.open('r') as f:
        prep_data = json.load(f)
    
    results = {}
    csp_relative_scores = []
    direct_relative_scores = []
    
    for scene_id, scene_data in prep_data.items():
        # print(f"Processing scene: {scene_id}")
        
        scene_description = scene_data['scene_description']
        gt_image_paths = scene_data['gt_image_paths']
        csp_generated_image_paths = scene_data['csp_generated_image_paths']
        direct_generated_image_paths = scene_data['direct_generated_image_paths']
        
        scene_results = {
            'gt_score': None,
            'csp_score': None,
            'csp_relative_score': None,
            'direct_score': None,
            'direct_relative_score': None
        }
        
        # Compute GT score
        if gt_image_paths:
            try:
                gt_response = query_similarity_service(scene_description, gt_image_paths)
                # Calculate average from individual similarity scores
                if 'results' in gt_response and gt_response['results']:
                    similarities = [result['similarity_score'] for result in gt_response['results']]
                    scene_results['gt_score'] = sum(similarities) / len(similarities)
                    # print(f"  GT score: {scene_results['gt_score']} (avg of {len(similarities)} images)")
                else:
                    print(f"  No similarity results in GT response: {gt_response}")
            except Exception as e:
                print(f"  Error computing GT score: {e}")
        else:
            print(f"  No GT images found, skipping GT score")
        
        # Compute CSP score
        if csp_generated_image_paths:
            try:
                csp_response = query_similarity_service(scene_description, csp_generated_image_paths)
                # Calculate average from individual similarity scores
                if 'results' in csp_response and csp_response['results']:
                    similarities = [result['similarity_score'] for result in csp_response['results']]
                    scene_results['csp_score'] = sum(similarities) / len(similarities)
                    # print(f"  CSP score: {scene_results['csp_score']} (avg of {len(similarities)} images)")
                    
                    # Calculate relative score
                    if scene_results['gt_score'] is not None and scene_results['csp_score'] is not None and scene_results['gt_score'] != 0:
                        scene_results['csp_relative_score'] = scene_results['csp_score'] / scene_results['gt_score']
                        csp_relative_scores.append(scene_results['csp_relative_score'])
                        # print(f"  CSP relative score: {scene_results['csp_relative_score']}")
                else:
                    print(f"  No similarity results in CSP response: {csp_response}")
            except Exception as e:
                print(f"  Error computing CSP score: {e}")
        else:
            print(f"  No CSP images found, skipping CSP score")
        
        # Compute Direct score
        if direct_generated_image_paths:
            try:
                direct_response = query_similarity_service(scene_description, direct_generated_image_paths)
                # Calculate average from individual similarity scores
                if 'results' in direct_response and direct_response['results']:
                    similarities = [result['similarity_score'] for result in direct_response['results']]
                    scene_results['direct_score'] = sum(similarities) / len(similarities)
                    # print(f"  Direct score: {scene_results['direct_score']} (avg of {len(similarities)} images)")
                    
                    # Calculate relative score
                    if scene_results['gt_score'] is not None and scene_results['direct_score'] is not None and scene_results['gt_score'] != 0:
                        scene_results['direct_relative_score'] = scene_results['direct_score'] / scene_results['gt_score']
                        direct_relative_scores.append(scene_results['direct_relative_score'])
                        # print(f"  Direct relative score: {scene_results['direct_relative_score']}")
                else:
                    print(f"  No similarity results in Direct response: {direct_response}")
            except Exception as e:
                print(f"  Error computing Direct score: {e}")
        else:
            print(f"  No Direct images found, skipping Direct score")
        
        results[scene_id] = scene_results
    
    # Calculate average relative scores
    if csp_relative_scores:
        results['average_csp_relative_score'] = sum(csp_relative_scores) / len(csp_relative_scores)
        print(f"Average CSP relative score: {results['average_csp_relative_score']}")
    else:
        results['average_csp_relative_score'] = None
        print("No CSP relative scores to average")
    
    if direct_relative_scores:
        results['average_direct_relative_score'] = sum(direct_relative_scores) / len(direct_relative_scores)
        print(f"Average Direct relative score: {results['average_direct_relative_score']}")
    else:
        results['average_direct_relative_score'] = None
        print("No Direct relative scores to average")
    
    # Save results
    output_path = job_folder / 'similarity_clip.json'
    with output_path.open('w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved CLIP similarity results to {output_path}")

if __name__ == '__main__':
    job_folder = Path("eval/evals/is_bedroom_100r33")
    dataset_path = Path("dataset/is_bedroom_100")
    exp_on_dataset_path = Path("exp/is_bedroom_100")
    
    # Prepare similarity data
    visual_similarity_prep(job_folder, dataset_path, exp_on_dataset_path)
    
    # Compute CLIP similarity scores
    clip_all(job_folder)

