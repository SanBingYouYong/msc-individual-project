from pathlib import Path
import sys
import os
import sys
sys.path.append(os.getcwd())
import json

def count_descriptions_exceeding_token_limit(dataset_path, token_limit):
    dataset_path = Path(dataset_path)
    count_exceeds = 0
    total = 0

    for subdir in dataset_path.iterdir():
        desc_file = subdir / "scene_description.txt"
        if desc_file.is_file():
            with open(desc_file, "r", encoding="utf-8") as f:
                desc = f.read().strip()
                if len(desc.split()) > token_limit:
                    count_exceeds += 1
                total += 1

    return total, count_exceeds

def count_scene_objects(dataset_path):
    dataset_path = Path(dataset_path)
    scene_object_counts = {}

    for subdir in dataset_path.iterdir():
        # read scene_with_bbox.json and count length of field 'object_descriptions'
        json_file = subdir / "scene_with_bbox.json"
        if json_file.is_file():
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                object_count = len(data.get("object_descriptions", []))
                scene_object_counts[subdir.name] = object_count
    return scene_object_counts

def analyze_object_counts(counts_dict):
    counts = list(counts_dict.values())
    if not counts:
        return None, None, None
    average = sum(counts) / len(counts)
    minimum = min(counts)
    maximum = max(counts)
    # also check the distribution of counts (e.g. number of 3-object scenes, 5-object scenes, etc.)
    unique_counts = set(counts)
    distribution = {count: counts.count(count) for count in unique_counts}
    print("Object count distribution:", distribution)
    return average, minimum, maximum

def count_scenes_with_object_counts(dataset_path, counts: list):
    dataset_path = Path(dataset_path)
    count_dict = {count: 0 for count in counts}

    for subdir in dataset_path.iterdir():
        json_file = subdir / "scene_with_bbox.json"
        if json_file.is_file():
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                object_count = len(data.get("object_descriptions", []))
                if object_count in count_dict:
                    count_dict[object_count] += 1
    return count_dict


if __name__ == '__main__':
    # Example usage
    dataset_path = "dataset/InstructScene/threed_front_bedroom/"
    token_limit = 77

    # total, count_exceeds = count_descriptions_exceeding_token_limit(dataset_path, token_limit)
    # print(f"Total descriptions: {total}") # ~4000
    # print(f"Descriptions exceeding {token_limit} tokens: {count_exceeds}") # ~100
    
    scene_object_counts = count_scene_objects(dataset_path)
    average, minimum, maximum = analyze_object_counts(scene_object_counts)
    print(f"Average number of objects per scene: {average}")
    print(f"Minimum number of objects in a scene: {minimum}")
    print(f"Maximum number of objects in a scene: {maximum}")
    
    # counts = count_scenes_with_object_counts(dataset_path, [3, 5, 7])
    # print(counts)