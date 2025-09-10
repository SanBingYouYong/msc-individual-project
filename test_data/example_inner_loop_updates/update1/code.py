import optuna
import json
import os
import math
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass, asdict

# Define the Layout dataclass
@dataclass
class Layout:
    location: Tuple[float, float, float]  # (x, y, z) location of the asset in 3D space, Z is up
    min: Tuple[float, float, float]  # minimum corner of the AABB bounding box (x, y, z)
    max: Tuple[float, float, float]  # maximum corner of the AABB bounding box (x, y, z)
    orientation: Tuple[float, float, float]  # Euler angles (pitch, yaw, roll) in radians, Z-up: (rot_x, rot_y, rot_z)

# --- Helper Functions ---
def calculate_aabb(location: Tuple[float, float, float], dims: Tuple[float, float, float], orientation: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculates an axis-aligned bounding box (AABB) for an object.
    This simplified version assumes the object's local axes are aligned with global axes
    when unrotated, and then rotated. For AABB, we need to find the min/max extents
    after rotation. For simplicity, we'll use the unrotated AABB centered at location.
    A more accurate AABB for a rotated object would involve rotating all 8 corners
    and finding the min/max of the resulting coordinates.
    Given the problem's focus on relations, a simple AABB based on dimensions and center
    is often sufficient for initial checks, especially if rotations are primarily yaw (around Z).

    dims: (width_x, depth_y, height_z)
    location: (center_x, center_y, center_z)
    """
    half_width_x = dims[0] / 2.0
    half_depth_y = dims[1] / 2.0
    half_height_z = dims[2] / 2.0

    min_corner = (
        location[0] - half_width_x,
        location[1] - half_depth_y,
        location[2] - half_height_z
    )
    max_corner = (
        location[0] + half_width_x,
        location[1] + half_depth_y,
        location[2] + half_height_z
    )
    return min_corner, max_corner

def check_xy_overlap(layout1: Layout, layout2: Layout) -> bool:
    """Checks if the XY projections of two AABBs overlap."""
    overlap_x = (layout1.min[0] < layout2.max[0]) and (layout1.max[0] > layout2.min[0])
    overlap_y = (layout1.min[1] < layout2.max[1]) and (layout1.max[1] > layout2.min[1])
    return overlap_x and overlap_y

# --- Scoring Functions for Relations (Z-up) ---

def score_against_wall(asset_layout: Layout, wall_plane_normal: Tuple[float, float, float], wall_plane_offset: float, expected_yaw: float = 0.0) -> float:
    """
    Scores how well an asset is positioned against a wall.
    Assumes the wall is a plane defined by normal and offset.
    For a wall at Y=0, normal (0, -1, 0) means the wall is in the XZ plane, and the normal points into the room.
    The asset's max Y should be close to the wall_plane_offset.
    Also, the asset's front (positive X) should be aligned with the expected yaw.
    """
    # Distance to wall
    # Assuming wall at Y=0, normal (0, -1, 0) means asset's max Y should be close to 0.
    # If wall is at Y=0, and normal is (0,1,0) (pointing out), then asset's min Y should be close to 0.
    # Let's assume the main wall is at Y=0, and the fireplace should be against it, facing +X.
    # So, fireplace's max Y should be close to 0.
    
    # For a wall at Y=0, the asset's Y-extent should be close to 0.
    # If the wall is at Y=0, and the asset is placed against it, its Y-coordinates should be small.
    # Let's assume the wall is at Y=0, and the asset's max Y should be close to 0.
    
    # Distance from the asset's max Y to the wall plane.
    dist_to_wall = abs(asset_layout.max[1] - wall_plane_offset)
    
    # Score based on distance to wall (higher for closer)
    k_dist = 10.0 # Sensitivity
    score_dist = math.exp(-k_dist * dist_to_wall**2)

    # Score based on orientation (yaw around Z-axis)
    # The asset's front (positive X) should point into the room.
    # So, its yaw (orientation[2]) should be close to expected_yaw (e.g., 0 for facing +X).
    yaw_diff = abs(math.fmod(asset_layout.orientation[2] - expected_yaw + math.pi, 2 * math.pi) - math.pi)
    k_yaw = 5.0 # Sensitivity
    score_yaw = math.exp(-k_yaw * yaw_diff**2)
    
    return score_dist * score_yaw

def score_focal_point(asset_layout: Layout, scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """
    Scores how well an asset is positioned as a focal point.
    A focal point is often centrally located and prominent.
    We'll score based on proximity to the scene center (XY plane) and being at floor level.
    """
    # Distance in XY plane to center
    xy_distance_to_center = math.dist(asset_layout.location[:2], scene_center[:2])
    
    # Score decreases as distance increases.
    k_xy = 0.5 # Sensitivity
    score_xy = math.exp(-k_xy * xy_distance_to_center**2)
    
    # Ensure it's on the floor (Z-coordinate of its base is 0)
    z_on_floor_diff = abs(asset_layout.min[2] - 0.0)
    k_z = 10.0 # High sensitivity for being on floor
    score_z = math.exp(-k_z * z_on_floor_diff**2)

    return score_xy * score_z

def score_in_front_of(asset_layout: Layout, target_layout: Layout, min_dist: float, max_dist: float, centered: bool = False) -> float:
    """
    Scores how well an asset is positioned in front of another asset.
    "Front" is along the positive X-axis relative to the target's orientation.
    """
    # Calculate vector from target to asset
    vec_x = asset_layout.location[0] - target_layout.location[0]
    vec_y = asset_layout.location[1] - target_layout.location[1]
    vec_z = asset_layout.location[2] - target_layout.location[2]

    # Distance in X direction (front)
    score_x_dist = 0.0
    if min_dist <= vec_x <= max_dist:
        # Score peaks within the desired range
        score_x_dist = 1.0 # Perfect score if within range
    elif vec_x > max_dist:
        score_x_dist = math.exp(-5.0 * (vec_x - max_dist)**2) # Penalize if too far
    else: # vec_x < min_dist
        score_x_dist = math.exp(-5.0 * (min_dist - vec_x)**2) # Penalize if too close or behind

    # Alignment in Y and Z (horizontal and vertical centering)
    k_align_y = 5.0 if centered else 1.0 # Stricter alignment if centered
    k_align_z = 5.0 if centered else 1.0

    score_align_y = math.exp(-k_align_y * vec_y**2)
    score_align_z = math.exp(-k_align_z * vec_z**2)

    # Orientation: If target faces +X (yaw=0), asset should face -X (yaw=pi)
    # Or, if target faces +X, and asset is in front, asset's front should be towards target.
    # So, asset's yaw should be target's yaw + pi (180 degrees).
    target_yaw = target_layout.orientation[2]
    asset_yaw = asset_layout.orientation[2]
    
    # Normalize yaw difference to be between 0 and pi
    yaw_diff = abs(math.fmod(asset_yaw - (target_yaw + math.pi) + math.pi, 2 * math.pi) - math.pi)
    k_orient = 5.0 # Sensitivity for orientation
    score_orient = math.exp(-k_orient * yaw_diff**2)

    return score_x_dist * score_align_y * score_align_z * score_orient

def score_under(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset is positioned directly under another.
    Asset's max Z should be below target's min Z, and their XY projections should overlap.
    """
    if not check_xy_overlap(asset_layout, target_layout):
        return 0.0
    
    # Vertical positioning: asset's max Z should be below target's min Z.
    z_diff = target_layout.min[2] - asset_layout.max[2]
    
    # Score is higher if z_diff is positive and close to 0 (meaning just under).
    k_z = 10.0 # Sensitivity
    score_z = math.exp(-k_z * z_diff**2) if z_diff >= 0 else 0.0 # Penalize if asset is above target

    return score_z

def score_near(asset_layout: Layout, target_layout: Layout, max_distance: float, extends_in_front: bool = False) -> float:
    """
    Scores how well assets are close to each other.
    If extends_in_front is true, it also checks for being in front.
    """
    distance = math.dist(asset_layout.location, target_layout.location)
    
    # Score is higher if distance is within max_distance.
    k_dist = 5.0 # Sensitivity
    score_dist = math.exp(-k_dist * max(0, distance - max_distance)**2)

    if extends_in_front:
        # For Area Rug extending in front of Sofa:
        # Rug should be at a larger X than Sofa, and overlap in Y.
        x_diff = asset_layout.location[0] - target_layout.location[0]
        y_overlap = check_xy_overlap(asset_layout, target_layout) # Check Y overlap specifically
        
        score_x = 0.0
        if x_diff > 0: # Rug is in front
            score_x = 1.0 # Good if in front
        else:
            score_x = math.exp(-5.0 * x_diff**2) # Penalize if behind or too aligned

        return score_dist * score_x * (1.0 if y_overlap else 0.0)
    
    return score_dist

def score_on(asset_layout: Layout, target_layout: Layout, draped: bool = False) -> float:
    """
    Scores how well an asset is positioned on top of another asset.
    Asset's min Z should be close to target's max Z, and their XY projections overlap.
    If 'draped' is true, allows for some vertical offset and less strict XY overlap.
    """
    if not check_xy_overlap(asset_layout, target_layout):
        return 0.0
    
    # Vertical positioning: asset's min Z should be close to target's max Z.
    z_diff = asset_layout.min[2] - target_layout.max[2]
    
    if draped:
        # For draped items (e.g., throw blanket), allow min Z to be slightly below target's max Z
        # and max Z to be above target's max Z.
        k_z = 5.0 # Less strict for draped
        score_z = math.exp(-k_z * z_diff**2)
        
        # Ensure it extends upwards from the target's top
        z_extent_above_target = asset_layout.max[2] - target_layout.max[2]
        score_extent = math.exp(-5.0 * max(0, -z_extent_above_target)**2) # Penalize if it doesn't extend
        
        return score_z * score_extent
    else:
        # For regular 'on' (e.g., cushions), min Z should be very close to target's max Z.
        k_z = 10.0 # Stricter for 'on'
        score_z = math.exp(-k_z * z_diff**2)
        return score_z

def score_part_of(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset is a component or part of another.
    This implies being 'on' or 'near' and typically smaller in size.
    We'll give a constant high score if the 'on' relation is met.
    """
    # If the asset is on the target, it's likely part of it.
    # This is a semantic relation, so we'll give a bonus if spatial conditions are met.
    on_score = score_on(asset_layout, target_layout)
    return on_score * 1.0 # A bonus if it's on

def score_complements(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well one asset enhances the aesthetic or function of another.
    This is a semantic relation. We'll give a constant high score if they are near or on each other.
    """
    # If assets are near or on each other, they likely complement each other.
    near_score = score_near(asset_layout, target_layout, max_distance=2.0) # Arbitrary max_distance for complementarity
    on_score = score_on(asset_layout, target_layout)
    return max(near_score, on_score) * 1.0 # A bonus if they are near or on

# --- Relation Mapping ---
# Maps relation names to their scoring functions and expected arguments.
RELATION_SCORERS = {
    "AGAINST": score_against_wall,
    "IN_FRONT_OF": score_in_front_of,
    "UNDER": score_under,
    "NEAR": score_near,
    "ON": score_on,
    "PART_OF": score_part_of,
    "COMPLEMENTS": score_complements,
    # "FocalPoint", "EmitsGlow", "HoldsDecorativeItems" are properties of single assets,
    # handled directly in objective or as part of asset definition.
}

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, asset_names: List[str], relations_graph: List[Tuple[str, List[str], Dict[str, Any]]]) -> float:
    """
    Optuna objective function to optimize the 3D scene layout.
    """
    asset_layouts: Dict[str, Layout] = {}
    
    # Define typical dimensions for assets (width_x, depth_y, height_z)
    asset_dims = {
        "Fireplace": (1.5, 0.5, 1.2),  # width, depth, height
        "Sofa": (2.0, 0.9, 0.8),
        "Coffee Table": (1.0, 0.5, 0.4),
        "Area Rug": (2.5, 1.8, 0.01), # very thin
        "Sofa Cushions": (0.5, 0.2, 0.2),
        "Throw Blanket": (1.5, 0.1, 1.0), # Blanket dimensions are tricky, this is for its bounding box
    }
    
    # Define search space for each asset
    for asset_name in asset_names:
        dims = asset_dims.get(asset_name, (1.0, 1.0, 1.0))
        half_height_z = dims[2] / 2.0

        # Suggest location parameters (x, y, z)
        # Z is up. For floor-placed items, z_location is half_height_z.
        # For others, z_location can vary.
        
        loc_x = trial.suggest_float(f"{asset_name}_loc_x", -2.5, 2.5)
        loc_y = trial.suggest_float(f"{asset_name}_loc_y", -2.5, 2.5)
        
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            loc_z = half_height_z # Base at Z=0
        else:
            # For items like cushions/blanket, their Z can be higher, relative to other objects.
            # We'll let the 'ON' relation guide their Z.
            loc_z = trial.suggest_float(f"{asset_name}_loc_z", 0.0, 1.5) # Max height of a room object

        # Suggest orientation (yaw around Z-axis)
        # Pitch (rot_x) and Roll (rot_y) are typically 0 for furniture.
        orientation_z = trial.suggest_float(f"{asset_name}_orient_z", -math.pi, math.pi) # Yaw
        orientation = (0.0, 0.0, orientation_z)
        
        min_corner, max_corner = calculate_aabb((loc_x, loc_y, loc_z), dims, orientation)
        
        asset_layouts[asset_name] = Layout(
            location=(loc_x, loc_y, loc_z),
            min=min_corner,
            max=max_corner,
            orientation=orientation
        )

    total_score = 0.0
    
    # Score for individual asset properties (not relation-based)
    # Fireplace as focal point
    total_score += score_focal_point(asset_layouts["Fireplace"])
    # Fireplace emits glow and holds decorative items (semantic, assume always true if fireplace exists)
    total_score += 1.0 # score_emits_glow
    total_score += 1.0 # score_holds_decorative_items

    # Calculate scores for each relation in the graph
    for relation_type, assets_involved, params in relations_graph:
        scorer_func = RELATION_SCORERS.get(relation_type)
        if not scorer_func:
            print(f"Warning: Scorer for relation type '{relation_type}' not found.")
            continue

        # Extract layouts for assets involved in this relation
        # The first asset is typically the subject, the second is the object/target.
        subject_layout = asset_layouts.get(assets_involved[0])
        
        # Handle special "Wall" target for AGAINST relation
        if relation_type == "AGAINST" and assets_involved[1] == "Wall":
            # Assume wall at Y=0, normal (0, -1, 0) (pointing into room), fireplace faces +X (yaw=0)
            score = scorer_func(subject_layout, wall_plane_normal=(0.0, -1.0, 0.0), wall_plane_offset=0.0, expected_yaw=0.0)
            total_score += score
            continue # Move to next relation

        # For other relations, extract target layout(s)
        target_layout = asset_layouts.get(assets_involved[1]) if len(assets_involved) > 1 else None

        if not subject_layout or (len(assets_involved) > 1 and not target_layout):
            # Skip if required assets are not found (shouldn't happen with correct asset_names)
            continue

        # Call the appropriate scoring function with layouts and parameters
        try:
            if relation_type == "IN_FRONT_OF":
                # Special handling for "Coffee Table in front of Sofa" to be centered
                is_centered = (assets_involved[0] == "Coffee Table" and assets_involved[1] == "Sofa")
                score = scorer_func(subject_layout, target_layout, **params, centered=is_centered)
            elif relation_type == "NEAR":
                # Special handling for "Area Rug near Sofa" to extend in front
                extends_in_front = (assets_involved[0] == "Area Rug" and assets_involved[1] == "Sofa")
                score = scorer_func(subject_layout, target_layout, **params, extends_in_front=extends_in_front)
            elif relation_type == "ON":
                # Special handling for "Throw Blanket on Sofa" to be draped
                is_draped = (assets_involved[0] == "Throw Blanket" and assets_involved[1] == "Sofa")
                score = scorer_func(subject_layout, target_layout, draped=is_draped)
            else:
                score = scorer_func(subject_layout, target_layout, **params)
            
            total_score += score
        except TypeError as e:
            print(f"Error calling scorer for {relation_type} with assets {assets_involved}: {e}")
            # Assign a low score if arguments mismatch, indicating a problem in relation definition or scorer.
            total_score += 0.0
        except Exception as e:
            print(f"General error scoring relation {relation_type} for {assets_involved}: {e}")
            total_score += 0.0

    return total_score

# --- Main Execution ---

if __name__ == "__main__":
    
    asset_names = [
        "Fireplace",
        "Sofa",
        "Coffee Table",
        "Area Rug",
        "Sofa Cushions",
        "Throw Blanket"
    ]
    
    # The graph structure: (RelationType, [SubjectAsset, ObjectAsset, ...], {optional_params})
    # Z-axis is up, X is front, Y is right.
    relations_graph = [
        ("AGAINST", ["Fireplace", "Wall"], {"wall_plane_normal": (0.0, -1.0, 0.0), "wall_plane_offset": 0.0, "expected_yaw": 0.0}), # Fireplace against wall at Y=0, facing +X
        ("IN_FRONT_OF", ["Sofa", "Fireplace"], {"min_dist": 1.5, "max_dist": 2.5, "centered": False}), # Sofa in front of Fireplace
        ("IN_FRONT_OF", ["Coffee Table", "Sofa"], {"min_dist": 0.5, "max_dist": 1.0, "centered": True}), # Coffee Table in front of Sofa, centered
        ("UNDER", ["Area Rug", "Coffee Table"], {}), # Area Rug under Coffee Table
        ("NEAR", ["Area Rug", "Sofa"], {"max_distance": 1.0, "extends_in_front": True}), # Area Rug near Sofa, extending in front
        ("ON", ["Sofa Cushions", "Sofa"], {}), # Sofa Cushions on Sofa
        ("ON", ["Throw Blanket", "Sofa"], {"draped": True}), # Throw Blanket draped on Sofa
        ("PART_OF", ["Sofa Cushions", "Sofa"], {}), # Sofa Cushions are part of Sofa
        ("PART_OF", ["Throw Blanket", "Sofa"], {}), # Throw Blanket is part of Sofa
        ("COMPLEMENTS", ["Area Rug", "Sofa"], {}),
        ("COMPLEMENTS", ["Sofa Cushions", "Sofa"], {}),
        ("COMPLEMENTS", ["Throw Blanket", "Sofa"], {}),
    ]
    
    # Create an Optuna study.
    study = optuna.create_study(direction="maximize")
    
    # Run the optimization.
    # Increased number of trials for better convergence.
    num_trials = 500
    study.optimize(lambda trial: objective(trial, asset_names, relations_graph), n_trials=num_trials)
    
    print("Optimization finished.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    
    # Get the best layout found.
    best_params = study.best_params
    
    # Reconstruct the best layout from the best parameters.
    best_asset_layouts: Dict[str, Layout] = {}
    asset_dims = {
        "Fireplace": (1.5, 0.5, 1.2),
        "Sofa": (2.0, 0.9, 0.8),
        "Coffee Table": (1.0, 0.5, 0.4),
        "Area Rug": (2.5, 1.8, 0.01),
        "Sofa Cushions": (0.5, 0.2, 0.2),
        "Throw Blanket": (1.5, 0.1, 1.0),
    }
    
    for asset_name in asset_names:
        dims = asset_dims.get(asset_name, (1.0, 1.0, 1.0))
        half_height_z = dims[2] / 2.0
        
        loc_x = best_params.get(f"{asset_name}_loc_x", 0.0)
        loc_y = best_params.get(f"{asset_name}_loc_y", 0.0)
        
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            loc_z = half_height_z # Base at Z=0
        else:
            loc_z = best_params.get(f"{asset_name}_loc_z", 0.0)
            
        orientation_z = best_params.get(f"{asset_name}_orient_z", 0.0)
        orientation = (0.0, 0.0, orientation_z) # Pitch and Roll are 0
        
        min_corner, max_corner = calculate_aabb((loc_x, loc_y, loc_z), dims, orientation)
        
        best_asset_layouts[asset_name] = Layout(
            location=(loc_x, loc_y, loc_z),
            min=min_corner,
            max=max_corner,
            orientation=orientation
        )

    # Save the optimal layout to a JSON file.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    layout_file_path = os.path.join(script_dir, "layout.json")
    
    layout_dict = {name: asdict(layout) for name, layout in best_asset_layouts.items()}
    
    with open(layout_file_path, "w") as f:
        json.dump(layout_dict, f, indent=4)
        
    print(f"Optimal layout saved to: {layout_file_path}")
    
    # Print the best layout found
    print("\n--- Best Layout ---")
    for asset_name, layout in best_asset_layouts.items():
        print(f"{asset_name}:")
        print(f"  Location: {layout.location}")
        print(f"  Min BBox: {layout.min}")
        print(f"  Max BBox: {layout.max}")
        print(f"  Orientation: {layout.orientation}")