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
    Calculates an axis-aligned bounding box (AABB) for an object given its center,
    unrotated dimensions, and Euler orientation (pitch, yaw, roll).
    Assumes dims are (width_x, depth_y, height_z) along local axes.
    For AABB, we need to consider the maximum extent after rotation.
    Given Z-up, we primarily care about yaw (orientation[2]).
    """
    half_width_x = dims[0] / 2.0
    half_depth_y = dims[1] / 2.0
    half_height_z = dims[2] / 2.0

    # Corners of the unrotated bounding box relative to its center
    corners_local = [
        (-half_width_x, -half_depth_y, -half_height_z),
        ( half_width_x, -half_depth_y, -half_height_z),
        (-half_width_x,  half_depth_y, -half_height_z),
        ( half_width_x,  half_depth_y, -half_height_z),
        (-half_width_x, -half_depth_y,  half_height_z),
        ( half_width_x, -half_depth_y,  half_height_z),
        (-half_width_x,  half_depth_y,  half_height_z),
        ( half_width_x,  half_depth_y,  half_height_z),
    ]

    yaw = orientation[2] # Rotation around Z-axis
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    for cx, cy, cz in corners_local:
        # Rotate only around Z-axis (yaw)
        rotated_x = cx * cos_yaw - cy * sin_yaw
        rotated_y = cx * sin_yaw + cy * cos_yaw
        rotated_z = cz # Z-axis is the rotation axis

        # Translate to global position
        global_x = location[0] + rotated_x
        global_y = location[1] + rotated_y
        global_z = location[2] + rotated_z

        min_x = min(min_x, global_x)
        max_x = max(max_x, global_x)
        min_y = min(min_y, global_y)
        max_y = max(max_y, global_y)
        min_z = min(min_z, global_z)
        max_z = max(max_z, global_z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

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
    # Distance from the asset's max Y to the wall plane.
    # For wall at Y=0, normal (0,-1,0), we want asset.max[1] to be near 0.
    dist_to_wall = abs(asset_layout.max[1] - wall_plane_offset)
    
    # Score based on distance to wall (higher for closer)
    k_dist = 50.0 # Increased sensitivity for strict wall placement
    score_dist = math.exp(-k_dist * dist_to_wall**2)

    # Score based on orientation (yaw around Z-axis)
    # The asset's front (positive X) should point into the room.
    # So, its yaw (orientation[2]) should be close to expected_yaw (e.g., 0 for facing +X).
    yaw_diff = abs(math.fmod(asset_layout.orientation[2] - expected_yaw + math.pi, 2 * math.pi) - math.pi)
    k_yaw = 20.0 # Increased sensitivity for orientation
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
    k_xy = 1.0 # Increased sensitivity for central placement
    score_xy = math.exp(-k_xy * xy_distance_to_center**2)
    
    # Ensure it's on the floor (Z-coordinate of its base is 0)
    z_on_floor_diff = abs(asset_layout.min[2] - 0.0)
    k_z = 50.0 # High sensitivity for being on floor
    score_z = math.exp(-k_z * z_on_floor_diff**2)

    return score_xy * score_z

def score_in_front_of(asset_layout: Layout, target_layout: Layout, min_dist: float, max_dist: float, centered: bool = False) -> float:
    """
    Scores how well an asset is positioned in front of another asset.
    "Front" is along the positive X-axis relative to the target's orientation.
    """
    # Calculate vector from target to asset
    # Rotate the vector by the inverse of target's yaw to align with target's local space
    target_yaw = target_layout.orientation[2]
    cos_yaw = math.cos(-target_yaw) # Inverse rotation
    sin_yaw = math.sin(-target_yaw)

    vec_x_global = asset_layout.location[0] - target_layout.location[0]
    vec_y_global = asset_layout.location[1] - target_layout.location[1]
    vec_z_global = asset_layout.location[2] - target_layout.location[2]

    # Transform vector to target's local coordinate system
    vec_x_local = vec_x_global * cos_yaw - vec_y_global * sin_yaw
    vec_y_local = vec_x_global * sin_yaw + vec_y_global * cos_yaw
    vec_z_local = vec_z_global # Z is up, not affected by yaw

    # Distance in X direction (front) in target's local space
    mid_dist = (min_dist + max_dist) / 2.0
    k_x_dist = 50.0 # Increased sensitivity for X positioning
    score_x_dist = math.exp(-k_x_dist * (vec_x_local - mid_dist)**2)

    # Alignment in Y and Z (horizontal and vertical centering)
    k_align_y = 50.0 if centered else 10.0 # Stricter alignment if centered
    k_align_z = 50.0 if centered else 10.0

    score_align_y = math.exp(-k_align_y * vec_y_local**2)
    score_align_z = math.exp(-k_align_z * vec_z_local**2)

    # Orientation: If target faces +X (yaw=0), asset should face -X (yaw=pi)
    # So, asset's yaw should be target's yaw + pi (180 degrees).
    asset_yaw = asset_layout.orientation[2]
    
    # Normalize yaw difference to be between 0 and pi
    yaw_diff = abs(math.fmod(asset_yaw - (target_yaw + math.pi) + math.pi, 2 * math.pi) - math.pi)
    k_orient = 20.0 # Increased sensitivity for orientation
    score_orient = math.exp(-k_orient * yaw_diff**2)

    return score_x_dist * score_align_y * score_align_z * score_orient

def score_under(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset is positioned directly under another.
    Asset's max Z should be below target's min Z, and their XY projections should overlap.
    """
    if not check_xy_overlap(asset_layout, target_layout):
        return 0.0
    
    # Vertical positioning: asset's max Z should be very close to target's min Z.
    # A small positive z_diff means asset is slightly below target (good).
    # A negative z_diff means asset is overlapping or above target (bad).
    z_diff = target_layout.min[2] - asset_layout.max[2]
    
    # Score is higher if z_diff is positive and close to 0.
    # Penalize heavily if z_diff is negative (overlap) or too large (gap).
    k_z = 100.0 # Very strict sensitivity
    score_z = math.exp(-k_z * z_diff**2) # Penalizes deviation from z_diff = 0
    
    # Add a penalty if asset is actually above target (z_diff < -0.01, small tolerance)
    if z_diff < -0.01: 
        score_z *= 0.01 # Heavy penalty

    return score_z

def score_near(asset_layout: Layout, target_layout: Layout, max_distance: float, extends_in_front: bool = False) -> float:
    """
    Scores how well assets are close to each other.
    If extends_in_front is true, it also checks for being in front.
    """
    distance = math.dist(asset_layout.location, target_layout.location)
    
    # Score is higher if distance is within max_distance.
    # Penalize if distance exceeds max_distance.
    k_dist = 20.0 # Increased sensitivity
    score_dist = math.exp(-k_dist * max(0, distance - max_distance)**2)

    if extends_in_front:
        # For Area Rug extending in front of Sofa:
        # Rug's back (min X) should be near sofa's front (max X) or slightly under.
        # Rug's front (max X) should be significantly in front of sofa's front.
        
        # Score for rug's back being near sofa's front
        # We want rug.min[0] to be slightly less than sofa.max[0] (e.g., -0.1 to -0.2 for partial overlap)
        dist_rug_back_to_sofa_front = asset_layout.min[0] - target_layout.max[0]
        k_rug_back = 50.0
        # Ideal: dist_rug_back_to_sofa_front is slightly negative (rug slightly under sofa). Aim for -0.15 (15cm under).
        score_rug_back = math.exp(-k_rug_back * (dist_rug_back_to_sofa_front - (-0.15))**2)

        # Score for rug's front extending significantly
        dist_rug_front_to_sofa_front = asset_layout.max[0] - target_layout.max[0]
        k_rug_front = 20.0
        # Penalize if rug's front does not extend beyond sofa's front (e.g., at least 0.5m)
        score_rug_front = math.exp(-k_rug_front * max(0, 0.5 - dist_rug_front_to_sofa_front)**2)

        # Ensure XY overlap (especially in Y for the rug to be under/in front of sofa)
        xy_overlap = check_xy_overlap(asset_layout, target_layout)
        
        return score_dist * score_rug_back * score_rug_front * (1.0 if xy_overlap else 0.0)
    
    return score_dist

def score_on(asset_layout: Layout, target_layout: Layout, draped: bool = False) -> float:
    """
    Scores how well an asset is positioned on top of another asset.
    Asset's min Z should be close to target's max Z, and their XY projections overlap.
    If 'draped' is true, allows for some vertical offset and less strict XY overlap.
    """
    # Check XY overlap first
    xy_overlap_score = 1.0
    if not check_xy_overlap(asset_layout, target_layout):
        xy_overlap_score = 0.0 # No overlap, very bad score
    
    # Vertical positioning: asset's min Z should be close to target's max Z.
    z_diff = asset_layout.min[2] - target_layout.max[2]
    
    if draped:
        # For draped items (e.g., throw blanket), allow min Z to be slightly below target's max Z
        # and max Z to be above target's max Z.
        # Ideal z_diff: slightly negative (e.g., -0.03, meaning it drapes down a bit)
        k_z = 100.0 # Increased sensitivity for draped
        score_z = math.exp(-k_z * (z_diff - (-0.03))**2)
        
        # Ensure it extends upwards from the target's top (has some height)
        z_extent_above_target = asset_layout.max[2] - target_layout.max[2]
        k_extent = 50.0 # Increased sensitivity
        score_extent = math.exp(-k_extent * max(0, -z_extent_above_target)**2) # Penalize if it doesn't extend upwards
        
        # Penalize if it's floating too high above the target
        if z_diff > 0.1: # If more than 10cm above target's max Z
            score_z *= 0.001 # Heavy penalty
        
        return score_z * score_extent * xy_overlap_score
    else:
        # For regular 'on' (e.g., cushions), min Z should be very close to target's max Z.
        # Ideal z_diff: 0.0 (asset base is exactly on target top)
        k_z = 200.0 # Much stricter for 'on'
        score_z = math.exp(-k_z * z_diff**2) # Penalize deviation from z_diff = 0
        
        # Add a penalty if asset is actually below target (z_diff < -0.01, small tolerance)
        if z_diff < -0.01: 
            score_z *= 0.001 # Heavy penalty
        
        # Add a penalty if asset is floating too high above target
        if z_diff > 0.05: # If more than 5cm above target's max Z
            score_z *= 0.001 # Heavy penalty
            
        return score_z * xy_overlap_score

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
    # Use a smaller max_distance for complementarity, implying closer proximity.
    near_score = score_near(asset_layout, target_layout, max_distance=1.0) # Arbitrary max_distance for complementarity
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
        "Throw Blanket": (1.5, 0.8, 0.2), # Adjusted dimensions for draped blanket
    }
    
    # Define search space for each asset
    for asset_name in asset_names:
        dims = asset_dims.get(asset_name, (1.0, 1.0, 1.0))
        
        # Narrowed search space for better convergence in a "cozy living room"
        loc_x = trial.suggest_float(f"{asset_name}_loc_x", -1.5, 1.5)
        loc_y = trial.suggest_float(f"{asset_name}_loc_y", -1.5, 1.5)
        
        # Z-location for floor-placed items is half their height (so min Z is 0)
        # For others, Z can vary, but the 'ON' relation will pull them.
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            loc_z = dims[2] / 2.0 # Base at Z=0
        else:
            # For items that go ON other items, their Z-location should be suggested
            # in a range that allows them to be on top of typical furniture.
            # Sofa top is around 0.8m. Cushions/blankets are small.
            # Their center Z could be 0.8 + (dims[2]/2).
            # Let's give a range around that, e.g., 0.7 to 1.1.
            loc_z = trial.suggest_float(f"{asset_name}_loc_z", 0.7, 1.1) # Adjusted Z range
            

        # Suggest orientation (yaw around Z-axis)
        orientation_z = trial.suggest_float(f"{asset_name}_orient_z", -math.pi, math.pi) # Yaw
        orientation = (0.0, 0.0, orientation_z) # Pitch and Roll are 0
        
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
        subject_layout = asset_layouts.get(assets_involved[0])
        
        # Handle special "Wall" target for AGAINST relation
        if relation_type == "AGAINST" and len(assets_involved) > 1 and assets_involved[1] == "Wall":
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
    num_trials = 2000 # Increased trials further
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
        "Throw Blanket": (1.5, 0.8, 0.2), # Adjusted dimensions for draped blanket
    }
    
    for asset_name in asset_names:
        dims = asset_dims.get(asset_name, (1.0, 1.0, 1.0))
        
        loc_x = best_params.get(f"{asset_name}_loc_x", 0.0)
        loc_y = best_params.get(f"{asset_name}_loc_y", 0.0)
        
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            loc_z = dims[2] / 2.0 # Base at Z=0
        else:
            loc_z = best_params.get(f"{asset_name}_loc_z", 0.0) # Use the optimized Z
            
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