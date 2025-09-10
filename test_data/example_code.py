import optuna
import json
import os
from typing import Tuple, Dict, List
from dataclasses import dataclass, asdict

# Define the Layout dataclass
@dataclass
class Layout:
    location: Tuple[float, float, float]  # location of the asset in 3D space
    min: Tuple[float, float, float]  # minimum corner of the AABB bounding box
    max: Tuple[float, float, float]  # maximum corner of the AABB bounding box
    orientation: Tuple[float, float, float]  # Euler angles (pitch, yaw, roll) in radians

# --- Scoring Functions for Relations ---

def score_against_wall(asset_layout: Layout, wall_normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> float:
    """
    Scores how well an asset is positioned against a wall.
    Assumes the wall is along the Y-axis (normal is (0, 1, 0)).
    A higher score means the asset's back (min y) is close to the wall's y=0 plane.
    """
    # We'll approximate "against the wall" by checking if the minimum y-coordinate is close to 0.
    # This is a simplification and might need adjustment based on the coordinate system and wall definition.
    # For a wall along the Y-axis at y=0, the asset's min y should be close to 0.
    # If the wall normal is (0, 1, 0), it means the wall is in the XZ plane.
    # If the asset's min y is close to 0, it's against the wall.
    # Let's assume the wall is at y=0 and the normal points outwards (0, 1, 0).
    # The asset's min y should be close to 0.
    # A more robust approach would involve checking the distance from the asset's bounding box
    # to a plane defined by the wall.
    
    # For simplicity, let's assume the wall is at y=0 and the asset's min y should be close to 0.
    # If the wall normal is (0, 1, 0), it means the wall is in the XZ plane.
    # The asset's min y should be close to 0.
    
    # Let's consider the asset's bounding box. If the wall is at y=0, and the normal is (0,1,0),
    # then the asset's min y should be close to 0.
    
    # A simple heuristic: check if the minimum y of the bounding box is close to 0.
    # This assumes the wall is at y=0.
    # The score is higher if the min_y is closer to 0.
    
    # Let's refine this: assume the wall is a plane. The distance from the asset's center
    # to the wall plane should be considered.
    # For simplicity, let's check the minimum y-coordinate of the bounding box.
    # If the wall is at y=0, and the normal is (0,1,0), then min_y should be close to 0.
    
    # A score that increases as min_y approaches 0.
    # We can use an exponential decay or a Gaussian-like function.
    # Let's use a simple inverse relationship with a small epsilon to avoid division by zero.
    # Score = 1 / (abs(min_y) + epsilon)
    # Or, a function that peaks at 0 and decreases.
    # Score = exp(-k * min_y^2)
    
    # Let's assume the wall is at y=0. The asset's min y should be close to 0.
    # The score should be high when min_y is close to 0.
    
    # A simple approach: if the asset's min_y is very close to 0, give a high score.
    # Otherwise, a low score.
    
    # Let's consider the asset's bounding box. If the wall is at y=0, and the normal is (0,1,0),
    # then the asset's min y should be close to 0.
    
    # A score that increases as min_y approaches 0.
    # Let's use a Gaussian-like function centered at 0.
    # score = exp(-k * (asset_layout.min[1]**2))
    # A higher k means a stricter requirement.
    
    k = 100.0  # Sensitivity parameter
    score = math.exp(-k * (asset_layout.min[1]**2))
    return score

def score_focal_point(asset_layout: Layout, scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
    """
    Scores how well an asset is positioned as a focal point.
    A focal point is often centrally located or prominent.
    We'll score based on proximity to the scene center.
    """
    distance_to_center = math.dist(asset_layout.location, scene_center)
    # Score decreases as distance increases.
    # Using an inverse relationship with a small epsilon.
    score = 1.0 / (distance_to_center + 0.1)
    return score

def score_in_front_of(asset_layout: Layout, target_layout: Layout, distance_threshold: float = 2.0) -> float:
    """
    Scores how well an asset is positioned in front of another asset.
    Assumes the 'front' is along the positive Y-axis relative to the target.
    """
    # Calculate the vector from the target asset's location to the asset's location.
    vector_to_asset = (
        asset_layout.location[0] - target_layout.location[0],
        asset_layout.location[1] - target_layout.location[1],
        asset_layout.location[2] - target_layout.location[2]
    )
    
    # For "in front of", we expect the asset to be in the positive Y direction relative to the target.
    # If the target is facing forward (e.g., along +Y), then "in front of" means a larger Y value.
    # Let's assume the target's orientation defines its "front".
    # For simplicity, let's assume the target's "front" is along its local +Y axis.
    # We can approximate this by checking if the asset's location has a larger Y coordinate
    # than the target's location, and is aligned in X and Z.
    
    # A simpler approach: check if the asset's location is in the positive Y direction
    # relative to the target's location, and within a certain distance.
    
    # Let's assume the target's "front" is along the positive Y axis.
    # The asset should have a larger Y coordinate than the target.
    # And its X and Z coordinates should be similar to the target's.
    
    # Score based on the difference in Y coordinates and proximity in XZ.
    y_diff = asset_layout.location[1] - target_layout.location[1]
    xz_dist = math.dist(asset_layout.location[:2], target_layout.location[:2]) # Distance in XZ plane
    
    # Score is higher if y_diff is positive and xz_dist is small.
    # We can use a Gaussian-like function for both.
    
    k_y = 0.5  # Sensitivity for Y difference
    k_xz = 1.0 # Sensitivity for XZ distance
    
    score_y = math.exp(-k_y * (y_diff - distance_threshold)**2) if y_diff > 0 else 0.0
    score_xz = math.exp(-k_xz * xz_dist**2)
    
    return score_y * score_xz

def score_centered_in_front_of(asset_layout: Layout, target_layout: Layout, distance_threshold: float = 2.0) -> float:
    """
    Scores how well an asset is positioned centrally in front of another asset.
    This is similar to 'InFrontOf' but with a stronger emphasis on X-axis alignment.
    """
    # Calculate the vector from the target asset's location to the asset's location.
    vector_to_asset = (
        asset_layout.location[0] - target_layout.location[0],
        asset_layout.location[1] - target_layout.location[1],
        asset_layout.location[2] - target_layout.location[2]
    )
    
    # For "centered in front of", we expect the asset to be in the positive Y direction
    # relative to the target, and its X coordinate should be very close to the target's X.
    # Let's assume the target's "front" is along the positive Y axis.
    
    y_diff = asset_layout.location[1] - target_layout.location[1]
    x_diff = asset_layout.location[0] - target_layout.location[0]
    z_diff = asset_layout.location[2] - target_layout.location[2] # Also consider Z alignment
    
    # Score is higher if y_diff is positive and x_diff is close to 0, and z_diff is close to 0.
    
    k_y = 0.5  # Sensitivity for Y difference
    k_x = 5.0  # Sensitivity for X difference (higher for centering)
    k_z = 5.0  # Sensitivity for Z difference
    
    score_y = math.exp(-k_y * (y_diff - distance_threshold)**2) if y_diff > 0 else 0.0
    score_x = math.exp(-k_x * x_diff**2)
    score_z = math.exp(-k_z * z_diff**2)
    
    return score_y * score_x * score_z

def score_within_reach_of(asset_layout: Layout, target_layout: Layout, reach_distance: float = 1.5) -> float:
    """
    Scores how well an asset is positioned within easy reach of another asset (e.g., sofa).
    Assumes reach is primarily in the forward direction (positive Y).
    """
    # Calculate the distance between the asset and the target.
    distance = math.dist(asset_layout.location, target_layout.location)
    
    # Score is higher if the distance is within the reach_distance.
    # We can use a Gaussian-like function centered at 0, but capped at reach_distance.
    
    # Let's consider the relative position. For a coffee table to be within reach of a sofa,
    # it should be in front of the sofa and within a certain distance.
    
    # Let's assume the target (sofa) is oriented such that its front is along +Y.
    # The asset (coffee table) should be in front of the sofa.
    
    y_diff = asset_layout.location[1] - target_layout.location[1]
    xz_dist = math.dist(asset_layout.location[:2], target_layout.location[:2])
    
    # Score is higher if y_diff is positive and within reach, and xz_dist is small.
    
    k_y = 0.5  # Sensitivity for Y difference
    k_xz = 1.0 # Sensitivity for XZ distance
    
    # Score for being in front and within reach distance
    score_y = math.exp(-k_y * (y_diff - reach_distance)**2) if y_diff > 0 else 0.0
    score_xz = math.exp(-k_xz * xz_dist**2)
    
    return score_y * score_xz

def score_under(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset is positioned under another asset.
    Assumes the 'under' relationship means the asset's max Y is below the target's min Y,
    and their XZ projections overlap.
    """
    # Check for overlap in XZ plane.
    overlap_x = (asset_layout.min[0] < target_layout.max[0]) and (asset_layout.max[0] > target_layout.min[0])
    overlap_z = (asset_layout.min[2] < target_layout.max[2]) and (asset_layout.max[2] > target_layout.min[2])
    
    if not (overlap_x and overlap_z):
        return 0.0
    
    # Check for vertical positioning: asset's max Y should be below target's min Y.
    # And asset's min Y should be below target's min Y.
    # A higher score if the asset is directly beneath and its top is below the target's bottom.
    
    # Let's assume the target is a coffee table and the asset is an area rug.
    # The rug should be under the coffee table.
    # This means the rug's Y range should be below the coffee table's Y range.
    
    # A simple score: the difference between the target's min Y and the asset's max Y.
    # This difference should be positive.
    
    y_diff = target_layout.min[1] - asset_layout.max[1]
    
    # Score is higher if y_diff is positive.
    # Using a Gaussian-like function centered at 0 for the difference.
    k = 5.0 # Sensitivity parameter
    score = math.exp(-k * y_diff**2) if y_diff > 0 else 0.0
    
    return score

def score_anchors_seating_arrangement(asset_layout: Layout, seating_assets: List[Layout], proximity_factor: float = 1.0) -> float:
    """
    Scores how well an asset (e.g., area rug) anchors a seating arrangement.
    This means the seating assets should be positioned on or around this asset.
    """
    # For an area rug anchoring a seating arrangement, the seating assets (sofa)
    # should be positioned on top of or very close to the rug.
    
    # Let's assume the rug is roughly centered at its location and has a certain extent.
    # We'll check if the seating assets' locations are within the rug's horizontal bounds
    # and slightly above the rug's top surface.
    
    # For simplicity, let's check if the seating assets are within a certain radius
    # of the rug's location, and their Y position is slightly above the rug's top.
    
    # Let's assume the rug is centered at asset_layout.location and has a width/depth
    # related to its bounding box.
    
    # A simpler approach: check if the seating assets are within a certain distance
    # of the rug's location, and their Y coordinate is slightly above the rug's top.
    
    total_score = 0.0
    num_seating_assets = len(seating_assets)
    
    if num_seating_assets == 0:
        return 0.0
    
    # Let's assume the rug is roughly centered at asset_layout.location.
    # The seating assets should be close to this location.
    
    # A score based on the average distance of seating assets to the rug's location.
    # Lower average distance is better.
    
    # Let's consider the rug's bounding box. The seating assets should be within
    # the horizontal projection of the rug's bounding box.
    
    # For simplicity, let's check if the seating assets are within a certain radius
    # of the rug's location.
    
    radius = 2.0 # Approximate radius of influence for anchoring
    
    for seating_asset in seating_assets:
        distance_to_rug_center = math.dist(seating_asset.location, asset_layout.location)
        
        # Score is higher if the seating asset is closer to the rug's center.
        # And if the seating asset's Y is slightly above the rug's top.
        
        y_diff = seating_asset.location[1] - asset_layout.max[1] # Assuming rug's max Y is its top
        
        # Score for proximity to rug center
        score_proximity = math.exp(-proximity_factor * distance_to_rug_center**2)
        
        # Score for being slightly above the rug
        score_y = math.exp(-5.0 * y_diff**2) if y_diff > 0 else 0.0
        
        total_score += score_proximity * score_y
        
    return total_score / num_seating_assets

def score_on(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset is positioned on top of another asset.
    Assumes the asset's min Y is close to the target's max Y, and their XZ projections overlap.
    """
    # Check for overlap in XZ plane.
    overlap_x = (asset_layout.min[0] < target_layout.max[0]) and (asset_layout.max[0] > target_layout.min[0])
    overlap_z = (asset_layout.min[2] < target_layout.max[2]) and (asset_layout.max[2] > target_layout.min[2])
    
    if not (overlap_x and overlap_z):
        return 0.0
    
    # Check for vertical positioning: asset's min Y should be close to target's max Y.
    # A higher score if the asset's bottom is close to the target's top.
    
    y_diff = asset_layout.min[1] - target_layout.max[1]
    
    # Score is higher if y_diff is close to 0.
    k = 10.0 # Sensitivity parameter
    score = math.exp(-k * y_diff**2)
    
    return score

def score_draped_over(asset_layout: Layout, target_layout: Layout) -> float:
    """
    Scores how well an asset (e.g., throw blanket) is draped over another asset (e.g., sofa).
    This implies the asset is positioned on top of the target, but not necessarily perfectly aligned.
    It should be partially on the target and partially hanging off.
    """
    # This is a more complex geometric relation. For simplicity, we'll approximate.
    # The asset should be mostly above the target's bounding box, with some part
    # extending beyond the target's top surface.
    
    # Let's assume the target is a sofa. The blanket should be on the sofa.
    # Check if the blanket's bounding box is mostly above the sofa's bounding box.
    
    # Check for overlap in XZ plane.
    overlap_x = (asset_layout.min[0] < target_layout.max[0]) and (asset_layout.max[0] > target_layout.min[0])
    overlap_z = (asset_layout.min[2] < target_layout.max[2]) and (asset_layout.max[2] > target_layout.min[2])
    
    if not (overlap_x and overlap_z):
        return 0.0
    
    # Check vertical positioning: the blanket's center should be above the sofa's top.
    # And some part of the blanket should be above the sofa's max Y.
    
    # Let's consider the blanket's bounding box.
    # The blanket's min Y should be close to the sofa's max Y.
    # And the blanket's max Y should be above the sofa's max Y.
    
    y_diff_min = asset_layout.min[1] - target_layout.max[1]
    y_diff_max = asset_layout.max[1] - target_layout.max[1]
    
    # Score is higher if the blanket's bottom is close to the sofa's top (y_diff_min close to 0)
    # and the blanket extends upwards (y_diff_max > 0).
    
    k_min = 10.0 # Sensitivity for bottom alignment
    k_max = 2.0  # Sensitivity for extending upwards
    
    score_min = math.exp(-k_min * y_diff_min**2)
    score_max = math.exp(-k_max * max(0, -y_diff_max)**2) # Penalize if it doesn't extend upwards
    
    return score_min * score_max

def score_emits_glow(asset_layout: Layout) -> float:
    """
    Scores how well an asset emits a warm, ambient glow.
    This is more of a property of the asset itself, but we can assign a high score
    if the asset is identified as a fireplace.
    """
    # In a real system, this might involve checking asset type or material properties.
    # For this simulation, we'll assume a fireplace inherently emits glow.
    # If the asset is a fireplace, give a high score.
    # This function would typically be called for the fireplace asset.
    return 1.0 # Assume the asset is a fireplace if this function is called for it.

def score_holds_decorative_items(asset_layout: Layout) -> float:
    """
    Scores how well an asset has a surface suitable for holding decorative items.
    This implies a flat, stable surface.
    """
    # For a fireplace, the mantelpiece is the relevant surface.
    # We can assume a fireplace's mantelpiece is suitable.
    # This function would typically be called for the fireplace asset.
    return 1.0 # Assume the asset is a fireplace if this function is called for it.

# --- Relation Mapping ---
# Maps relation names to their scoring functions and expected arguments.
# The arguments are placeholders for the actual layouts needed.
RELATION_SCORERS = {
    "AgainstWall": {"func": score_against_wall, "args": ["asset_layout"]},
    "FocalPoint": {"func": score_focal_point, "args": ["asset_layout"]},
    "InFrontOf": {"func": score_in_front_of, "args": ["asset_layout", "target_layout"]},
    "CenteredInFrontOf": {"func": score_centered_in_front_of, "args": ["asset_layout", "target_layout"]},
    "WithinReachOf": {"func": score_within_reach_of, "args": ["asset_layout", "target_layout"]},
    "Under": {"func": score_under, "args": ["asset_layout", "target_layout"]},
    "AnchorsSeatingArrangement": {"func": score_anchors_seating_arrangement, "args": ["asset_layout", "seating_assets"]},
    "On": {"func": score_on, "args": ["asset_layout", "target_layout"]},
    "DrapedOver": {"func": score_draped_over, "args": ["asset_layout", "target_layout"]},
    "EmitsGlow": {"func": score_emits_glow, "args": ["asset_layout"]},
    "HoldsDecorativeItems": {"func": score_holds_decorative_items, "args": ["asset_layout"]},
}

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, asset_names: List[str], relations: Dict[str, List[str]]) -> float:
    """
    Optuna objective function to optimize the 3D scene layout.
    """
    # Define the search space for each asset's layout parameters.
    asset_layouts: Dict[str, Layout] = {}
    
    # Define a base layout for reference, e.g., the center of the room.
    # We'll assume the room is roughly centered around (0,0,0) and extends in positive Y.
    room_center = (0.0, 0.0, 0.0)
    
    # Define typical dimensions for assets to guide the search.
    # These are rough estimates and can be adjusted.
    asset_dims = {
        "Fireplace": (1.5, 0.5, 1.2),  # width, depth, height
        "Sofa": (2.0, 0.9, 0.8),
        "Coffee Table": (1.0, 0.5, 0.4),
        "Area Rug": (2.5, 1.8, 0.01), # very thin
        "Sofa Cushions": (0.5, 0.2, 0.2),
        "Throw Blanket": (1.5, 0.1, 1.0),
    }
    
    # Define default orientations (e.g., facing forward along +Y)
    default_orientation = (0.0, 0.0, 0.0) # pitch, yaw, roll

    for asset_name in asset_names:
        dims = asset_dims.get(asset_name, (1.0, 1.0, 1.0)) # Default dimensions
        
        # Suggest location parameters
        # We'll suggest locations relative to the room center or other assets.
        # For simplicity, let's suggest locations within a reasonable range.
        # The actual constraints will come from the relation scores.
        
        # Initial guess for location: slightly offset from center.
        # The search space for location can be large, so we might need to
        # guide it with initial guesses or more sophisticated sampling.
        
        # Let's try suggesting locations within a bounding box, e.g., -5 to 5 in X, 0 to 5 in Y, -5 to 5 in Z.
        # The Y-axis is up, so we'll place things on the floor (Y=0).
        # Let's assume the room is roughly from y=0 to y=5.
        
        loc_x = trial.suggest_float(f"{asset_name}_loc_x", -5.0, 5.0)
        loc_y = trial.suggest_float(f"{asset_name}_loc_y", 0.0, 5.0) # Place on the floor
        loc_z = trial.suggest_float(f"{asset_name}_loc_z", -5.0, 5.0)
        
        # Suggest orientation parameters (Euler angles)
        # For simplicity, let's allow some rotation around the Y-axis (yaw).
        orientation_y = trial.suggest_float(f"{asset_name}_orient_y", -math.pi, math.pi)
        orientation = (0.0, orientation_y, 0.0) # Pitch and roll are zero for simplicity
        
        # Calculate bounding box based on location, dimensions, and orientation.
        # This is a simplified AABB calculation. A more accurate calculation
        # would involve transforming the corners of the axis-aligned bounding box
        # by the rotation matrix.
        
        # For simplicity, let's assume the bounding box is axis-aligned and
        # its center is at `location`. The dimensions are width, depth, height.
        # We need to map these to x, y, z extents.
        # Assuming orientation is around Y (yaw), width is along X, depth along Z.
        
        half_width = dims[0] / 2.0
        half_depth = dims[1] / 2.0
        half_height = dims[2] / 2.0
        
        # Axis-aligned bounding box based on location and dimensions.
        # This is a simplification as orientation is not fully considered here.
        # A more robust approach would involve rotating the corners.
        
        # For now, let's assume the bounding box is aligned with the axes and
        # centered at `location`. The dimensions are interpreted as extents along X, Y, Z.
        # This means we need to be careful about how dimensions map to X, Y, Z.
        # Let's assume dims = (extent_x, extent_y, extent_z).
        
        min_corner = (
            loc_x - dims[0] / 2.0,
            loc_y - dims[1] / 2.0, # Assuming depth is along Y for placement on floor
            loc_z - dims[2] / 2.0
        )
        max_corner = (
            loc_x + dims[0] / 2.0,
            loc_y + dims[1] / 2.0,
            loc_z + dims[2] / 2.0
        )
        
        # Correcting the bounding box calculation based on typical asset dimensions and placement.
        # For a sofa, width is along X, depth along Z, height along Y.
        # For a fireplace, width along X, depth along Z, height along Y.
        # For a coffee table, width along X, depth along Z, height along Y.
        # For an area rug, width along X, depth along Z, height is negligible.
        
        # Let's redefine dimensions to be (extent_x, extent_z, extent_y) for horizontal placement.
        # And then use the actual height for the Y extent.
        
        # Re-interpreting asset_dims: (width, depth, height)
        # width -> X extent, depth -> Z extent, height -> Y extent
        
        half_width = dims[0] / 2.0
        half_depth = dims[1] / 2.0
        half_height = dims[2] / 2.0
        
        # The location is the center of the object.
        # The bounding box extents are relative to the center.
        
        min_corner = (
            loc_x - half_width,
            loc_y - half_height, # Assuming Y is the up axis, so height is along Y
            loc_z - half_depth
        )
        max_corner = (
            loc_x + half_width,
            loc_y + half_height,
            loc_z + half_depth
        )
        
        # If the asset is placed on the floor, its min_y should be 0.
        # Let's adjust the location and bounding box accordingly.
        # If loc_y is intended to be the base of the object, then:
        # min_corner_y = loc_y
        # max_corner_y = loc_y + height
        
        # Let's assume `location` is the center of the object.
        # And `loc_y` is the height of the center from the floor.
        # So, min_y = loc_y - height/2, max_y = loc_y + height/2.
        
        # For objects placed on the floor, like sofas and coffee tables,
        # their base should be at Y=0. So, loc_y should be height/2.
        
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            # These are typically placed on the floor.
            # Adjust loc_y to be the center of the object's height.
            loc_y = half_height
            min_corner = (
                loc_x - half_width,
                loc_y - half_height,
                loc_z - half_depth
            )
            max_corner = (
                loc_x + half_width,
                loc_y + half_height,
                loc_z + half_depth
            )
        
        asset_layouts[asset_name] = Layout(
            location=(loc_x, loc_y, loc_z),
            min=min_corner,
            max=max_corner,
            orientation=orientation
        )

    # Calculate the total score based on the relations.
    total_score = 0.0
    
    # Store layouts for relations that need multiple assets (e.g., AnchorsSeatingArrangement)
    seating_assets_layouts = {}
    
    # First, collect all layouts for relations that require them.
    for asset_name, rels in relations.items():
        for rel_name in rels:
            if rel_name == "AnchorsSeatingArrangement":
                seating_assets_layouts[asset_name] = [] # Initialize list for this asset
    
    # Populate seating_assets_layouts
    for asset_name, rels in relations.items():
        for rel_name in rels:
            if rel_name == "AnchorsSeatingArrangement":
                # If asset_name is the one anchoring, we need to find the seating assets.
                # In this specific graph, the Area Rug anchors the Sofa.
                # So, when scoring "AnchorsSeatingArrangement" for "Area Rug",
                # we need the layout of the "Sofa".
                if asset_name == "Area Rug":
                    if "Sofa" in asset_layouts:
                        seating_assets_layouts["Area Rug"].append(asset_layouts["Sofa"])
                # If the Sofa was anchoring something, we'd add it to its list.
                # For now, only Area Rug anchors the Sofa.

    # Calculate scores for each relation.
    for asset_name, rels in relations.items():
        asset_layout = asset_layouts[asset_name]
        
        for rel_name in rels:
            if rel_name in RELATION_SCORERS:
                scorer_info = RELATION_SCORERS[rel_name]
                scorer_func = scorer_info["func"]
                scorer_args_names = scorer_info["args"]
                
                # Prepare arguments for the scoring function.
                args = []
                for arg_name in scorer_args_names:
                    if arg_name == "asset_layout":
                        args.append(asset_layout)
                    elif arg_name == "target_layout":
                        # This requires knowing which asset is the target.
                        # We need to parse the relation definition.
                        # For example, "InFrontOf": [Sofa, Fireplace] means Sofa is the asset, Fireplace is the target.
                        # The `relations` dictionary structure is {asset_name: [relation_names]}.
                        # We need a way to map relation names to their targets.
                        
                        # Let's assume a convention: if a relation involves two assets,
                        # the first one in the list is the subject (asset_name) and the second is the object (target).
                        # This requires a more structured input for relations.
                        
                        # For now, let's hardcode the targets based on the problem description.
                        if rel_name == "InFrontOf" and asset_name == "Sofa":
                            target_asset_name = "Fireplace"
                        elif rel_name == "CenteredInFrontOf" and asset_name == "Coffee Table":
                            target_asset_name = "Sofa"
                        elif rel_name == "WithinReachOf" and asset_name == "Coffee Table":
                            target_asset_name = "Sofa"
                        elif rel_name == "Under" and asset_name == "Area Rug":
                            target_asset_name = "Coffee Table"
                        elif rel_name == "On" and asset_name == "Sofa Cushions":
                            target_asset_name = "Sofa"
                        elif rel_name == "DrapedOver" and asset_name == "Throw Blanket":
                            target_asset_name = "Sofa"
                        else:
                            # Default or error case if target is not defined.
                            target_asset_name = None
                        
                        if target_asset_name and target_asset_name in asset_layouts:
                            args.append(asset_layouts[target_asset_name])
                        else:
                            # If target is not found, this relation cannot be scored.
                            # Assign a score of 0 or handle as an error.
                            args.append(None) # Placeholder for missing target
                            
                    elif arg_name == "seating_assets":
                        # For relations like AnchorsSeatingArrangement, we need a list of seating assets.
                        # This is specific to the relation.
                        if asset_name in seating_assets_layouts:
                            args.append(seating_assets_layouts[asset_name])
                        else:
                            args.append([]) # Empty list if no seating assets found for this anchor.
                    else:
                        # Handle other potential arguments if needed.
                        pass
                
                # Ensure all required arguments are provided before calling the scorer.
                if all(arg is not None for arg in args):
                    try:
                        score = scorer_func(*args)
                        total_score += score
                    except Exception as e:
                        print(f"Error scoring relation {rel_name} for {asset_name}: {e}")
                        # Assign a low score if an error occurs.
                        total_score += 0.0
                else:
                    # If a required argument was missing (e.g., target_layout), skip scoring.
                    pass
            else:
                print(f"Warning: Relation '{rel_name}' not found in RELATION_SCORERS.")

    return total_score

# --- Main Execution ---

if __name__ == "__main__":
    import math
    
    # Define the assets and their relations based on the problem description.
    asset_names = [
        "Fireplace",
        "Sofa",
        "Coffee Table",
        "Area Rug",
        "Sofa Cushions",
        "Throw Blanket"
    ]
    
    # The graph structure: {asset_name: [list_of_relation_names_it_satisfies]}
    # This is a simplified representation. A more complete graph would define
    # the relation type and the target asset(s).
    # For this problem, we'll infer targets based on relation names and common sense.
    
    relations = {
        "Fireplace": ["AgainstWall", "FocalPoint", "EmitsGlow", "HoldsDecorativeItems"],
        "Sofa": ["InFrontOf", "AnchorsSeatingArrangement"], # Sofa is anchored by rug, and anchors the coffee table.
        "Coffee Table": ["CenteredInFrontOf", "WithinReachOf", "On"], # Coffee table is on the rug.
        "Area Rug": ["Under", "AnchorsSeatingArrangement", "InFrontOf"], # Rug is under coffee table, anchors seating, in front of sofa.
        "Sofa Cushions": ["On"],
        "Throw Blanket": ["DrapedOver"]
    }
    
    # Create an Optuna study.
    study = optuna.create_study(direction="maximize")
    
    # Run the optimization.
    # We'll run a limited number of trials for demonstration.
    num_trials = 100
    study.optimize(lambda trial: objective(trial, asset_names, relations), n_trials=num_trials)
    
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
        half_width = dims[0] / 2.0
        half_depth = dims[1] / 2.0
        half_height = dims[2] / 2.0
        
        loc_x = best_params.get(f"{asset_name}_loc_x", 0.0)
        loc_z = best_params.get(f"{asset_name}_loc_z", 0.0)
        
        # Adjust loc_y for floor placement
        if asset_name in ["Fireplace", "Sofa", "Coffee Table", "Area Rug"]:
            loc_y = half_height
        else:
            loc_y = best_params.get(f"{asset_name}_loc_y", 0.0) # For items like cushions, their placement might be relative.
            
        orientation_y = best_params.get(f"{asset_name}_orient_y", 0.0)
        orientation = (0.0, orientation_y, 0.0)
        
        min_corner = (
            loc_x - half_width,
            loc_y - half_height,
            loc_z - half_depth
        )
        max_corner = (
            loc_x + half_width,
            loc_y + half_height,
            loc_z + half_depth
        )
        
        best_asset_layouts[asset_name] = Layout(
            location=(loc_x, loc_y, loc_z),
            min=min_corner,
            max=max_corner,
            orientation=orientation
        )

    # Save the optimal layout to a JSON file.
    # Get the directory of the current script.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    layout_file_path = os.path.join(script_dir, "layout.json")
    
    # Convert Layout dataclasses to dictionaries for JSON serialization.
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