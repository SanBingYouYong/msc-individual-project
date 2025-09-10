import bpy
import bmesh
import mathutils
import math
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree


def get_mesh_volume_bmesh(obj):
    """
    Calculate the volume of a mesh object using bmesh.
    
    Args:
        obj: Blender mesh object
        
    Returns:
        float: Volume of the mesh
    """
    # Create a new bmesh instance from the mesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    # Apply the object's transformation matrix
    bm.transform(obj.matrix_world)
    
    # Ensure the mesh has face indices
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    
    # Calculate volume using bmesh
    volume = bm.calc_volume()
    
    # Clean up
    bm.free()
    
    return abs(volume)  # Return absolute value to handle negative volumes


def create_boolean_intersection(obj1, obj2):
    """
    Create a boolean intersection of two mesh objects to calculate actual overlap.
    
    Args:
        obj1, obj2: Blender mesh objects
        
    Returns:
        float: Volume of the intersection
    """
    # Create copies of the objects to avoid modifying originals
    bpy.ops.object.select_all(action='DESELECT')
    
    # Duplicate first object
    obj1.select_set(True)
    bpy.context.view_layer.objects.active = obj1
    bpy.ops.object.duplicate()
    obj1_copy = bpy.context.active_object
    obj1_copy.name = f"{obj1.name}_temp_copy1"
    
    # Duplicate second object
    bpy.ops.object.select_all(action='DESELECT')
    obj2.select_set(True)
    bpy.context.view_layer.objects.active = obj2
    bpy.ops.object.duplicate()
    obj2_copy = bpy.context.active_object
    obj2_copy.name = f"{obj2.name}_temp_copy2"
    
    # Select the first copy and add boolean modifier
    bpy.ops.object.select_all(action='DESELECT')
    obj1_copy.select_set(True)
    bpy.context.view_layer.objects.active = obj1_copy
    
    # Add boolean modifier for intersection
    modifier = obj1_copy.modifiers.new(name="Boolean_Intersect", type='BOOLEAN')
    modifier.operation = 'INTERSECT'
    modifier.object = obj2_copy
    modifier.solver = 'FAST'  # Use exact solver for better precision
    
    # Apply the modifier
    try:
        bpy.ops.object.modifier_apply(modifier="Boolean_Intersect")
        
        # Calculate volume of intersection
        intersection_volume = get_mesh_volume_bmesh(obj1_copy)
        
    except Exception as e:
        print(f"Boolean operation failed: {e}")
        intersection_volume = 0.0
    
    # Clean up temporary objects
    bpy.data.objects.remove(obj1_copy, do_unlink=True)
    bpy.data.objects.remove(obj2_copy, do_unlink=True)
    
    return intersection_volume


def calculate_voxel_based_iou(obj1, obj2, resolution=64):
    """
    Calculate IoU using voxel-based sampling for more accurate results.
    This method samples points in 3D space and checks if they're inside each mesh.
    
    Args:
        obj1, obj2: Blender mesh objects
        resolution: Number of samples per dimension
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Get world-space bounding boxes
    def get_world_bbox(obj):
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        min_coords = Vector([min(corner[i] for corner in bbox_corners) for i in range(3)])
        max_coords = Vector([max(corner[i] for corner in bbox_corners) for i in range(3)])
        return min_coords, max_coords
    
    min1, max1 = get_world_bbox(obj1)
    min2, max2 = get_world_bbox(obj2)
    
    # Get combined bounding box
    combined_min = Vector([min(min1[i], min2[i]) for i in range(3)])
    combined_max = Vector([max(max1[i], max2[i]) for i in range(3)])
    combined_size = combined_max - combined_min
    
    # Create BVH trees for efficient point-in-mesh testing
    def create_bvh_tree(obj):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        return bvh
    
    bvh1 = create_bvh_tree(obj1)
    bvh2 = create_bvh_tree(obj2)
    
    # Sample points in the voxel grid
    step = [combined_size[i] / resolution for i in range(3)]
    
    intersection_count = 0
    union_count = 0
    
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # Calculate voxel center
                point = Vector([
                    combined_min[0] + (i + 0.5) * step[0],
                    combined_min[1] + (j + 0.5) * step[1],
                    combined_min[2] + (k + 0.5) * step[2]
                ])
                
                # Check if point is inside each mesh using ray casting
                # Cast rays in multiple directions to improve accuracy
                directions = [
                    Vector((1, 0, 0)),
                    Vector((-1, 0, 0)),
                    Vector((0, 1, 0)),
                    Vector((0, -1, 0)),
                    Vector((0, 0, 1)),
                    Vector((0, 0, -1))
                ]
                
                def is_point_inside_mesh(bvh, point):
                    inside_count = 0
                    for direction in directions:
                        hit, _, _, _ = bvh.ray_cast(point, direction)
                        if hit:
                            inside_count += 1
                    # Point is inside if odd number of intersections
                    return inside_count % 2 == 1
                
                in_obj1 = is_point_inside_mesh(bvh1, point)
                in_obj2 = is_point_inside_mesh(bvh2, point)
                
                if in_obj1 and in_obj2:
                    intersection_count += 1
                
                if in_obj1 or in_obj2:
                    union_count += 1
    
    if union_count == 0:
        return 0.0
    
    return intersection_count / union_count


def calculate_mesh_iou_precise(obj1, obj2, method='boolean'):
    """
    Calculate precise mesh IoU using either boolean operations or voxel sampling.
    
    Args:
        obj1, obj2: Blender mesh objects
        method: 'boolean' for boolean operations, 'voxel' for voxel sampling
        
    Returns:
        float: IoU value between 0 and 1
    """
    if method == 'boolean':
        # Calculate volumes
        vol1 = get_mesh_volume_bmesh(obj1)
        vol2 = get_mesh_volume_bmesh(obj2)
        
        # Calculate intersection volume using boolean operations
        intersection_vol = create_boolean_intersection(obj1, obj2)
        
        # Calculate IoU
        union_vol = vol1 + vol2 - intersection_vol
        
        if union_vol == 0:
            return 0.0
        
        return intersection_vol / union_vol
    
    elif method == 'voxel':
        return calculate_voxel_based_iou(obj1, obj2)
    
    else:
        raise ValueError("Method must be 'boolean' or 'voxel'")


def calculate_iou_from_obj_files(obj_path1, obj_path2, test_rotations=True, rotation_steps=4):
    """
    Calculate IoU between two OBJ files, testing different Z-axis rotations to find best overlap.
    
    Args:
        obj_path1 (str): Path to the first OBJ file
        obj_path2 (str): Path to the second OBJ file
        test_rotations (bool): Whether to test different Z-axis rotations
        rotation_steps (int): Number of rotation steps to test (e.g., 4 = 0°, 90°, 180°, 270°)
        
    Returns:
        dict: Dictionary containing:
            - 'best_iou': Best IoU value found
            - 'best_rotation': Best rotation angle in degrees
            - 'all_results': List of (rotation_degrees, iou_value) tuples
            Returns None if calculation fails
    """
    try:
        bpy.ops.wm.read_homefile(use_empty=True)
        # Clear the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        # Import first OBJ file
        bpy.ops.wm.obj_import(filepath=obj_path1)
        imported_objects_1 = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        
        if not imported_objects_1:
            print(f"Error: No mesh objects found in {obj_path1}")
            return None
        
        # Join all objects from first import into one
        if len(imported_objects_1) > 1:
            bpy.context.view_layer.objects.active = imported_objects_1[0]
            bpy.ops.object.join()
        
        obj1 = bpy.context.active_object
        obj1.name = "Object1"
        
        # Import second OBJ file
        bpy.ops.wm.obj_import(filepath=obj_path2)
        imported_objects_2 = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH' and obj != obj1]
        
        if not imported_objects_2:
            print(f"Error: No mesh objects found in {obj_path2}")
            return None
        
        # Join all objects from second import into one
        if len(imported_objects_2) > 1:
            # Deselect first object and select only second import objects
            obj1.select_set(False)
            for obj in imported_objects_2:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = imported_objects_2[0]
            bpy.ops.object.join()
        
        obj2 = bpy.context.active_object
        obj2.name = "Object2"
        
        # Store original rotation of obj2
        original_rotation_z = obj2.rotation_euler[2]
        
        # Calculate individual volumes
        vol1 = get_mesh_volume_bmesh(obj1)
        vol2 = get_mesh_volume_bmesh(obj2)
        
        if not test_rotations:
            # Original behavior - no rotation testing
            try:
                iou = calculate_mesh_iou_precise(obj1, obj2, method='boolean')
                return {'best_iou': iou, 'best_rotation': 0, 'all_results': [(0, iou)]}
            except Exception as e:
                print(f"Boolean method failed for {obj_path1} and {obj_path2}: {e}")
                try:
                    iou = calculate_mesh_iou_precise(obj1, obj2, method='voxel')
                    return {'best_iou': iou, 'best_rotation': 0, 'all_results': [(0, iou)]}
                except Exception as e2:
                    print(f"Voxel method also failed: {e2}")
                    return None
        
        # Test different rotations
        best_iou = 0.0
        best_rotation = 0
        all_results = []
        
        rotation_step_degrees = 360 / rotation_steps
        
        for step in range(rotation_steps):
            rotation_degrees = step * rotation_step_degrees
            
            # Reset obj2 to original rotation and apply new rotation
            obj2.rotation_euler[2] = original_rotation_z + math.radians(rotation_degrees)
            
            # Update the object's transform
            bpy.context.view_layer.update()
            
            try:
                # Try boolean method first
                iou = calculate_mesh_iou_precise(obj1, obj2, method='boolean')
            except Exception as e:
                try:
                    # Fallback to voxel method
                    iou = calculate_mesh_iou_precise(obj1, obj2, method='voxel')
                except Exception as e2:
                    print(f"Both methods failed at rotation {rotation_degrees}°: {e2}")
                    iou = 0.0
            
            all_results.append((rotation_degrees, iou))
            
            if iou > best_iou:
                best_iou = iou
                best_rotation = rotation_degrees
            
            # print(f"Rotation {rotation_degrees:6.1f}°: IoU = {iou:.6f}")
        
        # Restore original rotation
        obj2.rotation_euler[2] = original_rotation_z
        bpy.context.view_layer.update()
        
        # print(f"\nBest result: {best_iou:.6f} at {best_rotation}° rotation")
        
        return {
            'best_iou': best_iou,
            'best_rotation': best_rotation,
            'all_results': all_results
        }
    
    except Exception as e:
        print(f"Error in calculate_iou_from_obj_files: {e}")
        return None


def calculate_iou_from_obj_files_simple(obj_path1, obj_path2, test_rotations=True, rotation_steps=4):
    """
    Calculate IoU between two OBJ files, returning just the best IoU value for backward compatibility.
    
    Args:
        obj_path1 (str): Path to the first OBJ file
        obj_path2 (str): Path to the second OBJ file
        test_rotations (bool): Whether to test different Z-axis rotations
        rotation_steps (int): Number of rotation steps to test (e.g., 4 = 0°, 90°, 180°, 270°)
        
    Returns:
        float: Best IoU value between 0 and 1, or None if calculation fails
    """
    result = calculate_iou_from_obj_files(obj_path1, obj_path2, test_rotations, rotation_steps)
    return result['best_iou'] if result else None


def main():
    """
    Main function to calculate IoU between selected and active objects in Blender.
    """
    # Get the active object
    active_obj = bpy.context.active_object
    if not active_obj:
        print("Error: No active object found!")
        return
    
    # Get selected objects (excluding the active one)
    selected_objs = [obj for obj in bpy.context.selected_objects if obj != active_obj]
    
    if not selected_objs:
        print("Error: No other selected objects found!")
        return
    
    if len(selected_objs) > 1:
        print(f"Warning: Multiple objects selected ({len(selected_objs)}), using the first one.")
    
    selected_obj = selected_objs[0]
    
    # Check if both objects are meshes
    if active_obj.type != 'MESH' or selected_obj.type != 'MESH':
        print("Error: Both objects must be mesh objects!")
        return
    
    print(f"Calculating IoU between:")
    print(f"  Active object: {active_obj.name}")
    print(f"  Selected object: {selected_obj.name}")
    
    # Calculate individual volumes
    vol1 = get_mesh_volume_bmesh(active_obj)
    vol2 = get_mesh_volume_bmesh(selected_obj)
    print(f"Individual volumes:")
    print(f"  {active_obj.name}: {vol1:.6f}")
    print(f"  {selected_obj.name}: {vol2:.6f}")
    
    # Calculate IoU using boolean operations (more precise)
    try:
        print("\nCalculating IoU using boolean intersection...")
        boolean_iou = calculate_mesh_iou_precise(active_obj, selected_obj, method='boolean')
        print(f"Boolean-based IoU: {boolean_iou:.6f}")
    except Exception as e:
        print(f"Boolean method failed: {e}")
        boolean_iou = None
    
    # Calculate IoU using voxel sampling (fallback method)
    try:
        print("\nCalculating IoU using voxel sampling...")
        voxel_iou = calculate_mesh_iou_precise(active_obj, selected_obj, method='voxel')
        print(f"Voxel-based IoU: {voxel_iou:.6f}")
    except Exception as e:
        print(f"Voxel method failed: {e}")
        voxel_iou = None
    
    # Return the results
    result = {
        'active_object': active_obj.name,
        'selected_object': selected_obj.name,
        'volume_1': vol1,
        'volume_2': vol2,
        'boolean_iou': boolean_iou,
        'voxel_iou': voxel_iou
    }
    
    return result


if __name__ == "__main__":
    # Example usage with OBJ files:
    # 
    # Test with rotation (new behavior):
    # result = calculate_iou_from_obj_files("/path/to/object1.obj", "/path/to/object2.obj")
    # if result:
    #     print(f"Best IoU: {result['best_iou']:.6f} at {result['best_rotation']}° rotation")
    #     print(f"All results: {result['all_results']}")
    #
    # For backward compatibility (just get the best IoU value):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obj_path1 = os.path.join(script_dir, "original.obj")
    obj_path2 = os.path.join(script_dir, "rotated90.obj")
    iou_value = calculate_iou_from_obj_files_simple(obj_path1, obj_path2)
    print(f"Best IoU between objects: {iou_value}")
    #
    # Disable rotation testing:
    # result = calculate_iou_from_obj_files("/path/to/object1.obj", "/path/to/object2.obj", test_rotations=False)
    # print(f"IoU without rotation testing: {result['best_iou']}")
    
    # Run the main function for selected objects in Blender
    # result = main()
    # if result:
    #     print(f"\nFinal Results: {result}")