import bpy
import mathutils
import math

def normalize_scene():
    '''
    Normalize all objects to fit within a unit cube centered at the origin.
    '''
    # --- Normalize to unit cube ---
    # Compute bounding box of all objects in world space
    coords = []
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
            for v in obj.bound_box:
                coords.append(obj.matrix_world @ mathutils.Vector(v))
    if not coords:
        return

    min_corner = [min(c[i] for c in coords) for i in range(3)]
    max_corner = [max(c[i] for c in coords) for i in range(3)]

    size = [max_corner[i] - min_corner[i] for i in range(3)]
    max_dim = max(size)

    # Scale factor to fit into unit cube
    scale = 1.0 / max_dim if max_dim > 0 else 1.0

    # Center of the bounding box
    center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]

    # Apply transform: move to origin and scale
    for obj in bpy.context.scene.objects:
        obj.location = (obj.location[0] - center[0],
                        obj.location[1] - center[1],
                        obj.location[2] - center[2])
        obj.location = [coord * scale for coord in obj.location]
        obj.scale = [s * scale for s in obj.scale]

def rotate_and_normalize_scene():
    '''
    Rotate the entire scene 90 degrees around the global X-axis, then normalize all objects to fit within a unit cube centered at the origin.
    '''
    # --- Rotate everything as a block around global X ---
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
    empty = bpy.context.active_object
    objs = [obj for obj in bpy.context.scene.objects if obj != empty]

    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.parent_set(type='OBJECT')

    empty.rotation_euler[0] += math.radians(90)

    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    bpy.data.objects.remove(empty, do_unlink=True)

    # --- Normalize to unit cube ---
    # Compute bounding box of all objects in world space
    coords = []
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
            for v in obj.bound_box:
                coords.append(obj.matrix_world @ mathutils.Vector(v))
    if not coords:
        return

    min_corner = [min(c[i] for c in coords) for i in range(3)]
    max_corner = [max(c[i] for c in coords) for i in range(3)]

    size = [max_corner[i] - min_corner[i] for i in range(3)]
    max_dim = max(size)

    # Scale factor to fit into unit cube
    scale = 1.0 / max_dim if max_dim > 0 else 1.0

    # Center of the bounding box
    center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]

    # Apply transform: move to origin and scale
    for obj in bpy.context.scene.objects:
        obj.location = (obj.location[0] - center[0],
                        obj.location[1] - center[1],
                        obj.location[2] - center[2])
        obj.location = [coord * scale for coord in obj.location]
        obj.scale = [s * scale for s in obj.scale]
