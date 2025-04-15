import bpy
from mathutils import Vector
import os
import shutil
import math
from typing import Tuple
import re
import yaml


def select_collection(collections):
    bpy.ops.object.select_all(action="DESELECT")
    for collection in collections:
        for object in bpy.data.collections[collection].objects:
            object.select_set(True)


def save_mitsuba_xml(folder_dir, filename, collections):
    filepath = os.path.join(folder_dir, f"{filename}.xml")
    bpy.ops.object.select_all(action="DESELECT")
    select_collection(collections)
    bpy.ops.export_scene.mitsuba(
        filepath=filepath,
        check_existing=True,
        filter_glob="*.xml",
        use_selection=True,
        split_files=False,
        export_ids=True,
        ignore_background=True,
        axis_forward="Y",
        axis_up="Z",
    )


def get_center_bbox(tile: bpy.types.Object) -> Vector:
    """Get the center of the bounding box of the tile."""
    local_bbox_center = 0.125 * sum((Vector(b) for b in tile.bound_box), Vector())
    global_bbox_center = tile.matrix_world @ local_bbox_center
    return global_bbox_center


def compute_rot_angle(pt1: list, pt2: list) -> Tuple[float, float, float]:
    """Compute the rotation angles for vector pt1 to pt2."""
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    z = pt2[2] - pt1[2]

    return cartesian2spherical(x, y, z)


def cartesian2spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical2cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    # theta: zenith angle (0, pi), phi: azimuthal angle (0, 2pi)
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def constraint_angle(angle: float, low: float, high: float) -> float:
    """
    Constrain the angle within the angle_delta.

    Args:
    angle: float
        The angle to be constrained. Unit: radians.
    low: float
        The minimum angle delta. Unit: radians.
    high: float
        The maximum angle delta. Unit: radians.

    Returns:
    float
        The constrained angle. Unit: radians.
    """
    angle = low if angle <= low else angle
    angle = high if angle >= high else angle
    return angle


def mkdir_with_replacement(folder_dir):
    if os.path.exists(folder_dir):
        shutil.rmtree(folder_dir)
    os.makedirs(folder_dir)
