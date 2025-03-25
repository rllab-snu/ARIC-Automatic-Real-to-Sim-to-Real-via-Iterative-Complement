import math
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

TARGET_VIEW = np.array([
    [-1.00, 0.00, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, -1.00],
])

def remove_unvalid_points(object_points, camera_path):
    points_origin = np.copy(object_points[:,:3])
    norms_origin = np.copy(object_points[:,3:6])

    with open(camera_path, "r") as c_info:
        camera_info = json.load(c_info)

    camera_mat_z, z_view_idx = None, 0
    for curr_camera in camera_info:
        if int(curr_camera["img_name"]) == z_view_idx:
            camera_mat_z = np.array(curr_camera["rotation"])
    assert camera_mat_z is not None

    rot_mat0 = np.matmul(TARGET_VIEW, np.linalg.inv(camera_mat_z))

    points_rotated = np.dot(rot_mat0, np.transpose(points_origin))
    points_rotated = np.transpose(points_rotated)

    norms_rotated = np.dot(rot_mat0, np.transpose(norms_origin))
    norms_rotated = np.transpose(norms_rotated)

    points_tmp = np.copy(points_rotated[:,2])
    points_tmp = np.sort(points_tmp)

    percentage_ = 0.001
    thres_idx = int(len(points_tmp)*percentage_)
    z_threshold = points_tmp[thres_idx]

    points_valid = np.where(points_rotated[:,2]>=z_threshold)[0]
    points_copied = np.copy(object_points)[points_valid]
    points_copied[:,:3] = points_rotated[points_valid]
    points_copied[:,3:6] = norms_rotated[points_valid]
    return points_copied
