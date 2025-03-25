import math
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation   


from PIL import Image
from plyfile import PlyData, PlyElement

TARGET_VIEW = np.array([
    [-1.00, 0.00, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, -1.00],
])

# CAMERA_TARGET_VIEW = np.array([
#     [-1.00, 0.00, 0.00],
#     [0.00, -1.00, 0.00],
#     [0.00, 0.00, 1.00],
# ])

def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """
    EPS = np.finfo(float).eps * 4.

    q = np.array(quaternion, dtype=np.float32, copy=True)[[3, 0, 1, 2]]
    n = np.dot(q, q)
    if n < EPS: return np.identity(3)

    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )

def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))

    points = np.array(points)
    return points

def normalize_pcd(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance
	return points


def project_camera2sphere(camera_path, fibonacci_points, z_view_idx, print_results=False):
    with open(camera_path, "r") as c_info:
        camera_info = json.load(c_info)

    camera_z_mat = np.array(camera_info[z_view_idx]["rotation"])
    camera_z_mat_inv = np.linalg.inv(camera_z_mat)

    camera_pose_list = []
    for i, curr_camera in enumerate(camera_info):
        camera_mat = np.array(curr_camera["rotation"])
        camera_mat = np.matmul(camera_z_mat_inv, camera_mat)
        camera_mat = np.matmul(TARGET_VIEW, camera_mat)
        new_point = np.matmul(camera_mat, np.array([[0.0],[0.0],[-1.0]]))
        # if not i in [0,1,2,3,4,5,6,19,20,21,22,23,24,25,26,27,28,29,30,31,32,45,46,47,48,49,50,51]: continue
        camera_pose_list.append(np.reshape(new_point, -1))

    camera_pose_list = np.array(camera_pose_list)

    # camera_pose = o3d.geometry.PointCloud()
    # camera_pose.points = o3d.utility.Vector3dVector(camera_pose_list)
    # o3d.io.write_point_cloud("{}/cameras.ply".format(ply_path), camera_pose)

    n_points_per_area_threshold = 1
    r_area = 0.60
    n_finonacci_samples = fibonacci_points.shape[0]

    selected_points_shperal, score_list = [], []
    for i in range(n_finonacci_samples):
        curr_area_point_ = fibonacci_points[i]
        copied_camera = np.copy(camera_pose_list)
        dist_ = np.linalg.norm(copied_camera-np.reshape(curr_area_point_, (1,-1)), axis=-1)

        n_counts = np.sum(np.where(dist_<=r_area, True, False))
        if n_counts >= n_points_per_area_threshold:
            selected_points_shperal.append(curr_area_point_)

        score_ = min(max(n_counts/n_points_per_area_threshold, 0.0), 1.0)
        score_list.append(score_)

    score_list = np.array(score_list)
    selected_points_shperal = np.array(selected_points_shperal)

    # pcd_spheral_selected = o3d.geometry.PointCloud()
    # pcd_spheral_selected.points = o3d.utility.Vector3dVector(selected_points_shperal)
    # o3d.io.write_point_cloud("{}/camera_selected{}.ply".format(ply_path, n_finonacci_samples), pcd_spheral_selected)

    if print_results:
        print()
        print("* CAMERA SCORE1: {:.4f}".format(np.mean(score_list)))
        print("* CAMERA SCORE2: {:.4f}".format(len(selected_points_shperal)/n_finonacci_samples))
    return selected_points_shperal, score_list


def project_pcd2sphere(object_points, camera_path, fibonacci_points, z_view_idx, print_results=False):
    # ply_data = "{}/point_cloud.ply".format(ply_path)
    # pcd_origin = o3d.io.read_point_cloud(ply_data)
    # points_origin = np.asarray(pcd_origin.points)
    points_origin = np.copy(object_points[:,:3])
    points_normed = normalize_pcd(points_origin)

    with open(camera_path, "r") as c_info:
        camera_info = json.load(c_info)

    camera_mat_z = None
    for curr_camera in camera_info:
        if int(curr_camera["img_name"]) == z_view_idx:
            camera_mat_z = np.array(curr_camera["rotation"])
    assert camera_mat_z is not None

    rot_mat0 = np.matmul(TARGET_VIEW, np.linalg.inv(camera_mat_z))

    points_rotated = np.dot(rot_mat0, np.transpose(points_normed))
    points_rotated = np.transpose(points_rotated)

    # pcd_rotated = o3d.geometry.PointCloud()
    # pcd_rotated.points = o3d.utility.Vector3dVector(points_rotated)
    # o3d.io.write_point_cloud("{}/point_cloud_rot.ply".format(ply_path), pcd_rotated)

    points_shperal = points_rotated/np.linalg.norm(points_rotated, axis=1, keepdims=True)

    # pcd_spheral = o3d.geometry.PointCloud()
    # pcd_spheral.points = o3d.utility.Vector3dVector(points_shperal)
    # o3d.io.write_point_cloud("{}/point_cloud_sph.ply".format(ply_path), pcd_spheral)

    n_points_per_area_threshold = 8 # int(n_points/finonacci_samples*0.5)
    r_area = 0.15
    n_finonacci_samples = fibonacci_points.shape[0]

    selected_points_shperal, score_list = [], []
    for i in range(n_finonacci_samples):
        curr_area_point_ = fibonacci_points[i]
        copied_shperal = np.copy(points_shperal)
        dist_ = np.linalg.norm(copied_shperal-np.reshape(curr_area_point_, (1,-1)), axis=-1)

        n_counts = np.sum(np.where(dist_<=r_area, True, False))
        if n_counts >= n_points_per_area_threshold:
            selected_points_shperal.append(curr_area_point_)

        score_ = min(max(n_counts/n_points_per_area_threshold, 0.0), 1.0)
        score_list.append(score_)

    score_list = np.array(score_list)
    selected_points_shperal = np.array(selected_points_shperal)

    # pcd_spheral_selected = o3d.geometry.PointCloud()
    # pcd_spheral_selected.points = o3d.utility.Vector3dVector(selected_points_shperal)
    # o3d.io.write_point_cloud("{}/point_cloud_selected{}.ply".format(ply_path, n_finonacci_samples), pcd_spheral_selected)

    if print_results:
        print()
        print("* PCD SCORE1: {:.4f}".format(np.mean(score_list)))
        print("* PCD SCORE2: {:.4f}".format(len(selected_points_shperal)/n_finonacci_samples))
    return selected_points_shperal, score_list

def get_pcd_score(
        object_points,
        camera_path,
        pcd_path,
        n_finonacci_samples=1000,
        consider_cameras=False,
        save_pcd=True,
    ):
    do_resample = False
    num_resample = 5000
    if do_resample and len(object_points) > num_resample:
        object_points = np.copy(object_points)
        np.random.shuffle(object_points)
        object_points = object_points[:num_resample]

    points_shperal_uniform = fibonacci_sphere(n_finonacci_samples)
    z_view_idx = 0

    selected_points1, scores1 = project_pcd2sphere(
        object_points, camera_path, points_shperal_uniform, z_view_idx
    )
    if consider_cameras:
        selected_points2, scores2 = project_camera2sphere(
            camera_path, points_shperal_uniform, z_view_idx
        )
        total_selected = np.concatenate((selected_points1,selected_points2), axis=0)
        total_selected = np.unique(total_selected, axis=0)
    else:
        total_selected = np.copy(selected_points1)

    if save_pcd:
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(object_points[:,:3])
        pcd_name = "{}/voxel_points.ply".format(pcd_path)
        pcd_tmp = pcd_tmp.voxel_down_sample(0.02)
        o3d.io.write_point_cloud(pcd_name, pcd_tmp)

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(selected_points1[:,:3])
        pcd_name = "{}/projected_object_points.ply".format(pcd_path)
        o3d.io.write_point_cloud(pcd_name, pcd_tmp)

        if consider_cameras:
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(selected_points2[:,:3])
            pcd_name = "{}/projected_camera_points.ply".format(pcd_path)
            o3d.io.write_point_cloud(pcd_name, pcd_tmp)

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(total_selected[:,:3])
        pcd_name = "{}/projected_total_points.ply".format(pcd_path)
        o3d.io.write_point_cloud(pcd_name, pcd_tmp)

    n_selected = len(total_selected)
    return n_selected / n_finonacci_samples
    

if __name__ == "__main__":
    ply_name = "bowl2" 
    obj_idx = 1
    z_view_idx = 6

    data_path = "/home/minjae/Desktop/Real2Sim/Dataset"
    ply_path = data_path+"/{}/object_{}/point_cloud/iteration_15000".format(ply_name,obj_idx)
    camera_path = data_path+"/{}/object_{}/cameras.json".format(ply_name,obj_idx)

    finonacci_samples = 1000
    points_shperal_uniform = fibonacci_sphere(finonacci_samples)

    selected_points1, scores1 = project_pcd2sphere(ply_path, camera_path, points_shperal_uniform, z_view_idx)
    selected_points2, scores2 = project_camera2sphere(ply_path, camera_path, points_shperal_uniform, z_view_idx)

    total_selected = np.concatenate((selected_points1,selected_points2), axis=0)
    total_selected = np.unique(total_selected, axis=0)
    n_selected = len(total_selected)

    print()
    print("* FINAL SCORE: {:.4f}".format(n_selected/finonacci_samples))

