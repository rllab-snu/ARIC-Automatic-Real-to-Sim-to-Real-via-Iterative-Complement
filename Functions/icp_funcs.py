import math
import copy
import random
import numpy as np
import open3d as o3d

from PIL import Image
from plyfile import PlyData, PlyElement


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_pcd, target_pcd, voxel_size):
    # print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source_pcd.transform(trans_init)
    # draw_registration_result(source_pcd, target_pcd, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    return source_pcd, target_pcd, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)    
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def normalize_pcd(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	# furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	# points /= furthest_distance
	return points


def apply_ransac(source_points, target_points, scale_=1.0, voxel_size=0.03, do_sampling=True, view_result=False):
    source_points = np.copy(source_points)
    target_points = np.copy(target_points)

    if scale_ != 1.0:
        scale_mat = np.identity(3) * scale_
        target_points = np.dot(scale_mat, np.transpose(target_points))
        target_points = np.transpose(target_points)

    if do_sampling:
        sample_points = 100000

        sampled_idxes = sorted(random.sample(range(len(source_points)), sample_points))
        source_points = source_points[sampled_idxes]

        sampled_idxes = sorted(random.sample(range(len(target_points)), sample_points))
        target_points = target_points[sampled_idxes]

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source_pcd, target_pcd, voxel_size
    )

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    trans_init = result_ransac.transformation

    if view_result:
        draw_registration_result(source_down, target_down, trans_init)

    return source, target, source_down, target_down, trans_init

def apply_icp(source_points, target_points, trans_init, view_result=False):
    icp_threshold = 0.02

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_points, source_points, icp_threshold, trans_init,
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    if view_result:
        draw_registration_result(source_points, target_points, reg_p2p.transformation)

    pcd_info = [source_points, target_points]
    target2source_mat = np.copy(reg_p2p.transformation)
    trans_score = o3d.pipelines.registration.evaluate_registration(
        source_points, target_points, icp_threshold, reg_p2p.transformation
    )
    return pcd_info, target2source_mat, trans_score

def apply_lepard(source_points, target_points, scale_=1.0, voxel_size=0.03, do_sampling=True, view_result=False):
    source_points = np.copy(source_points)
    target_points = np.copy(target_points)

    if do_sampling:
        sample_points = 100000

        sampled_idxes = sorted(random.sample(range(len(source_points)), sample_points))
        source_points = source_points[sampled_idxes]

        sampled_idxes = sorted(random.sample(range(len(target_points)), sample_points/2))
        target_points = target_points[sampled_idxes]

    if scale_ != 1.0:
        scale_mat = np.identity(3) / scale_
        source_points = np.dot(scale_mat, np.transpose(source_points))
        source_points = np.transpose(source_points)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source_pcd, target_pcd, voxel_size
    )
    
    from Functions.lepard_funcs import find_best_transformation_lepard
    trans_init, trans_score = find_best_transformation_lepard(
        source_points[:,:3],
        target_points[:,:3],
    )

    if view_result:
        draw_registration_result(source_down, target_down, trans_init)

    return source, target, source_down, target_down, trans_init, trans_score


   



if __name__ == "__main__":
    # ply0_name = "object21_edeedaab-4" # coil

    ply0_name = "object24_45e48f8e-1" # haunter0
    ply1_name = "object25_73364406-0" # haunter1

    ply0_path = "./output/test516/{}/point_cloud/iteration_15000".format(ply0_name)
    ply1_path = "./output/test516/{}/point_cloud/iteration_15000".format(ply1_name)

    perform_icp(ply0_path, ply1_path)



