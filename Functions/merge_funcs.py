import numpy as np
from Functions.icp_funcs import apply_icp, apply_ransac, apply_lepard, draw_registration_result

def find_ransac_transformation(
        source_points,
        target_points,
        voxel_size,
        scale_list,
    ):
    min_scale, max_scale, scale_int = scale_list
    n_tries, curr_scale = 3, min_scale

    best_score, best_scale, best_mat, best_results = -1.0, -1.0, None, None
    num_tries = int((max_scale-min_scale)/scale_int) + 1
    achieved_tries = 0

    while curr_scale <= max_scale:
        score_list = []
        tmp_best_score, tmp_best_results = -1.0, None

        for _ in range(n_tries):
            source_tmp, target_tmp = np.copy(source_points), np.copy(target_points)
            
            source, target, source_down_tmp, target_down_tmp, trans_init = apply_ransac(
                source_tmp, target_tmp,
                scale_=curr_scale, voxel_size=voxel_size,
                do_sampling=False, view_result=False
            )

            pcd_list, t2s_transform, t2s_score = apply_icp(
                source, target, trans_init,
                view_result=False
            )

            curr_score = float(t2s_score.fitness)
            if tmp_best_score < curr_score:
                tmp_best_score = curr_score
                tmp_best_results = [pcd_list, t2s_transform, t2s_score, curr_scale]

            score_list.append(curr_score)

        mean_score = np.mean(score_list)
        if best_score <= mean_score:
            best_score = mean_score
            best_scale = curr_scale
            best_mat = t2s_transform
            best_results = tmp_best_results

        curr_scale += scale_int
        achieved_tries += 1

        report_ = "  [{}/{}] BEST SCORE: {:.5f} | BEST SCALE: {:.3f}".format(achieved_tries, num_tries, best_score, best_scale)
        print(report_, end="\r")
    print(report_)
    return best_results

def merge_two_pointclouds(
        source_data, target_data,
        source_camera_path, target_camera_path,
        merge_method="lepard",
    ):
    source_data = np.copy(source_data)
    target_data = np.copy(target_data)

    source_camera_info = "{}/cameras.npz".format(source_camera_path)
    source_camera_info = np.load(source_camera_info)
    target_camera_info = "{}/cameras.npz".format(target_camera_path)
    target_camera_info = np.load(target_camera_info)

    source_camera_scale = source_camera_info["scale_mat_0"][0,0]
    target_camera_scale = target_camera_info["scale_mat_0"][0,0]

    init_scale = target_camera_scale / source_camera_scale
    # init_scale = 0.80 #  0.85 # 5

    voxel_size = 0.01

    source, target, source_down, target_down, trans_init, trans_score = apply_lepard(
        np.copy(source_data[:,:3]),
        np.copy(target_data[:,:3]),
        scale_=init_scale,
        voxel_size=voxel_size,
        do_sampling=False, view_result=True
    )
    print("  * [Lepard] Scale: {:.3f} | Score: {:.4f}".format(init_scale, trans_score))
    
    t2s_transform = np.copy(trans_init)
    t2s_transform[:3,:3] /= init_scale

    transformed_points = np.copy(source_data[:,:3])
    transformed_points =\
        np.concatenate((transformed_points, np.ones((len(transformed_points),1))), axis=1)
    transformed_points = np.dot(t2s_transform, np.transpose(transformed_points))
    source_data[:,:3] = np.transpose(transformed_points)[:,:3]

    transformed_normals = np.copy(source_data[:,3:6])
    transformed_normals =\
        np.concatenate((transformed_normals, np.ones((len(transformed_normals),1))), axis=1)
    transformed_normals = np.dot(t2s_transform, np.transpose(transformed_normals))
    source_data[:,3:6] = np.transpose(transformed_normals)[:,:3]

    merged_points = np.concatenate(
        (source_data, target_data), axis=0
    )    
    return merged_points, t2s_transform


def find_transformation(
        source_data, target_data, init_scale=0.22,
    ):
    voxel_size = 0.01
  
    source, target, source_down, target_down, trans_init, trans_score = apply_lepard(
        source_data[:,:3],
        target_data[:,:3],
        scale_=init_scale,
        voxel_size=voxel_size,
        do_sampling=False, view_result=False
    )
    print("  * Scale: {:.3f} | Score: {:.4f}".format(init_scale, trans_score))
    
    t2s_transform = np.copy(trans_init)
    t2s_transform[:3,:3] /= init_scale

    transformed_points = np.copy(source_data[:,:3])
    transformed_points =\
        np.concatenate((transformed_points, np.ones((len(transformed_points),1))), axis=1)
    transformed_points = np.dot(t2s_transform, np.transpose(transformed_points))
    source_data[:,:3] = np.transpose(transformed_points)[:,:3]
    return t2s_transform, source_data

def find_transformation2(
        source_data, target_data,
    ):
    # source_camera_info = "{}/cameras.npz".format(source_camera_path)
    # source_camera_info = np.load(source_camera_info)
    # target_camera_info = "{}/cameras.npz".format(target_camera_path)
    # target_camera_info = np.load(target_camera_info)

    # source_camera_scale = source_camera_info["scale_mat_0"][0,0]
    # target_camera_scale = target_camera_info["scale_mat_0"][0,0]

    # voxel_size = 0.03
    voxel_size = 0.01

    # init_scale = target_camera_scale / source_camera_scale
    # init_scale = source_camera_scale / target_camera_scale
    init_scale = 1.0

    max_scale, min_scale, scale_int = init_scale+0.20, init_scale-0.20, 0.05
    
    best_results = find_ransac_transformation(
        source_data[:,:3],
        target_data[:,:3],
        voxel_size=voxel_size,
        scale_list=[min_scale, max_scale, scale_int],
    )

    pcd_list, t2s_transform, t2s_score, t2s_scale = best_results
    source, target = pcd_list
    trans_init = t2s_transform

    t2s_transform[:3,:3] /= t2s_scale

    transformed_points = np.copy(source_data[:,:3]) # / t2s_scale
    transformed_points =\
        np.concatenate((transformed_points, np.ones((len(transformed_points),1))), axis=1)
    transformed_points = np.dot(t2s_transform, np.transpose(transformed_points))
    source_data[:,:3] = np.transpose(transformed_points)[:,:3]

    return t2s_transform, source_data
 