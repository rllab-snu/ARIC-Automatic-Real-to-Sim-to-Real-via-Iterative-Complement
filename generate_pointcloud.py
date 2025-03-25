import os
import open3d as o3d
import torch
import numpy as np
from PIL import Image

from Functions.sam_funcs import inferenceFastSam, inferenceSam2
from Functions.sfm_funcs import perform_sfm_using_colmap, make_camera_matrix, perform_sfm_using_robotpose
from Functions.surfels_funcs import optimize_gaussian_surfels, render_gaussian_surfels, generate_mesh, convert_mesh
from Functions.merge_funcs import merge_two_pointclouds
from Functions.score_funcs import get_pcd_score
from Functions.point_utils import remove_unvalid_points


def scene_processing(
        scene,
        scene_num,
        sfm_type="colmap",
        dataset_path="./Dataset",
    ):
    scene_name = "{}{}".format(scene, scene_num)
    scene_path = "{}/{}".format(dataset_path, scene_name)

    rawobs_path = '{}/Raws/'.format(scene_path)
    origin_image_path = '{}/Origin_Observations/'.format(scene_path)
    image_path = '{}/Observations/'.format(scene_path)
    output_path = '{}/Masks'.format(scene_path)
    if not os.path.exists(output_path): os.makedirs(output_path)

    init_frame = 0

    # Raw Observations => RGB Images ##############
    if os.path.exists(rawobs_path) and not os.path.exists(image_path):
        os.makedirs(image_path)

        img_list = os.listdir(rawobs_path)
        img_list = [i_path for i_path in img_list if ".npz" in i_path]
        n_images = len(img_list)

        for i in range(n_images):
            data_ = np.load("{}/{}_obs{:03d}.npz".format(rawobs_path, scene_name, i))
            rgb_ = data_["rgb_image"]
            img = Image.fromarray(rgb_)
            img.save("{}/{:06d}.jpg".format(image_path, i))

    elif os.path.exists(origin_image_path) and not os.path.exists(image_path):
        os.makedirs(image_path)

        img_list = os.listdir(origin_image_path)
        img_list = [i_path for i_path in img_list if ".jpg" in i_path or ".png" in i_path]
        img_list = sorted(img_list)
        n_images = len(img_list)

        for i, img_name in enumerate(img_list):
            img = Image.open("{}{}".format(origin_image_path, img_name))
            img = img.resize((640,640))
            img.save("{}/{:06d}.jpg".format(image_path, i))

    # RGB Images => Object Masks ##################
    object1_path = '{}/mask_1/'.format(output_path)
    if os.path.exists(object1_path):     
        object_list = os.listdir(output_path)
        object_list = [i_path for i_path in object_list if "mask_" in i_path and "png" not in i_path]
        num_detected_objects = len(object_list)
    else:
        if scene_num == 1:
            first_segm_info = None
            init_img = '{}{:06d}.jpg'.format(image_path, init_frame)
            obj_mask_coords = inferenceFastSam(init_img, output_path)
            np.savez(
                "{}/segmentation_info.npz".format(output_path),
                init_img_path='{}/{:06d}.jpg'.format(image_path, init_frame),
                init_mask_coords=obj_mask_coords,
            )
        else:
            first_scene_path = "{}/{}1/Masks/segmentation_info.npz".format(dataset_path, scene)
            first_segm_info = np.load(first_scene_path, allow_pickle=True)
            obj_mask_coords = first_segm_info["init_mask_coords"].item()

        inferenceSam2(image_path, output_path, obj_mask_coords, first_segm_info)
        num_detected_objects = len(obj_mask_coords)

    # RGBs + Masks => Camera Matrice ###############
    camera_path = '{}/cameras.npz'.format(scene_path)
    if not os.path.exists(camera_path):
        if sfm_type == "colmap":
            image_paths = perform_sfm_using_colmap(scene_name, dataset_path)
            make_camera_matrix(scene_name, image_paths, dataset_path)
        elif sfm_type == "robot":
            perform_sfm_using_robotpose(
                scene_name, dataset_path
            )
    return num_detected_objects


def generate_pointcloud(
        scene,
        scene_num,
        num_detected_objects,
        pointcloud_name = "origin_points",
        dataset_path="./Dataset",
    ):
    scene_name = "{}{}".format(scene, scene_num)
    scene_path = "{}/{}".format(dataset_path, scene_name)
    output_path = '{}/Masks'.format(scene_path)

    generated_results = {}

    # RGBs + Masks + Cameras => Object Point-Clouds #
    for object_index in range(num_detected_objects):
        source_path = "{}/{}".format(dataset_path, scene_name)
        output_path = "{}/{}/object_{}".format(dataset_path, scene_name, object_index+1)

        # try:
        if not os.path.exists(output_path):
            optimize_gaussian_surfels(source_path, output_path, object_index+1)
            sampled_points = render_gaussian_surfels(output_path)
            sampled_points = sampled_points.detach().cpu().numpy()

            sampled_points = remove_unvalid_points(
                sampled_points,
                camera_path="{}/cameras.json".format(output_path),
            )

            np.save("{}/{}.npy".format(output_path, pointcloud_name), sampled_points)
        else:
            points_path = "{}/{}.npy".format(output_path, pointcloud_name)
            sampled_points = np.load(points_path)
        
        points_mean = np.mean(sampled_points[:,:3], axis=0, keepdims=True)
        sampled_points[:,0:3] = sampled_points[:,0:3] - points_mean
        sampled_points[:,3:6] = sampled_points[:,3:6] - points_mean

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(sampled_points[:,:3])
        o3d.io.write_point_cloud("{}/{}.ply".format(output_path, pointcloud_name), pcd_tmp)

        object_score = get_pcd_score(
            sampled_points,
            camera_path="{}/cameras.json".format(output_path),
            pcd_path=output_path,
            consider_cameras=True,
        )

        object_info = {
            "pcd": sampled_points,
            "score": object_score,
        }
        # except:
        #     print()
        #     print("[ERROR] Fail to Generate Object{}'s Point-Cloud.".format(object_index+1))
            
        #     object_info = {
        #         "pcd": np.zeros((1,9)),
        #         "score": 0.0,
        #     }
        
        generated_results["object{}".format(object_index+1)] = object_info

    print()
    print("Generating Results: {}".format(scene_name))
    for object_index in range(num_detected_objects):
        object_score = generated_results["object{}".format(object_index+1)]["score"]
        print("* Object{}-Score: {:.3f}".format(object_index+1, object_score))
    return generated_results



def main():
    scene = "test_scene"
    sfm_type = "colmap"           # {robot, colmap}
    merge_type = "lepard"
    dataset_path = "./Dataset"
    scene_num = 2

    num_objects = scene_processing( 
        scene, scene_num,
        sfm_type=sfm_type,
        dataset_path=dataset_path,
    )

    source_pointclouds = generate_pointcloud( 
        scene, scene_num, num_objects,
        pointcloud_name="origin_points",
        dataset_path=dataset_path,
    )

    if scene_num >= 2: # merge scenes
        if scene_num == 2:
            target_name = "origin_points"
        else:
            target_name = "merged_points_step{}".format(scene_num-2)
        
        target_pointclouds = generate_pointcloud( 
            scene, 1, num_objects,
            pointcloud_name=target_name,
            dataset_path=dataset_path,
        )

        for object_idx in range(num_objects):
            merged_points, _ = merge_two_pointclouds(
                source_data=source_pointclouds["object{}".format(object_idx+1)]["pcd"],
                target_data=target_pointclouds["object{}".format(object_idx+1)]["pcd"],
                source_camera_path="{}/{}{}".format(dataset_path, scene, scene_num),
                target_camera_path="{}/{}1".format(dataset_path, scene),
                merge_method=merge_type,
            )

            old_object_score = get_pcd_score(
                target_pointclouds["object{}".format(object_idx+1)]["pcd"],
                camera_path="{}/{}1/object_{}/cameras.json".format(dataset_path, scene, object_idx+1),
                pcd_path="{}/{}1/object_{}".format(dataset_path, scene, object_idx+1),
                consider_cameras=True,
                save_pcd=False,
            )

            new_object_score = get_pcd_score(
                merged_points,
                camera_path="{}/{}{}/object_{}/cameras.json".format(dataset_path, scene, scene_num, object_idx+1),
                pcd_path="{}/{}{}/object_{}".format(dataset_path, scene, scene_num, object_idx+1),
                consider_cameras=True,
            )
            
            print("* Merged-Object{}-Score: {:.3f} -> {:.3f}".format(object_idx+1, old_object_score, new_object_score))

            merging_name = "merged_points_step{}".format(scene_num-1)
            save_path = "{}/{}1/object_{}".format(dataset_path, scene, object_idx+1)
            np.save("{}/{}.npy".format(save_path, merging_name), merged_points)

            merged_torch = torch.from_numpy(merged_points).to("cuda:0")
            generate_mesh(merged_torch, save_path, merging_name)
            convert_mesh(save_path, merging_name)

            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(merged_points[:,:3])
            o3d.io.write_point_cloud("{}/{}.ply".format(save_path, merging_name), pcd_tmp)

    else:
        for object_idx in range(num_objects):
            curr_data = source_pointclouds["object{}".format(object_idx+1)]["pcd"]
            save_path = "{}/{}1/object_{}".format(dataset_path, scene, object_idx+1)
            target_name = "origin_points"
            curr_torch = torch.from_numpy(curr_data).to("cuda:0")
            generate_mesh(curr_torch, save_path, target_name)
            convert_mesh(save_path, target_name)

if __name__ == "__main__":
    main()