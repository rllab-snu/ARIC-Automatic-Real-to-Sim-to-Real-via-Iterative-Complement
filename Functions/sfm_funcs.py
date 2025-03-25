import os
import math
import shutil
import pycolmap
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from References.IDR.preprocess_cameras import get_normalization

def modify_datafolder(data_path, scene_name, img_list):
    observation_path = "{}/{}/Observations".format(data_path, scene_name)
    mask_path = "{}/{}/Masks".format(data_path, scene_name)

    os.rename(observation_path, observation_path+"_Before")
    os.rename(mask_path, mask_path+"_Before")

    os.makedirs(observation_path)
    os.makedirs(mask_path)

    shutil.copytree("{}_Before/merged_mask".format(mask_path), "{}/merged_mask".format(mask_path))

    object_list = os.listdir(mask_path+"_Before")
    object_list = [i_path for i_path in object_list if "mask_" in i_path and "png" not in i_path]
    num_detected_objects = len(object_list)
    for j in range(num_detected_objects):
        os.makedirs("{}/mask_{}".format(mask_path, j+1))

    for i, img_name in enumerate(img_list):
        img_type = img_name.split(".")[-1]
        img_idx = int(img_name.split("/")[-1].split(".")[0])

        curr_img = Image.open("{}_Before/{:06d}.{}".format(observation_path, img_idx, img_type))
        curr_img.save("{}/{:06d}.{}".format(observation_path, i, img_type))

        for j in range(num_detected_objects):
            curr_mask = Image.open("{}_Before/mask_{}/{:06d}.png".format(mask_path, j+1, img_idx))
            curr_mask.save("{}/mask_{}/{:06d}.png".format(mask_path, j+1, i))


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
    
def form_T(R, p):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = np.array(p).reshape((3, 1))
    return T

def make_extrinsic_matrix(num_images, image_info, intrinsic_mat):
    img_idx_list = []
    for i in range(num_images):
        line_num = 4 + 2*i
        camera_info = image_info[line_num].split(" ")
        img_idx = int(camera_info[-1].split(".")[0])
        img_idx_list.append(img_idx)
    img_idx_list = sorted(img_idx_list)

    cameras = {}
    for i in range(num_images):
        line_num = 4 + 2*i
        camera_info = image_info[line_num].split(" ")
        
        quat_ = [float(camera_info[2]), float(camera_info[3]), float(camera_info[4]), float(camera_info[1])]
        tran_ = [float(camera_info[5]), float(camera_info[6]), float(camera_info[7])]
                
        r = R.from_quat(quat_)
        rot_mat = np.array(r.as_matrix())
        tran_mat = np.array(tran_).reshape((-1,1))
        rot_tran_mat = np.concatenate((rot_mat,tran_mat), axis=1)
        
        world_mat = np.matmul(intrinsic_mat, rot_tran_mat)
        world_mat = np.concatenate((
            world_mat,
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ), axis=0)
        
        img_idx = int(camera_info[-1].split(".")[0])
        img_idx = img_idx_list.index(img_idx)
        cameras["world_mat_{}".format(img_idx)] = world_mat
    return cameras

def make_intrinsic_matrix(cameras_info):
    # make intrinsic matrix
    intrinsic_info = cameras_info[3].split(" ")
    intrinsic_mat = np.identity(3)
    intrinsic_mat[0,0] = float(intrinsic_info[4])
    intrinsic_mat[1,1] = float(intrinsic_info[4])
    intrinsic_mat[0,2] = float(intrinsic_info[5])
    intrinsic_mat[1,2] = float(intrinsic_info[6])
    return intrinsic_mat

def make_camera_matrix(
        scene_name,
        image_list,
        data_path="./Dataset",
    ):
    num_images = len(image_list)

    camera_path = "{}/{}/Cameras".format(data_path, scene_name)
    image_info_path = "{}/images.txt".format(camera_path)
    cameras_info_path = "{}/cameras.txt".format(camera_path)

    cameras_file = open(cameras_info_path, "r")
    cameras_info = cameras_file.readlines()
    intrinsic_mat = make_intrinsic_matrix(cameras_info)

    image_file = open(image_info_path, "r")
    image_info = image_file.readlines()
    camera_mat = make_extrinsic_matrix(num_images, image_info, intrinsic_mat)
    np.savez("{}/cameras_colmap.npz".format(camera_path), **camera_mat)

    get_normalization(data_path, scene_name, camera_type="colmap")

    return camera_mat
    

def perform_sfm_using_colmap(
        scene_name,
        data_path="./Dataset",
    ):
    observation_path = "{}/{}/Observations".format(data_path, scene_name)
    camera_path = "{}/{}/Cameras".format(data_path, scene_name)
    if not os.path.exists(camera_path): os.makedirs(camera_path)
    database_path = camera_path + "/database.db"
    
    pycolmap.extract_features(
        database_path,
        observation_path,
        camera_mode = pycolmap.CameraMode.SINGLE,
        camera_model = 'SIMPLE_PINHOLE',
        # camera_model = 'SIMPLE_RADIAL',
        # camera_model = 'OPENCV',
    )
    pycolmap.match_exhaustive(database_path)
    # pycolmap.match_sequential(database_path)
    # pycolmap.match_spatial(database_path)
    maps = pycolmap.incremental_mapping(database_path, observation_path, camera_path)
    maps[0].write(camera_path)

    reconstruction = pycolmap.Reconstruction(camera_path)
    # print(reconstruction.summary())
    reconstruction.write_text(camera_path)

    image_name_list = []
    for image_id, image in reconstruction.images.items():
        image_name_list.append(image.name)
    image_name_list = sorted(image_name_list)

    if reconstruction.num_images() < 10:
        print("  [WARNING] There are so few images ({}) detected by COLMAP.".format(reconstruction.num_images()))
    
    img_list = os.listdir(observation_path)
    img_list = [i_path for i_path in img_list if ".jpg" in i_path]    
    # assert len(image_name_list) == len(img_list)  
    assert len(image_name_list) >= 20
    if len(image_name_list) != len(img_list):
        print("  [COLMAP] Reduce Observations: {} => {}".format(len(img_list), len(image_name_list)))
        modify_datafolder(data_path, scene_name, image_name_list)
    return image_name_list


def perform_sfm_using_robotpose(
        scene_name,
        data_path="./Dataset",
    ):
    info_path = "{}/{}/Raws".format(data_path, scene_name)
    observation_path = "{}/{}/Observations".format(data_path, scene_name)
    camera_path = "{}/{}/Cameras".format(data_path, scene_name)
    if not os.path.exists(camera_path): os.makedirs(camera_path)

    img_list = os.listdir(observation_path)
    img_list = [i_path for i_path in img_list if ".jpg" in i_path]
    n_images = len(img_list)
    if "-1.jpg" in img_list: n_images -= 1

    camera_mat = {}
    for i in range(n_images):
        data_ = np.load("{}/{}_obs{:03d}.npz".format(info_path, scene_name, i))
        
        intrinsic_mat = np.array(data_["camera_intrinsic"])
        eef_pose = np.array(data_["robot_pose"])
        eef_quat = np.array(data_["robot_quat"])
        
        T_robot_to_eef = form_T(quat2mat(eef_quat), eef_pose)
        # T_eef_to_rs = np.array(data_["eef_to_rs_matrix"])
        T_eef_to_rs = np.load("./RealRobot/rs_extrinsic.npy")
        M_robot_to_rs = np.dot(T_robot_to_eef, T_eef_to_rs)
        
        Inv_robot_to_rs = np.linalg.inv(M_robot_to_rs)
            
        world_mat = np.matmul(intrinsic_mat, Inv_robot_to_rs[0:3,:])
        world_mat = np.concatenate((
            world_mat,
            np.array([[0.0, 0.0, 0.0, 1.0]])
        ), axis=0)            
        camera_mat["world_mat_{}".format(i)] = world_mat

    np.savez("{}/cameras_robot.npz".format(camera_path), **camera_mat)

    get_normalization(data_path, scene_name, camera_type="robot")


 
def main():
    dataset_path = "./Dataset"
    scene_name = "object1"
    object_index = 1

    image_paths = perform_sfm_using_colmap(scene_name, dataset_path)
    make_camera_matrix(scene_name, object_index, image_paths, dataset_path)


if __name__ == "__main__":
    main()