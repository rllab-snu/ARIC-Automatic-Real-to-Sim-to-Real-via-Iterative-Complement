import os
import math
import torch
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("./References/FastSAM")
from Functions.fastsam_funcs import get_masks_with_FastSam

def save_img(obs_img, img_name, img_path):
    rgb_ = Image.fromarray(obs_img)
    if not os.path.exists(img_path): os.makedirs(img_path)
    rgb_.save("{}/{}.png".format(img_path, img_name))
    
def load_img(obs_img, img_name, img_path, do_padding=False):
    save_img(obs_img, img_name, img_path)
    obs_ = np.array(obs_img)
    
    mask_ = get_merged_mask(img_name, img_path)
    obs_ = obs_ * mask_
    
    if np.shape(obs_)[0] != np.shape(obs_)[1]:
        # obs_size = 480x640
        obs_ = np.array(obs_)[:,80:560,:]
    if do_padding:
        obs_ = padding_img(obs_)
    obs_ = Image.fromarray(obs_).resize((128,128))
    obs_ = np.array(obs_)
    return obs_

def convert_img(obs_numpy):
    obs_torch = torch.from_numpy(obs_numpy)
    obs_torch = obs_torch.unsqueeze(0).permute(0,3,1,2).to("cuda:0")
    return obs_torch

def padding_img(obs_numpy, img_size=680):
    obs_size = np.shape(obs_numpy)[0]
    obs_zeros = np.zeros((img_size,img_size,3), dtype=np.uint8)
    pixel_min = int(img_size/2) - int(obs_size/2)
    obs_zeros[pixel_min:pixel_min+obs_size,pixel_min:pixel_min+obs_size,:] = obs_numpy[:,:,:]
    return obs_zeros
    
def get_merged_mask(obs_name, obs_path):
    img_path = "{}/{}.png".format(obs_path, obs_name)
    mask_path = "{}/{}".format(obs_path, obs_name)

    masks_ = get_masks_with_FastSam(img_path, mask_path)

    masks_np = masks_.detach().cpu().numpy()
    merged_mask = np.zeros_like(masks_np[0], dtype=bool)
    for i in range(len(masks_np)):
        curr_mask = np.array(masks_np[i], dtype=bool)
        merged_mask = np.logical_or(curr_mask, merged_mask)
    merged_mask = np.array(merged_mask, dtype=np.uint8)
    merged_mask = np.expand_dims(merged_mask, -1)
    return merged_mask
    
def draw_push_action(input_image, action_):
    input_image = np.copy(input_image)
    
    act_x = -action_[0]
    act_y = -action_[1]
    act_th = action_[2] + 1.0

    pixel_margin1 = 3
    pixel_margin2 = 2
    distance_ = 0.15

    pixel_s = int(128*4.7/14.0)
    pixel_e = int(128*9.3/14.0)

    pixel_x = int(64+(pixel_e-pixel_s)/2*act_x)
    pixel_y = int(64+(pixel_e-pixel_s)/2*act_y)

    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,0] = 255
    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,1] = 255
    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,2] = 255
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,0] = 255
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,1] = 0
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,2] = 0

    pixel_x = int(64+(pixel_e-pixel_s)/2*act_x) + int((pixel_e-pixel_s)/2*(distance_*10.0*math.cos(act_th*math.pi)))
    pixel_y = int(64+(pixel_e-pixel_s)/2*act_y) + int((pixel_e-pixel_s)/2*(distance_*10.0*math.sin(act_th*math.pi)))

    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,0] = 255
    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,1] = 255
    input_image[pixel_x-pixel_margin1:pixel_x+pixel_margin1,pixel_y-pixel_margin1:pixel_y+pixel_margin1,2] = 255
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,0] = 0
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,1] = 255
    input_image[pixel_x-pixel_margin2:pixel_x+pixel_margin2,pixel_y-pixel_margin2:pixel_y+pixel_margin2,2] = 0
    return input_image


