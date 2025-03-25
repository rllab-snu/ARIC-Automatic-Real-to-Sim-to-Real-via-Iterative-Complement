import os
import sys
sys.path.append('./References/FastSAM')
import ast
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
from tqdm import tqdm

from fastsam import FastSAM, FastSAMPrompt 

device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
)

##### FASTSAM
fastsam_model_path = './References/FastSAM/pretrained/FastSAM-x.pt'


def filterMasksPart1(ann, height, width, area_threshold=0.002):
    initial_frame_idx = 0
    while True:
        initial_mask = ann[initial_frame_idx]
        if torch.sum(initial_mask) < height*width*0.5: break
        else: initial_frame_idx += 1
    selected_masks = initial_mask.unsqueeze(0).clone()

    for i in range(0, len(ann)):
        if i == initial_frame_idx: continue

        target_mask = ann[i]

        if torch.sum(target_mask) > height*width*0.5 or torch.sum(target_mask) < height*width*area_threshold: continue
        add = True

        for j in range(len(selected_masks)):
            saved_mask = selected_masks[j]
            comp_result = (target_mask * saved_mask).sum()

            if comp_result > torch.sum(saved_mask)*0.7 or comp_result > torch.sum(target_mask)*0.7:
                if torch.sum(target_mask) > torch.sum(saved_mask):
                    selected_masks[j] = target_mask
                add = False        
                break
            
        if add:
            selected_masks = torch.cat((selected_masks, target_mask.unsqueeze(0)), dim=0)

    return selected_masks


def filterMasksPart2(selected_masks):
    for i in range(len(selected_masks)-1):
        for j in range(i+1, len(selected_masks)):
            mask1 = selected_masks[i]
            mask2 = selected_masks[j]
            comp_result = (mask1 * mask2).sum()

            if comp_result > torch.sum(mask1)*0.7 or comp_result > torch.sum(mask2)*0.7:
                if torch.sum(mask1) > torch.sum(mask2):
                    selected_masks = torch.cat((selected_masks[:j], selected_masks[j+1:]), dim=0)
                else:
                    selected_masks = torch.cat((selected_masks[:i], selected_masks[i+1:]), dim=0)
                return True, selected_masks
    return False, selected_masks


def inferenceFastSam(img_path, output_path):
    model = FastSAM(fastsam_model_path)
    input = Image.open(img_path)
    input = input.convert("RGB")
    width, height = input.size

    everything_results = model(
        input,
        device=device,
        retina_masks=True, #high-resolution segmentation mask
        imgsz=1024, #image size
        conf=0.6, #confidence score
        iou=0.9 #iou threshold
        )
    
    prompt_process = FastSAMPrompt(input, everything_results, device)
    ann = prompt_process.everything_prompt()

    #for debug
    os.makedirs(f"{output_path}/original", exist_ok=True)
    for i, mask in enumerate(ann):
        mask = mask.cpu().numpy().astype(np.uint8)
        mask *= 255
        cv2.imwrite(f'{output_path}/original/mask_{i}.png', mask)

    prompt_process.plot(
        annotations=ann,
        output_path=f"{output_path}/"+img_path.split("/")[-1],
        bboxes = None,
        points = None,
        point_label = None,
        withContours = False,
        better_quality = False,
    )

    selected_masks = filterMasksPart1(ann, height, width)
    
    #for debug
    os.makedirs(f"{output_path}/filter1", exist_ok=True)
    for i, mask in enumerate(selected_masks):
        mask = mask.cpu().numpy().astype(np.uint8)
        mask *= 255
        cv2.imwrite(f'{output_path}/filter1/mask_{i+1}.png', mask)

    change = True
    while change:
        change, selected_masks = filterMasksPart2(selected_masks)


    # delete those regarded as background(on the edge of the image) and save valid masks
    obj_mask_coords = {}
    count = 1
    for mask in selected_masks:
        thresh = 5

        mask = mask.cpu().numpy().astype(np.uint8)
        mask_indices = np.argwhere(mask == 1)

        y1, x1 = mask_indices.min(axis=0)
        y2, x2 = mask_indices.max(axis=0)

        if (y1 < thresh or y2 > height-thresh or x1 < thresh or x2 > width-thresh):
            continue
        
        coords_candidate = []
        for x_coord in range(x1, x2, 2):
            for y_coord in range(y1, y2, 2):
                if mask[y_coord][x_coord]: coords_candidate.append([x_coord, y_coord])
        coords = [coords_candidate[random.randint(0, len(coords_candidate)-1)] for i in range(3)]

        if len(coords):
            obj_mask_coords[f"obj_{count}"] = coords

        mask *= 255
        for coord in coords:
            cv2.circle(mask, coord, 3, 0, -1)
        cv2.imwrite(f'{output_path}/mask_{count}.png', mask)

        count += 1

    return obj_mask_coords


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def get_masks_with_FastSam(img_path, output_path):
    model = FastSAM(fastsam_model_path)
    input = Image.open(img_path)
    input = input.convert("RGB")
    width, height = input.size

    everything_results = model(
        input,
        device=device,
        retina_masks=True, #high-resolution segmentation mask
        imgsz=1024, #image size
        conf=0.6, #confidence score
        iou=0.9 #iou threshold
        )
    
    prompt_process = FastSAMPrompt(input, everything_results, device)
    ann = prompt_process.everything_prompt()

    #for debug
    os.makedirs(f"{output_path}/original", exist_ok=True)
    for i, mask in enumerate(ann):
        mask = mask.cpu().numpy().astype(np.uint8)
        mask *= 255
        cv2.imwrite(f'{output_path}/original/mask_{i}.png', mask)

    prompt_process.plot(
        annotations=ann,
        output_path=f"{output_path}/"+img_path.split("/")[-1],
        bboxes = None,
        points = None,
        point_label = None,
        withContours = False,
        better_quality = False,
    )

    selected_masks = filterMasksPart1(ann, height, width, area_threshold=0.005)
    _, selected_masks = filterMasksPart2(selected_masks)
    return selected_masks
    
