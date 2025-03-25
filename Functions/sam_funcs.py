import os
import sys
sys.path.append('./References/FastSAM')
sys.path.append('./References/SAM2')

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
from sam2.build_sam import build_sam2_video_predictor

device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
)

##### FASTSAM
fastsam_model_path = './References/FastSAM/pretrained/FastSAM-x.pt'

##### SAM2
sam2_checkpoint = "./References/SAM2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


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

    selected_masks = filterMasksPart1(ann, height, width, area_threshold=0.003)
    
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


def inferenceSam2(video_path, output_path, obj_mask_coords, first_segm_info):
    frame_names = [
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    if first_segm_info is not None:
        new_init_path = "{}-1.jpg".format(video_path)
        if "-1.jpg" not in frame_names:
            init_img_path = first_segm_info["init_img_path"].item()
            new_init_img = Image.open(init_img_path)
            new_init_img.save(new_init_path)
            frame_names.insert(0, "-1.jpg")

    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    for idx, obj in enumerate(obj_mask_coords.keys()):
        coords = obj_mask_coords[obj]
        ann_obj_id = idx+1  # give a unique id to each object we interact with (it can be any integers)

        points = np.array(coords, dtype=np.float32)
        labels = np.array([1,1,1], np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # overlay image of all masks
    os.makedirs(f"{output_path}/result", exist_ok=True)
    plt.close("all")
    result_path = os.path.join(output_path, "result")
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc="all masks"):
        img_idx = int(frame_names[out_frame_idx].split(".")[-2])
        if img_idx < 0: continue
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_path, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(f"{result_path}/{img_idx:06}.png", dpi=300, bbox_inches='tight')

    # mask per objects
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc="masks per obj"):
        img_idx = int(frame_names[out_frame_idx].split(".")[-2])
        if img_idx < 0: continue
        total_mask = None
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            obj_path = os.path.join(output_path, f"mask_{out_obj_id}")
            os.makedirs(obj_path, exist_ok=True)

            out_mask = np.squeeze(np.array(out_mask, dtype=np.uint8)*255)
            cv2.imwrite(f'{obj_path}/{img_idx:06}.png', out_mask)

            if total_mask is None:
                total_mask = np.copy(out_mask)
            else:
                total_mask = np.where(out_mask==255, 255, total_mask)
        total_mask_path = os.path.join(output_path, f"merged_mask")
        os.makedirs(total_mask_path, exist_ok=True)
        cv2.imwrite(f'{total_mask_path}/{img_idx:06}.png', total_mask)

    if first_segm_info is not None and os.path.exists(new_init_path):
        os.remove(new_init_path)

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

    selected_masks = filterMasksPart1(ann, height, width, area_threshold=0.004)
    _, selected_masks = filterMasksPart2(selected_masks)
    return selected_masks
    
    
def main():
    data_path = "./Dataset"
    scene_name = "object1"
    init_frame = 0

    image_path = '{}/{}/Observations/'.format(data_path, scene_name)
    init_img = '{}/{}/Observations/{:06d}.jpg'.format(data_path, scene_name, init_frame)
    output_path = '{}/{}/Masks'.format(data_path, scene_name)
    os.makedirs(output_path, exist_ok=True)
        
    obj_mask_coords = inferenceFastSam(init_img, output_path)
    print(f"generated initial mask for {len(obj_mask_coords)} objects")

    inferenceSam2(image_path, output_path, obj_mask_coords)


if __name__ == "__main__":
    main()
