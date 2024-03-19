import cv2  # type: ignore

# from segment_anything import sam_model_registry

import numpy as np
import torch
import glob
import time
import os
import matplotlib
import matplotlib.pyplot as plt
# from typing import Any, Dict, List
from typing import Any, Dict, List, Tuple

import zipfile


from seg_decoder import SegHead, SegHeadUpConv
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F


from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        # if ann['pred_class'] == 1:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]
        img[m] = color_mask
    ax.imshow(img)


def show_anns_2(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        if ann['pred_class'] >= 0.9:
            m = ann['segmentation']
            img[m] = [0.25526778, 0.19120787, 0.67079563, 0.35]

    ax.imshow(img)


def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    x = x.to(torch.float32).to(pixel_mean.device)
    
    if x.shape[2] != 1024 or x.shape[3] != 1024: 
        x = F.interpolate(
            x,
            (1024, 1024),
            mode="bilinear",
        )
    
    x = (x - pixel_mean) / pixel_std
    return x

def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (64, 64),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks



def main():
    print("Loading model...")
    # sam_model = sam_model_registry['vit_h'](checkpoint='/raid/yehongliang_data/SAM_ckpts/sam_vit_h_4b8939.pth')
    # sam_model = sam_model_registry['vit_l'](checkpoint='/raid/yehongliang_data/SAM_ckpts/sam_vit_l_0b3195.pth')
    # sam_model = sam_model_registry['vit_h'](checkpoint='/home/zj/code/kyxz_sam_roadseg/SAM-checkpoint/sam_vit_h_4b8939.pth')

    # efficientsam = build_efficient_sam_vitt()
    with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
        zip_ref.extractall("weights")
    efficientsam = build_efficient_sam_vits()

    # device = torch.device("cuda:2")
    device = torch.device("cuda:0")

    efficientsam.to(device)

    seg_decoder = SegHead()
    ckpt_path = 'ckpts/orfd/best_epoch.pth'
    seg_decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    seg_decoder.eval()
    seg_decoder.to(device)

   
    image_file = 'test_image'

    img_list = sorted(glob.glob(os.path.join(image_file, '*.png')))

    save_path = 'test_output'
    os.makedirs(save_path, exist_ok=True)

    transform = ResizeLongestSide(1024)


    for t in img_list:
        print("Processing image:", t)
        img_name = os.path.split(t)[1]
        image = cv2.imread(t)
        print("Image shape:", image.shape)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = rgb_image.shape[:2]

        start_time = time.time()
        input_image = transform.apply_image(rgb_image)
        input_size = input_image.shape[:2]

        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # input_image_torch = efficientsam.preprocess(input_image_torch)
        input_image_torch = preprocess(input_image_torch)
        input_image_torch = input_image_torch.to(device)
        


        with torch.no_grad():
            image_embedding = efficientsam.image_encoder(input_image_torch)
            pred_mask = seg_decoder(image_embedding)
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size)

        fps = 1 / (time.time() - start_time)
        print(fps)

        pred = torch.softmax(pred_mask, dim=1).float()[0][1]
        pred = torch.where(pred > 0.98, 1, 0)
        pred = torch.as_tensor(pred, dtype=torch.uint8)

        pred = pred.cpu().detach().numpy()

        index = np.where(pred == 1)
        image[index[0], index[1], :] = [255, 0, 85]

        cv2.imwrite(os.path.join(save_path, img_name), image)
        print(img_name)



if __name__ == '__main__':
    main()