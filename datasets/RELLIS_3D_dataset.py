import os.path
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils import data
import cv2
import numpy as np
import glob
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F

class RELLIS3DDataset(data.Dataset):
    """dataloader for ORFD dataset"""
    def __init__(self, root, mode='train', target_size=1024):
        self.root = root  # path for the dataset
        self.target_size = target_size
        self.mode = mode
        self.img_size = (1200, 1920)
        self.transform = ResizeLongestSide(target_size)

        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]

        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        if self.mode == "train":
            self.image_list = glob.glob(os.path.join(self.root, 'training', '*', 'pylon_camera_node', '*.jpg'))
        elif self.mode == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', '*', 'pylon_camera_node', '*.jpg')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', '*', 'pylon_camera_node', '*.jpg')))

    def preprocess(self, x, is_img=True):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if is_img:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.target_size - h
        padw = self.target_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def map_colors(self, rgb_color):
        
        passable_colors = {
            (108,64,20),    #dirt
            (0,102,0),      #grass
            (64,64,64),     #asphalt
            (170,170,170),  #concrete
            (99,66,34),     #mud
        }
        tolerance = 2

        # 进行颜色近似匹配
        mapped_label = np.zeros_like(rgb_color)
        for target_color in passable_colors:
            condition = np.all(np.abs(rgb_color - target_color) <= tolerance, axis=-1)
            mapped_label[condition] = 255

        return mapped_label


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.imread(os.path.join(useDir, 'pylon_camera_node', name))
        oriHeight, oriWidth, _ = rgb_image.shape
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        label_img_name = name.split('.')[0] + ".png"
        label_dir = os.path.join(useDir, 'pylon_camera_node_label_color', label_img_name)
        label = cv2.imread(label_dir)

        # 对label进行颜色映射
        label = self.map_colors(label)

        input_image = self.transform.apply_image(rgb_image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = self.preprocess(input_image_torch, is_img=True)
        input_image_torch = input_image_torch.squeeze(0)

        label = self.transform.apply_image(label)
        label_torch = torch.as_tensor(label)
        label_torch = label_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        label_torch = self.preprocess(label_torch, is_img=False).float()
        label_torch = F.interpolate(label_torch, size=(256, 256), mode='bilinear', align_corners=False)
        # torch.set_printoptions(profile="full")
        # print(label_torch)

        label_torch = label_torch.squeeze(0)[0] / 255
        label_torch = torch.where(label_torch > 0.9, 1, 0)
        # print('rgb_image', input_image_torch, 'label', label_torch, 'oriSize',(oriWidth, oriHeight))
        return {'rgb_image': input_image_torch, 'label': label_torch, 'oriSize': (oriWidth, oriHeight)}

if __name__ == '__main__':
    d = RELLIS3DDataset('/home/zj/code/Rellis_3D/', mode='train', target_size=1024)
    for i, data in enumerate(d):
        image = data['rgb_image']
        image = image.permute(1, 2, 0).cpu().numpy()
        image = image * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite('image.jpg', image)

        label = data['label']
        label = label.cpu().numpy()
        cv2.imwrite('label.png', label * 255)
        pass









