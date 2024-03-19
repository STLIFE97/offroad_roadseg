# ROD: RGB-Only Fast and Efficient Off-road Freespace Detection
## Introduction
This is the official PyTorch implementation of ROD: RGB-Only Fast and Efficient Off-road Freespace Detection

![vs](https://github.com/STLIFE97/offroad_roadseg/blob/main/picture/vs.png)


Off-road freespace detection is more challenging than on-road scenarios because of the blurred boundaries of traversable areas. Consequently, previous state-of-the-art (SOTA) methods employ multi-modal fusion of RGB images and LiDAR data. Due to the large increase in inference time when calculating surface normal maps from LiDAR data, multi-modal methods are not suitable for real-time applications. This paper presents a novel RGB-only approach for off-road freespace detection, named ROD, eliminating the reliance on LiDAR data and its computational demands. Utilizing a pre-trained Vision Transformer (ViT), our method extracts rich features from RGB images, improving both precision and inference speed. ROD achieves performance on the ORFD and RELLIS-3D Dataset, with 97.0% F1_score and 94.1% IoU on the ORFD dataset, and 95.1% F1_score and 97.6% Accuracy on the RELLIS-3D dataset. Additionally, ROD achieves an inference speed of 32 FPS, comfortably meeting real-time requirements.

![cost](https://github.com/STLIFE97/offroad_roadseg/blob/main/picture/cost.png)


## Requirements
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install the same environment as [SAM](https://github.com/facebookresearch/segment-anything).


## Datasets
The ORFD dataset we used can be found at [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection). Extract and organize as follows:
```
|-- datasets
 |  |-- ORFD
 |  |  |-- training
 |  |  |  |-- sequence   |-- calib
 |  |  |                 |-- sparse_depth
 |  |  |                 |-- dense_depth
 |  |  |                 |-- lidar_data
 |  |  |                 |-- image_data
 |  |  |                 |-- gt_image
 ......
 |  |  |-- validation
 ......
 |  |  |-- testing
 ......
```

## Usage
### Image_Demo
```
python demo.py
```


## Acknowledgement
Our code is inspired by [EfficientSAM](https://github.com/yformer/EfficientSAM), [ORFD](https://github.com/chaytonmin/Off-Road-Freespace-Detection)
