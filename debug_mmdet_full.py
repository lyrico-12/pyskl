
import argparse
import copy as cp
import mmcv
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm
import decord

import pyskl
from pyskl.smp import mrlines

import mmdet
from mmdet.apis import inference_detector, init_detector

import mmpose
from mmpose.apis import inference_top_down_pose_model, init_pose_model

print("Imports successful")

det_config = 'demo/faster_rcnn_r50_fpn_1x_coco-person.py'
det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

print("Initializing detector...")
det_model = init_detector(det_config, det_ckpt, 'cuda')
print("Detector initialized")

pose_config = 'demo/hrnet_w32_coco_256x192.py'
pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

print("Initializing pose model...")
pose_model = init_pose_model(pose_config, pose_ckpt, 'cuda')
print("Pose model initialized")

video_path = '/home/lyrico1202/AI_excersize/movie/側転斜状/側転斜状_10_左.mp4'
print(f"Reading video {video_path}")
vid = decord.VideoReader(video_path)
frames = [x.asnumpy() for x in vid]
print(f"Frames read: {len(frames)}")

print("Running inference")
result = inference_detector(det_model, frames[0])
print("Inference done")
