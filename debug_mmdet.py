import mmcv
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model
import torch
import decord
import numpy as np
import pyskl

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MMCV: {mmcv.__version__}")
print(f"PYSKL imported: {pyskl.__file__}")

# Initialize object detection model
config = 'demo/faster_rcnn_r50_fpn_1x_coco-person.py'
checkpoint = '/home/lyrico1202/.cache/torch/hub/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

print("Initializing detector...")
try:
    model = init_detector(config, checkpoint, device='cuda:0')
    print("Detector initialized successfully")
except Exception as e:
    print(f"Error: {e}")

# Initialize pose estimation model
pose_config = 'demo/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'

print("Initializing pose model...")
try:
    pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda:0')
    print("Pose model initialized successfully")
except Exception as e:
    print(f"Error initializing pose model: {e}")

video_path = '/home/lyrico1202/AI_excersize/movie/側転斜状/側転斜状_10_左.mp4'
print(f"Reading video: {video_path}")

try:
    vid = decord.VideoReader(video_path)
    print(f"Video read successfully. Frames: {len(vid)}")
    frames = [x.asnumpy() for x in vid]
    print(f"Frames extracted. Shape: {frames[0].shape}")
except Exception as e:
    print(f"Error reading video: {e}")

from mmdet.apis import inference_detector

print("Running detection inference on first frame...")
try:
    result = inference_detector(model, frames[0])
    print("Detection inference successful")
except Exception as e:
    print(f"Error during detection inference: {e}")
