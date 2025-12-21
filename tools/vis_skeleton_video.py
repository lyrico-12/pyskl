
import argparse
import mmcv
import os
import os.path as osp
from pyskl.utils.visualize import Vis2DPose
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize skeleton results')
    parser.add_argument('pkl_file', help='pickle file containing skeleton annotations')
    parser.add_argument('video_list', help='video list file used for extraction')
    parser.add_argument('out_dir', help='output directory for visualized videos')
    parser.add_argument('--fps', type=float, default=30, help='video fps')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load annotations
    annotations = mmcv.load(args.pkl_file)
    
    # Load video list to map frame_dir to video path
    video_paths = {}
    with open(args.video_list, 'r') as f:
        for line in f:
            path = line.strip().split()[0]
            frame_dir = osp.basename(path).split('.')[0]
            video_paths[frame_dir] = path
            
    os.makedirs(args.out_dir, exist_ok=True)
    
    label_map = {
        1: '側転斜状',
        2: '振り突き',
        3: '前転突き'
    }
    
    for i, anno in enumerate(tqdm(annotations)):
        frame_dir = anno['frame_dir']
        label = anno.get('label', 0)
        action_name = label_map.get(label, 'unknown')
        
        if frame_dir in video_paths:
            video_path = video_paths[frame_dir]
            # print(f"Visualizing {frame_dir} ({action_name}) using video {video_path}")
            
            try:
                vis_video = Vis2DPose(anno, thre=0.2, out_shape=None, layout='coco', fps=args.fps, video=video_path)
                
                action_dir = osp.join(args.out_dir, action_name)
                os.makedirs(action_dir, exist_ok=True)
                
                out_path = osp.join(action_dir, frame_dir + '_vis.mp4')
                vis_video.write_videofile(out_path, logger=None) # Disable moviepy logger to reduce noise
            except Exception as e:
                print(f"Failed to visualize {frame_dir}: {e}")
        else:
            print(f"Warning: Video path for {frame_dir} not found in video list.")

if __name__ == '__main__':
    main()
