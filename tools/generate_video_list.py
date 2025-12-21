
import os
import re

movie_root = '/home/lyrico1202/AI_excersize/movie'
output_file = 'examples/extract_skeleton/all_actions.list'

categories = {
    '側転斜状': 1,
    '振り突き': 2,
    '前転突き': 3
}

video_extensions = ('.mp4', '.mov')

all_videos = []

def extract_number(filename):
    # Extract the number from the filename (e.g., "側転斜状_10_左.mp4" -> 10)
    match = re.search(r'_(\d+)(?:_|\.)', filename)
    if match:
        return int(match.group(1))
    return 0

for category, label in categories.items():
    dir_path = os.path.join(movie_root, category)
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found: {dir_path}")
        continue
        
    videos = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(video_extensions):
            full_path = os.path.join(dir_path, filename)
            videos.append(full_path)
    
    # Sort videos by number
    videos.sort(key=lambda x: extract_number(os.path.basename(x)))
    
    for video_path in videos:
        all_videos.append(f"{video_path} {label}")

with open(output_file, 'w') as f:
    for line in all_videos:
        f.write(line + '\n')

print(f"Generated {output_file} with {len(all_videos)} videos.")
