
import mmcv
import os.path as osp

# Load existing annotations
original_pkl = 'examples/extract_skeleton/all_actions_annos.pkl'
new_pkl = 'examples/extract_skeleton/all_actions_annos_add.pkl'
merged_pkl = 'examples/extract_skeleton/all_actions_annos.pkl'

print(f"Loading {original_pkl}...")
try:
    original_data = mmcv.load(original_pkl)
    print(f"Original data length: {len(original_data)}")
except FileNotFoundError:
    print(f"File not found: {original_pkl}. Starting with empty list.")
    original_data = []

print(f"Loading {new_pkl}...")
try:
    new_data = mmcv.load(new_pkl)
    print(f"New data length: {len(new_data)}")
except FileNotFoundError:
    print(f"File not found: {new_pkl}. Cannot merge.")
    exit(1)

# Create a dictionary for faster lookup and to avoid duplicates based on frame_dir
merged_dict = {item['frame_dir']: item for item in original_data}

# Update or add new data
for item in new_data:
    merged_dict[item['frame_dir']] = item

# Convert back to list
merged_data = list(merged_dict.values())
print(f"Merged data length: {len(merged_data)}")

# Save merged data
print(f"Saving merged data to {merged_pkl}...")
mmcv.dump(merged_data, merged_pkl)
print("Done.")
