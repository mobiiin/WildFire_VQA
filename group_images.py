#!/usr/bin/env python3
"""
group_images.py
Detects near-duplicate images in a VQA dataset.

IMPROVEMENTS:
1. Folder Isolation: Only compares images within the same directory.
2. Multimodal: Checks both RGB and Thermal. (If EITHER matches, it's a duplicate).
3. Tunable Knobs: Threshold, RANSAC, etc.
"""

import argparse
import cv2
import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_parent_folder(path):
    return os.path.dirname(os.path.abspath(path))

def load_image_data(json_dir):
    """
    Returns a dictionary grouped by folder:
    {
       "/path/to/location_A": [ {rgb:..., thr:...}, {rgb:..., thr:...} ],
       "/path/to/location_B": [ ... ]
    }
    """
    print(f"Scanning {json_dir}...")
    files = sorted([f for f in os.listdir(json_dir) if f.lower().endswith(".json")])
    
    # Store unique items to avoid processing the same image twice if it appears in multiple questions
    seen_rgb = set()
    grouped_data = defaultdict(list)
    
    for fn in files:
        fpath = os.path.join(json_dir, fn)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                results = data.get("results", []) if isinstance(data, dict) else data
                
                for item in results:
                    rgb = item.get("rgb_path", "").strip()
                    thr = item.get("thermal_path", "").strip()
                    
                    if rgb and rgb not in seen_rgb:
                        folder = get_parent_folder(rgb)
                        grouped_data[folder].append({"rgb": rgb, "thr": thr})
                        seen_rgb.add(rgb)
                        
        except Exception as e:
            print(f"[WARN] Failed to read {fn}: {e}")
    
    # Sort images within each folder to ensure sequential comparison
    for folder in grouped_data:
        grouped_data[folder].sort(key=lambda x: x["rgb"])
        
    return grouped_data

def check_pair(path1, path2, orb, bf, min_matches, ransac_thresh, ratio_thresh):
    """
    Helper to check if two image files match.
    """
    if not path1 or not path2 or not os.path.exists(path1) or not os.path.exists(path2):
        return False

    # 1. Load Grayscale
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None: return False

    # 2. Resize (Standardize size for consistent feature count)
    # 800px is a good balance for finding features in rotation
    h, w = img1.shape
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)

    # 3. Detect
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < min_matches or len(des2) < min_matches:
        return False

    # 4. Match
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    # 5. Geometric Verify
    if len(good) >= min_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is None: return False
        
        inliers = np.sum(mask)
        return inliers >= min_matches

    return False

def is_duplicate_multimodal(item1, item2, orb, bf, args):
    """
    Checks RGB first. If no match, checks Thermal.
    """
    # Check RGB
    if check_pair(item1["rgb"], item2["rgb"], orb, bf, args.threshold, args.ransac, args.ratio):
        return True
    
    # Check Thermal (Backup plan)
    if check_pair(item1["thr"], item2["thr"], orb, bf, args.threshold, args.ransac, args.ratio):
        return True
        
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Directory with dataset .json files")
    ap.add_argument("--out", default="image_groups.json", help="Output mapping file")
    
    # Tunable Knobs
    ap.add_argument("--threshold", type=int, default=15, help="Min matching keypoints (Default: 15)")
    ap.add_argument("--ransac", type=float, default=20.0, help="RANSAC error threshold (Default: 10.0)")
    ap.add_argument("--features", type=int, default=8000, help="ORB feature count (Default: 3000)")
    ap.add_argument("--ratio", type=float, default=0.8, help="Lowe's ratio test (Default: 0.8)")
    
    args = ap.parse_args()

    # Init ORB
    orb = cv2.ORB_create(nfeatures=args.features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 1. Load and Group by Folder
    grouped_data = load_image_data(args.json_dir)
    if not grouped_data:
        print("No images found.")
        return

    mapping = {}
    total_images = 0
    total_duplicates = 0
    
    print(f"Processing {len(grouped_data)} location folders...")

    # 2. Process each folder independently
    for folder, items in grouped_data.items():
        if not items: continue
        
        total_images += len(items)
        
        # The first image in the folder is the first Keyframe
        current_keyframe = items[0]
        mapping[current_keyframe["rgb"]] = current_keyframe["rgb"]
        
        # Loop through the rest of the folder
        # We use tqdm manually to show progress per folder is too noisy, 
        # so let's just print folder names or a global bar if preferred.
        # For simplicity, we just print the folder name.
        print(f"  -> Folder: {os.path.basename(folder)} ({len(items)} images)")

        for i in range(1, len(items)):
            curr_img = items[i]
            
            is_dup = is_duplicate_multimodal(current_keyframe, curr_img, orb, bf, args)
            
            if is_dup:
                mapping[curr_img["rgb"]] = current_keyframe["rgb"]
                total_duplicates += 1
            else:
                current_keyframe = curr_img
                mapping[curr_img["rgb"]] = current_keyframe["rgb"]

    # 3. Save
    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)

    unique_keys = set(mapping.values())
    reduction = 0
    if total_images > 0:
        reduction = 100 * (total_duplicates / total_images)

    print(f"\nDone!")
    print(f"  Total Images: {total_images}")
    print(f"  Keyframes (Unique): {len(unique_keys)}")
    print(f"  Duplicates Found: {total_duplicates}")
    print(f"  Workload Reduction: {reduction:.1f}%")
    print(f"  Map saved to: {args.out}")

if __name__ == "__main__":
    main()