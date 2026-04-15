import json
import argparse

def extract_unevaluated_images(groups_json_path, output_filename):
    print(f"Loading data from {groups_json_path}...")
    
    with open(groups_json_path, 'r') as f:
        data = json.load(f)
    
    # 1. Identify the Full Dataset (All Keys)
    all_images = set(data.keys())
    
    # 2. Identify the Evaluated Images (All Unique Values)
    # These are the keyframes humans have already looked at.
    evaluated_images = set(data.values())
    
    print(f"Total images in dataset: {len(all_images)}")
    print(f"Images already evaluated (Keyframes): {len(evaluated_images)}")
    
    # 3. Find the Difference (Unevaluated = All - Evaluated)
    # These are the images that were NOT selected as keyframes.
    unevaluated_images = list(all_images - evaluated_images)
    
    print(f"Images remaining for evaluation: {len(unevaluated_images)}")
    
    # 4. Save the result
    # We save it as a simple list of file paths.
    with open(output_filename, 'w') as f:
        json.dump(unevaluated_images, f, indent=2)
        
    print(f"Saved unevaluated image list to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images that have not been human-evaluated.")
    parser.add_argument("--groups_json", type=str, default="image_groups_80.json", help="Path to image_groups_80.json")
    parser.add_argument("--output", type=str, default="remaining_images.json", help="Output filename")
    
    args = parser.parse_args()
    
    extract_unevaluated_images(args.groups_json, args.output)