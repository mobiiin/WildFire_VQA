import argparse
import json
import os
import re

# CHANGED: base folder where your jsonl files live
BASE_DIR = "/home/mhabibp/FlameVQA"  # CHANGED

def frame_key(obj):
    img = obj.get("image_id", "")

    # header line first
    if "prompt_template" in obj and not img:
        return (-1, -1)

    # try direct int parse (e.g., "00001")
    try:
        return (0, int(img))
    except Exception:
        pass

    # fallback: extract last number
    m = re.findall(r"\d+", str(img))
    if m:
        return (0, int(m[-1]))

    return (1, float("inf"))

def main():
    parser = argparse.ArgumentParser()
    # CHANGED: accept filename from CLI and append to BASE_DIR
    parser.add_argument("filename", help="JSONL filename (e.g., vqa_response_Sycan_2A_FIRE.jsonl)")
    args = parser.parse_args()

    # CHANGED: build full path = BASE_DIR + filename
    jsonl_path = os.path.join(BASE_DIR, args.filename)
    json_path = jsonl_path.replace(".jsonl", ".json")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # CHANGED: sort by frame/image_id
    records.sort(key=frame_key)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Converted + sorted:\n  {jsonl_path}\n-> {json_path}\nTotal records: {len(records)}")

if __name__ == "__main__":
    main()
