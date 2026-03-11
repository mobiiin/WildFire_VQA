import os

path = "/home/mhabibp/anaconda3/envs/flamevqa/lib/python3.11/site-packages/vllm/model_executor/models/transformers/multimodal.py"

with open(path, "r") as f:
    content = f.read()

# The line exactly as it currently looks from the last sed command
target = 'image_tokens = mm_tokens["num_image_tokens"] if isinstance(mm_tokens["num_image_tokens"], int) else mm_tokens["num_image_tokens"][0]'

# The robust try/except replacement
replacement = """try:
            image_tokens = int(mm_tokens["num_image_tokens"][0])
        except Exception:
            image_tokens = int(mm_tokens["num_image_tokens"])"""

if target in content:
    with open(path, "w") as f:
        f.write(content.replace(target, replacement))
    print("✅ Python patch applied successfully!")
else:
    print("⚠️ Target string not found. It might already be patched.")
