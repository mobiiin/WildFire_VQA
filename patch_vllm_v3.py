import re

path = "/home/mhabibp/anaconda3/envs/flamevqa/lib/python3.11/site-packages/vllm/model_executor/models/transformers/multimodal.py"

with open(path, "r") as f:
    content = f.read()

# Pattern 1: Matches the try-except block from our last attempt
pattern1 = re.compile(r'try:\s*image_tokens = int\(mm_tokens\["num_image_tokens"\]\[0\]\)\s*except Exception:\s*image_tokens = int\(mm_tokens\["num_image_tokens"\]\)')

# Pattern 2: Matches the original vLLM unpatched line (just in case)
pattern2 = re.compile(r'image_tokens = mm_tokens\["num_image_tokens"\]\[0\]')

# The ultimate fallback logic
replacement = """if isinstance(mm_tokens, int):
            image_tokens = mm_tokens
        elif isinstance(mm_tokens, dict):
            val = mm_tokens.get("num_image_tokens", 1)
            image_tokens = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        else:
            image_tokens = int(mm_tokens)"""

patched = False
if pattern1.search(content):
    content = pattern1.sub(replacement, content)
    print("✅ Patch v3 successful (replaced previous try-except block)!")
    patched = True
elif pattern2.search(content):
    content = pattern2.sub(replacement, content)
    print("✅ Patch v3 successful (replaced original vLLM line)!")
    patched = True
else:
    print("⚠️ Could not find the text to patch. It might already be fixed.")

if patched:
    with open(path, "w") as f:
        f.write(content)
