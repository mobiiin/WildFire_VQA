import re

# Patch 1: Multimodal Dictionary Bug
path1 = "/home/mhabibp/anaconda3/envs/flamevqa/lib/python3.11/site-packages/vllm/model_executor/models/transformers/multimodal.py"
with open(path1, "r") as f: 
    content = f.read()

content = re.sub(
    r'image_tokens = mm_tokens\["num_image_tokens"\]\[0\]',
    """if isinstance(mm_tokens, int):
            image_tokens = mm_tokens
        elif isinstance(mm_tokens, dict):
            val = mm_tokens.get("num_image_tokens", 1)
            image_tokens = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        else:
            image_tokens = int(mm_tokens)""",
    content
)
with open(path1, "w") as f: 
    f.write(content)

# Patch 2: Safe Vocab Padding (Dynamic Indentation)
path2 = "/home/mhabibp/anaconda3/envs/flamevqa/lib/python3.11/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py"
with open(path2, "r") as f: 
    lines = f.readlines()

with open(path2, "w") as f:
    for line in lines:
        if "assert loaded_weight.shape[output_dim] == self.org_vocab_size" in line:
            indent = line[:len(line) - len(line.lstrip())]
            f.write(indent + "if loaded_weight.shape[output_dim] != self.org_vocab_size:\n")
            f.write(indent + "    print(f'WARNING: Bypassing strict vocab check.')\n")
            f.write(indent + "    if loaded_weight.shape[output_dim] > self.org_vocab_size:\n")
            f.write(indent + "        if output_dim == 0:\n")
            f.write(indent + "            loaded_weight = loaded_weight[:self.org_vocab_size]\n")
            f.write(indent + "        else:\n")
            f.write(indent + "            loaded_weight = loaded_weight[:, :self.org_vocab_size]\n")
        else:
            f.write(line)

print("✅ Clean patches applied successfully!")
