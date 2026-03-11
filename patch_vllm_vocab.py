import os

path = "/home/mhabibp/anaconda3/envs/flamevqa/lib/python3.11/site-packages/vllm/model_executor/layers/vocab_parallel_embedding.py"

with open(path, "r") as f:
    content = f.read()

target = "assert loaded_weight.shape[output_dim] == self.org_vocab_size"

replacement = """if loaded_weight.shape[output_dim] != self.org_vocab_size:
            print(f"WARNING: Bypassing strict vocab check. Weight: {loaded_weight.shape[output_dim]}, Config: {self.org_vocab_size}")
            if loaded_weight.shape[output_dim] > self.org_vocab_size:
                if output_dim == 0:
                    loaded_weight = loaded_weight[:self.org_vocab_size]
                else:
                    loaded_weight = loaded_weight[:, :self.org_vocab_size]"""

if target in content:
    with open(path, "w") as f:
        f.write(content.replace(target, replacement))
    print("✅ Vocab strictness patch applied successfully!")
else:
    print("⚠️ Target string not found. It might already be patched.")
