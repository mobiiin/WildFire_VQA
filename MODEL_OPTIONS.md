# Supported VLM Models in eval_vlm_checkpoint.py

## Available Models

| Model Argument | Full Model ID | Family | Parameters |
|---------------|---------------|--------|------------|
| `llava` | `llava-hf/llava-v1.6-mistral-7b-hf` | llava | 7B |
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` | qwen | 8B |
| `llama3.2` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | llama | 11B |
| `internvl2` | `OpenGVLab/InternVL2-8B` | internvl | 8B |
| `minicpm` | `openbmb/MiniCPM-V-2_6` | minicpm | 8B |
| `pixtral` | `mistralai/Pixtral-12B-2409` | pixtral | 12B |

## Usage Examples

### LLaVA 1.6 (Existing)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llava \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./results_llava
```

### Qwen3-VL (Existing)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model qwen \
  --input-mode rgb_thermal \
  --backend transformers \
  --outdir ./results_qwen
```

### Llama 3.2 Vision 11B (NEW)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llama3.2 \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./results_llama32
```

### InternVL2-8B (NEW)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model internvl2 \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./results_internvl2
```

### MiniCPM-V-2.6 (NEW)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model minicpm \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./results_minicpm
```

### Pixtral 12B (NEW)
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model pixtral \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./results_pixtral
```

## Ablation Configurations

All models support the following ablations:

- **Input modes**: `rgb`, `thermal`, `rgb_thermal`
- **Sandwich prompting**: `--repeat-prompt`
- **Temperature summary**: `--no-temp-summary`
- **Backends**: `transformers`, `vllm`

### Full Ablation Example
```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llama3.2 \
  --input-mode thermal \
  --backend vllm \
  --repeat-prompt \
  --no-temp-summary \
  --percent 0.2 \
  --outdir ./results_ablation
```

## Backend Recommendations

| Model | Recommended Backend | Notes |
|-------|-------------------|-------|
| llava | vllm | Best performance with PagedAttention |
| qwen | vllm | Native vLLM support |
| llama3.2 | vllm | Official vLLM support |
| internvl2 | vllm | Good vLLM compatibility |
| minicpm | vllm or transformers | Test both for stability |
| pixtral | vllm | Native Mistral support |

## Model-Specific Notes

### Llama 3.2 Vision
- Requires authentication token for gated model access
- Use: `huggingface-cli login` before running
- Supports flash attention 2

### InternVL2
- Excellent multilingual support
- Good for OCR-heavy tasks

### MiniCPM-V-2.6
- Efficient 8B parameter model
- Good balance of speed and accuracy

### Pixtral 12B
- Latest from Mistral AI
- Strong instruction following
