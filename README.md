# FlameVQA

FlameVQA is a benchmarking workspace for evaluating vision-language models on wildfire-focused visual question answering tasks using RGB imagery, thermal imagery, or both together.

The repository is built around a unified evaluation pipeline that:
- loads checkpoint-style or list-style JSON datasets,
- runs multimodal inference with several VLM families,
- supports both `transformers` and `vllm` backends,
- evaluates ablations such as `rgb`, `thermal`, `rgb_thermal`, repeated prompting, and temperature-summary removal,
- writes per-question predictions and aggregate metrics to JSON/JSONL files.

## Supported models

Current model shortcuts:

| Argument | Model ID | Family |
|---|---|---|
| `llava` | `llava-hf/llava-v1.6-mistral-7b-hf` | LLaVA |
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` | Qwen3-VL |
| `llama3.2` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Llama Vision |
| `internvl2` | `OpenGVLab/InternVL2-8B` | InternVL2 |
| `minicpm` | `openbmb/MiniCPM-V-2_6` | MiniCPM-V |
| `pixtral` | `mistralai/Pixtral-12B-2409` | Pixtral |

See [MODEL_OPTIONS.md](MODEL_OPTIONS.md) for quick examples.

## Repository layout

Main files in this repo:

- [eval_vlm_checkpoint.py](eval_vlm_checkpoint.py) — main unified evaluator
- [eval_vlm_checkpoint_minicpm.py](eval_vlm_checkpoint_minicpm.py) — MiniCPM-focused evaluator and vLLM experiments
- [llama_eval_vlm_checkpoint.py](llama_eval_vlm_checkpoint.py) — alternate evaluator variant
- [run_all_evals.sh](run_all_evals.sh) — local sweep runner for models / modes / ablations
- [submit_evals.sh](submit_evals.sh) — generates a Slurm array submission map
- [eval_array.slurm](eval_array.slurm) — Slurm array job definition
- [update_response16_image_paths.py](update_response16_image_paths.py) — updates image path prefixes in dataset JSON files and validates accessibility
- [check_image_paths.py](check_image_paths.py) — path checking utility
- [response_v16](response_v16) — evaluation dataset JSON files
- [benchmark_results](benchmark_results) — generated predictions and metrics

## Dataset format

The evaluator supports two input styles:

1. **Checkpoint-style JSON**
   - top-level `type: "checkpoint"`
   - per-item metadata stored under `items`
   - human answer is mapped to `gt_answer`

2. **List-style JSON**
   - list of rows with question entries

Typical row fields include:
- `image_id`
- `rgb_path`
- `thermal_path`
- `temp_summary`
- `category`
- `question_id`
- `question`
- `options`
- `answer`

Example:

```json
{
  "image_id": "00001",
  "rgb_path": "/path/to/rgb.jpg",
  "thermal_path": "/path/to/thermal.jpg",
  "temp_summary": {
    "min": 16.9,
    "max": 25.2,
    "mean": 20.8,
    "top3_mean": 22.7
  },
  "category": "Presence and Detection",
  "question_id": "PD1",
  "question": "Are active thermal hotspots detected?",
  "options": ["Yes", "No"],
  "answer": "No"
}
```

## Environment setup

This repo includes a Conda environment file:
- [flamevqa.yml](flamevqa.yml)

Create and activate it:

```bash
conda env create -f flamevqa.yml
conda activate flamevqa
```

If you use gated Hugging Face models, configure a token in your shell or `.env` file. The batch script reads values such as:
- `HUGGINGFACE_TOKEN`
- `HF_API_TOKEN`
- `HF_HUB_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `HF_TOKEN`

## Basic usage

### Run a single evaluation

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llava \
  --input-mode rgb_thermal \
  --backend vllm \
  --outdir ./benchmark_results
```

### Evaluate only a fraction of images

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model qwen \
  --input-mode rgb_thermal \
  --backend transformers \
  --percent 0.1 \
  --outdir ./benchmark_results
```

### MiniCPM with vLLM

```bash
python eval_vlm_checkpoint_minicpm.py \
  --input ./response_v16 \
  --model minicpm \
  --backend vllm \
  --input-mode rgb_thermal \
  --percent 0.1 \
  --vllm-batch-size 1 \
  --outdir ./benchmark_results
```

### Dry run only

Use this to verify dataset rows and file paths without loading model weights:

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llava \
  --dry-run
```

## Important evaluator options

Common CLI flags:

- `--input` — input JSON file or directory of JSON files
- `--model` — one of `llava`, `qwen`, `llama3.2`, `internvl2`, `minicpm`, `pixtral`
- `--backend` — `transformers` or `vllm`
- `--input-mode` — `rgb`, `thermal`, or `rgb_thermal`
- `--percent` — sample fraction of unique images
- `--outdir` — output directory for predictions and metrics
- `--max-side` — image resize limit
- `--max-new-tokens` — generation cap
- `--repeat-prompt` — repeats the question after images to reduce lost-in-the-middle effects
- `--no-temp-summary` — removes numeric temperature summary from the prompt
- `--dry-run` — validate dataset and paths only

vLLM-specific flags:
- `--vllm-batch-size`
- `--vllm-tensor-parallel-size`
- `--vllm-gpu-memory-utilization`
- `--vllm-max-model-len`
- `--disable-vllm-prefix-caching`

## Batch execution

### Local sweep

[run_all_evals.sh](run_all_evals.sh) runs ablation sweeps across:
- models,
- input modes,
- repeated prompting,
- temperature-summary inclusion/removal.

Example:

```bash
bash run_all_evals.sh --percent 0.1 --model minicpm --input-mode rgb_thermal
```

Useful options include:
- `--dry-run`
- `--input-dir <path>`
- `--python-bin <executable>`
- `--model <name|a,b,c>`
- `--input-mode <mode|a,b,c>`
- `--percent <0..1>`
- `--outdir <path>`
- `--repeat-prompt`
- `--no-temp-summary`

### Slurm submission

[submit_evals.sh](submit_evals.sh) generates a flat command map and submits a Slurm job array via [eval_array.slurm](eval_array.slurm).

```bash
bash submit_evals.sh
```

## Output files

Each run writes:

1. **Predictions JSONL**
   - one line per evaluated question
   - includes model info, paths, question, options, ground truth, prediction, raw output, and errors

2. **Metrics JSON**
   - overall accuracy
   - per-category accuracy
   - bookkeeping about skipped rows, missing files, and exceptions

Output filenames encode the run configuration, for example:

```text
eval_minicpm_openbmb_MiniCPM-V-2_6_backend-vllm_mode-rgb_thermal_maxside-768_maxtok-128_repeatprompt-0_notempsummary-0_p10_seed123_preds.jsonl
```

## Image path rewriting

If dataset JSON files contain stale absolute image paths, use [update_response16_image_paths.py](update_response16_image_paths.py).

It can:
- replace old path prefixes recursively in JSON files,
- validate every absolute path found in each JSON,
- print missing or inaccessible files with error reasons.

Example:

```bash
python update_response16_image_paths.py --root-dir ./response_v16
```

Validate without modifying files:

```bash
python update_response16_image_paths.py --root-dir ./response_v16 --validate-only
```

## Notes on MiniCPM + vLLM

MiniCPM support in this workspace uses a dedicated vLLM path with multimodal prompt placeholders and `multi_modal_data` requests.

If you see vLLM initialization failures:
- ensure only one heavy vLLM engine is running on the GPU,
- reduce `--vllm-gpu-memory-utilization`,
- set `--vllm-batch-size 1` for debugging,
- try a dry run first to verify the dataset,
- check that image paths are valid and readable.

## Common issues

### 1. Missing image files
Run a dry run or use [update_response16_image_paths.py](update_response16_image_paths.py) to validate paths.

### 2. Hugging Face gated model access
For models such as Llama 3.2 Vision, authenticate first:

```bash
huggingface-cli login
```

### 3. vLLM startup or cache errors on clusters
This repo configures writable runtime cache directories under the output folder to avoid stale or inaccessible scratch/cache paths.

### 4. Concurrent GPU usage
Running multiple vLLM engines at once can cause startup failures or out-of-memory errors. Check active processes with:

```bash
nvidia-smi
```

## Results folders

Existing results are organized in folders such as:
- [benchmark_results](benchmark_results)
- [internvl2_results](internvl2_results)
- [minicpm_results](minicpm_results)
- [pixtral_results](pixtral_results)

## License

No license file is currently included in this repository. Add one if you plan to distribute the code publicly.
