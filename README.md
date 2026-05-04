# FlameVQA Evaluation Toolkit

This repository contains a unified evaluation pipeline for multimodal VLM benchmarking on FlameVQA-style datasets.

- Main evaluator: `eval_vlm_checkpoint.py`
- Batch ablation runner: `run_all_evals.sh`

## Open Dataset Releases

- Kaggle: https://www.kaggle.com/datasets/caseypiere/wildfire-vqa
- Hugging Face: https://huggingface.co/datasets/mobiiin/WildFire_VQA
- FLAME-3: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management
- FLAME-2: https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset
- FLAME-1: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

This project is open-source and the WildFire VQA dataset is publicly available on both platforms above.

The pipeline supports:
- Multiple open-source VLMs
- Multiple input modes (`rgb`, `thermal`, `rgb_thermal`)
- Prompt sandwich ablation (`--repeat-prompt`)
- Temperature-summary ablation (`--no-temp-summary`)
- Dataset validation mode (`--dry-run`)

---

## 1) Supported Models

`eval_vlm_checkpoint.py --model` supports:
- `llava` → `llava-hf/llava-v1.6-mistral-7b-hf`
- `qwen` → `Qwen/Qwen3-VL-8B-Instruct`
- `llama3.2` → `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `internvl2` → `OpenGVLab/InternVL2-8B`
- `minicpm` → `openbmb/MiniCPM-V-2_6`
- `pixtral` → `mistralai/Pixtral-12B-2409`

---

## 2) Installation

## 2.1 Create environment

```bash
conda create -n flamevqa python=3.11 -y
conda activate flamevqa
```

## 2.2 Install dependencies

Install PyTorch for your CUDA version first (recommended from pytorch.org), then:

```bash
pip install transformers pillow tqdm huggingface_hub
pip install vllm
```

Optional / model-specific:

```bash
pip install bitsandbytes
pip install accelerate
```

---

## 3) Hugging Face Access (gated models)

Some models (especially Llama vision models) may require HF auth:

```bash
huggingface-cli login
```

or export token:

```bash
export HUGGINGFACE_TOKEN="hf_xxx"
```

`run_all_evals.sh` includes a token-based login step for `llama3.2` runs.

---

## 4) Dataset Format Expectations

Evaluator accepts:
- Checkpoint-style JSON (`{"type":"checkpoint","items":...}`)
- List-style JSON (`[{...}, {...}]`)

Required per-row fields for evaluation:
- `question_id`
- `question`
- `options` (non-empty list)
- `gt_answer` (or `answer` in list-style)
- `rgb_path` for modes `rgb`, `rgb_thermal`
- `thermal_path` for modes `thermal`, `rgb_thermal`

Optional:
- `temp_summary` (`min`, `max`, `mean`, `top3_mean`)

---

## 5) Single-Run Usage (`eval_vlm_checkpoint.py`)

Basic:

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model qwen \
  --input-mode rgb_thermal \
  --outdir ./benchmark_results
```

Common flags:
- `--backend {transformers,vllm}`
- `--repeat-prompt`
- `--no-temp-summary`
- `--percent 0.2`
- `--dry-run`

Example (ablation):

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model qwen \
  --backend transformers \
  --input-mode thermal \
  --repeat-prompt \
  --no-temp-summary \
  --outdir ./benchmark_results
```

Dry validation only:

```bash
python eval_vlm_checkpoint.py \
  --input ./response_v16 \
  --model llava \
  --input-mode rgb_thermal \
  --dry-run
```

---

## 6) Full Sweep Usage (`run_all_evals.sh`)

Make executable:

```bash
chmod +x run_all_evals.sh
```

Default behavior:
- Sweeps all 3 input modes
- Sweeps repeat prompt ON/OFF
- Sweeps temp summary ON/OFF

So per model: `3 × 2 × 2 = 12` runs.

### Select model subset

```bash
./run_all_evals.sh --model qwen
./run_all_evals.sh --model llava,qwen
```

### Force only repeat-prompt branch

```bash
./run_all_evals.sh --model qwen --repeat-prompt
```

Per model here: `3 × 1 × 2 = 6` runs.

### Force only no-temp-summary branch

```bash
./run_all_evals.sh --model qwen --no-temp-summary
```

### Dry-run sweep (no model inference)

```bash
./run_all_evals.sh --dry-run --model qwen
```

### Use specific Python env

```bash
./run_all_evals.sh --model qwen --python-bin /home/<user>/anaconda3/envs/flamevqa/bin/python
```

---

## 7) Output Files

For each run, output files are written to `--outdir` with parameterized names:

- `*_preds.jsonl`: per-question predictions
- `*_metrics.json`: aggregate metrics + run metadata
- `*_dryrun.json`: dry-run validation report (if `--dry-run`)

`metrics.json` includes:
- `overall` accuracy
- `by_category`
- `run_config`
- `more_info` (queue, skipped, exceptions, etc.)

---

## 8) Reproducibility Tips

- Pin package versions before long experiments.
- Keep a fixed `--seed` and `--percent` for fair ablations.
- Use `--dry-run` before launching long sweeps.
- Save terminal logs per run batch.

---

## 9) Troubleshooting

### A) `ModuleNotFoundError` (torch/transformers/vllm)
Install missing packages in the same Python env used for execution.

### B) Llama + vLLM startup errors (`MllamaProcessor` / multimodal token issues)
Use transformers backend for Llama runs:

```bash
python eval_vlm_checkpoint.py --model llama3.2 --backend transformers ...
```

### C) HF login fails but token is correct
- Ensure token has model access scope.
- Confirm model access is approved on Hugging Face.
- Try explicit login in the same environment:

```bash
python -c "from huggingface_hub import login; login('hf_xxx')"
```

### D) Missing file skips
Check `rgb_path` / `thermal_path` validity and relative vs absolute paths in JSON.

---

## 10) Recommended Workflow

1. Validate combinations quickly:

```bash
./run_all_evals.sh --dry-run --model qwen
```

2. Run one short sample eval:

```bash
python eval_vlm_checkpoint.py --input ./response_v16 --model qwen --percent 0.05 --outdir ./smoke_results
```

3. Launch full ablation sweep:

```bash
./run_all_evals.sh --model qwen
```

4. Compare `*_metrics.json` across runs for ablation analysis.

---

## 11) How to Cite

```bibtex
@misc{habibpour2026wildfirevqalargescaleradiometricthermal,
  title={WildFireVQA: A Large-Scale Radiometric Thermal VQA Benchmark for Aerial Wildfire Monitoring}, 
  author={Mobin Habibpour and Niloufar Alipour Talemi and John Spodnik and Camren J. Khoury and Fatemeh Afghah},
  year={2026},
  eprint={2604.20190},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.20190}, 
}
```
