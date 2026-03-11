#!/bin/bash
# Reroute Hugging Face cache to high-capacity storage
export HF_HOME="/scratch/mhabibp/hf_cache"  

# Trick Triton into finding the missing libcuda.so file
export LD_LIBRARY_PATH="$HOME/custom_cuda_lib:$LD_LIBRARY_PATH"
# Defaults
INPUT_DIR="./response_v16"
PYTHON_BIN="python"
DRY_RUN_MODE=false
PERCENT="1.0"
ALL_MODELS=("llava" "qwen" "llama3.2" "internvl2" "minicpm" "pixtral")
MODELS=()
REQUESTED_MODELS=()
INPUT_MODES=("rgb_thermal" "rgb" "thermal")
FORCE_REPEAT_PROMPT=false
FORCE_NO_TEMP_SUMMARY=false
OUTDIR=""

# --------------------------------------------------
# Hugging Face token (needed for gated models like Llama 3.2 Vision)
# Paste your token below before running, or export HF_TOKEN in shell.
# Example: HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# --------------------------------------------------
source .env  # Load from .env if it exists, but don't fail if it's missing.
# Resolve token from environment first (highest priority), then script-local HF_TOKEN.
EFFECTIVE_HF_TOKEN=""
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    EFFECTIVE_HF_TOKEN="$HUGGINGFACE_TOKEN"
elif [ -n "$HF_API_TOKEN" ]; then
    EFFECTIVE_HF_TOKEN="$HF_API_TOKEN"
elif [ -n "$HF_HUB_TOKEN" ]; then
    EFFECTIVE_HF_TOKEN="$HF_HUB_TOKEN"
elif [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
    EFFECTIVE_HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
elif [ -n "$HF_TOKEN" ]; then
    EFFECTIVE_HF_TOKEN="$HF_TOKEN"
fi

# Optional args:
#   --dry-run                 -> passes --dry-run to eval script
#   --input-dir <path>        -> override input directory
#   --python-bin <executable> -> override python executable
#   --model <name|a,b,c>      -> run subset of models (repeat flag or comma-separated)
#   --input-mode <mode|a,b,c> -> choose input mode(s): rgb_thermal,rgb,thermal
#   --percent <0..1>          -> fraction of unique images to evaluate (e.g., 0.1)
#   --outdir <path>           -> override output directory used by eval script
#   --repeat-prompt            -> sweep sandwich-prompt OFF/ON (without flag: OFF only)
#   --no-temp-summary          -> only run no-temp-summary branch
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN_MODE=true
            shift
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --percent)
            if [ -z "$2" ]; then
                echo "Missing value for --percent"
                echo "Usage: $0 [--dry-run] [--input-dir <path>] [--python-bin <executable>] [--percent <0..1>] [--outdir <path>] [--model <name|a,b,c>] [--input-mode <mode|a,b,c>] [--repeat-prompt] [--no-temp-summary]"
                exit 1
            fi
            if ! [[ "$2" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]]; then
                echo "Invalid --percent value: $2"
                echo "Use a number in (0,1], e.g., 0.1"
                exit 1
            fi
            if [ "$2" = "0" ] || [ "$2" = "0.0" ]; then
                echo "Invalid --percent value: $2"
                echo "Use a number in (0,1], e.g., 0.1"
                exit 1
            fi
            PERCENT="$2"
            shift 2
            ;;
        --model)
            if [ -z "$2" ]; then
                echo "Missing value for --model"
                echo "Usage: $0 [--dry-run] [--input-dir <path>] [--python-bin <executable>] [--percent <0..1>] [--outdir <path>] [--model <name|a,b,c>] [--input-mode <mode|a,b,c>] [--repeat-prompt] [--no-temp-summary]"
                exit 1
            fi
            IFS=',' read -r -a _models_split <<< "$2"
            for _m in "${_models_split[@]}"; do
                _m="${_m// /}"
                if [ -n "$_m" ]; then
                    REQUESTED_MODELS+=("$_m")
                fi
            done
            shift 2
            ;;
        --outdir)
            if [ -z "$2" ]; then
                echo "Missing value for --outdir"
                echo "Usage: $0 [--dry-run] [--input-dir <path>] [--python-bin <executable>] [--percent <0..1>] [--outdir <path>] [--model <name|a,b,c>] [--input-mode <mode|a,b,c>] [--repeat-prompt] [--no-temp-summary]"
                exit 1
            fi
            OUTDIR="$2"
            shift 2
            ;;
        --input-mode)
            if [ -z "$2" ]; then
                echo "Missing value for --input-mode"
                exit 1
            fi
            INPUT_MODES=()
            IFS=',' read -r -a _modes_split <<< "$2"
            for _mode in "${_modes_split[@]}"; do
                _mode="${_mode// /}"
                case "$_mode" in
                    rgb_thermal|rgb|thermal)
                        INPUT_MODES+=("$_mode")
                        ;;
                    *)
                        echo "Unknown input mode: $_mode"
                        echo "Supported input modes: rgb_thermal rgb thermal"
                        exit 1
                        ;;
                esac
            done
            if [ ${#INPUT_MODES[@]} -eq 0 ]; then
                echo "No valid input modes provided"
                exit 1
            fi
            shift 2
            ;;
        --repeat-prompt)
            FORCE_REPEAT_PROMPT=true
            shift
            ;;
        --no-temp-summary)
            FORCE_NO_TEMP_SUMMARY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--input-dir <path>] [--python-bin <executable>] [--percent <0..1>] [--outdir <path>] [--model <name|a,b,c>] [--input-mode <mode|a,b,c>] [--repeat-prompt] [--no-temp-summary]"
            exit 1
            ;;
    esac
done

# Resolve selected models (default: all)
if [ ${#REQUESTED_MODELS[@]} -eq 0 ]; then
    MODELS=("${ALL_MODELS[@]}")
else
    for req in "${REQUESTED_MODELS[@]}"; do
        found=false
        for supported in "${ALL_MODELS[@]}"; do
            if [ "$req" = "$supported" ]; then
                found=true
                break
            fi
        done
        if ! $found; then
            echo "Unknown model in --model: $req"
            echo "Supported models: ${ALL_MODELS[*]}"
            exit 1
        fi

        duplicate=false
        for existing in "${MODELS[@]}"; do
            if [ "$existing" = "$req" ]; then
                duplicate=true
                break
            fi
        done
        if ! $duplicate; then
            MODELS+=("$req")
        fi
    done
fi

# Define ablation arrays based on explicit user choice
# Default (no --repeat-prompt): run only non-sandwich branch.
# With --repeat-prompt: sweep both non-sandwich and sandwich branches.
if $FORCE_REPEAT_PROMPT; then
    REPEAT_PROMPTS=("" "--repeat-prompt")
else
    REPEAT_PROMPTS=("")
fi

if $FORCE_NO_TEMP_SUMMARY; then
    TEMP_SUMMARIES=("--no-temp-summary")
else
    TEMP_SUMMARIES=("" "--no-temp-summary")
fi

# Keep track of the run count
TOTAL_RUNS=$(( ${#MODELS[@]} * ${#INPUT_MODES[@]} * ${#REPEAT_PROMPTS[@]} * ${#TEMP_SUMMARIES[@]} ))
CURRENT_RUN=1

if $DRY_RUN_MODE; then
    echo "Starting $TOTAL_RUNS DRY-RUN validation runs..."
else
    echo "Starting $TOTAL_RUNS evaluation runs..."
fi
echo "Python: $PYTHON_BIN"
echo "Input dir: $INPUT_DIR"
if [ -n "$OUTDIR" ]; then
    echo "Outdir: $OUTDIR"
else
    echo "Outdir: default (eval script behavior)"
fi
echo "Percent: $PERCENT"
echo "Models: ${MODELS[*]}"
echo "Input modes: ${INPUT_MODES[*]}"
if $FORCE_REPEAT_PROMPT; then
    echo "Repeat prompt: sweep OFF/ON"
else
    echo "Repeat prompt: OFF only (pass --repeat-prompt to sweep OFF/ON)"
fi
if $FORCE_NO_TEMP_SUMMARY; then
    echo "Temp summary: forced OFF (--no-temp-summary)"
else
    echo "Temp summary: sweep ON/OFF"
fi
echo "=================================================="

# Preflight: if llama3.2 is requested, verify native vLLM Mllama support.
# This prevents silent fallback to transformers backend when true vLLM is required.
NEEDS_LLAMA_CHECK=false
for _m in "${MODELS[@]}"; do
    if [ "$_m" = "llama3.2" ]; then
        NEEDS_LLAMA_CHECK=true
        break
    fi
done

if $NEEDS_LLAMA_CHECK; then
    echo "Checking native vLLM support for llama3.2..."
    "$PYTHON_BIN" - <<'PY'
import sys

try:
    from vllm.model_executor.models import registry as vllm_registry
except Exception as exc:
    print(f"VLLM_CHECK_ERROR: unable to import vLLM registry: {exc}")
    sys.exit(2)

models = getattr(vllm_registry, "_VLLM_MODELS", {})
if "MllamaForConditionalGeneration" in models:
    print("VLLM_CHECK_OK: native Mllama support is available.")
    sys.exit(0)

print("VLLM_CHECK_FAIL: native Mllama support not found in this vLLM build.")
sys.exit(3)
PY

    CHECK_EXIT=$?
    if [ $CHECK_EXIT -ne 0 ]; then
        echo "ERROR: llama3.2 native vLLM support is unavailable in the current environment."
        echo "This setup will fall back to transformers instead of true vLLM kernels."
        echo "Suggested fix: install a vLLM build with native Mllama support (commonly vllm<=0.10.1)."
        echo "Example: pip uninstall -y vllm && pip install 'vllm==0.10.1'"
        exit 1
    fi
fi

# Global Hugging Face handshake.
# Important because HF_HOME is rerouted above; relying on prior ~/.cache auth can fail.
if [ -n "$EFFECTIVE_HF_TOKEN" ]; then
    export HF_TOKEN="$EFFECTIVE_HF_TOKEN"
    export HUGGINGFACE_TOKEN="$EFFECTIVE_HF_TOKEN"
    export HF_API_TOKEN="$EFFECTIVE_HF_TOKEN"
    export HF_HUB_TOKEN="$EFFECTIVE_HF_TOKEN"
    export HUGGINGFACE_HUB_TOKEN="$EFFECTIVE_HF_TOKEN"

    echo "Logging into Hugging Face for model access..."
    # Use python API directly to avoid CLI path/credential-helper issues across machines.
    "$PYTHON_BIN" - <<PY
try:
    from huggingface_hub import login
except Exception:
    print("missing_huggingface_hub")
else:
    login(token="""$EFFECTIVE_HF_TOKEN""", add_to_git_credential=False)
    print("ok")
PY
    if [ $? -ne 0 ]; then
        echo "WARNING: Hugging Face login failed. Gated models may fail if access is required."
    else
        echo "Hugging Face token exported to environment."
        echo "(If huggingface_hub is installed in $PYTHON_BIN, local HF_HOME token cache is also refreshed.)"
    fi
else
    echo "WARNING: No Hugging Face token provided; gated model downloads may fail."
    echo "Set HF_TOKEN in this script, or export HF_TOKEN/HUGGINGFACE_TOKEN/HF_API_TOKEN/HF_HUB_TOKEN before running."
fi

# Loop through all combinations
for MODEL in "${MODELS[@]}"; do
    for MODE in "${INPUT_MODES[@]}"; do
        for REPEAT in "${REPEAT_PROMPTS[@]}"; do
            for TEMP in "${TEMP_SUMMARIES[@]}"; do
                
                # 1. Start building the command with required arguments
                CMD=("$PYTHON_BIN" "llama_eval_vlm_checkpoint.py" "--input" "$INPUT_DIR" "--model" "$MODEL" "--input-mode" "$MODE" "--percent" "$PERCENT")

                # Use vLLM workaround path for Llama 3.2 Vision; keep transformers for others.
                if [ "$MODEL" = "llama3.2" ]; then
                    CMD+=("--backend" "vllm" "--vllm-model-impl" "vllm" "--force-vllm-llama" "--vllm-batch-size" "1" "--vllm-limit-mm-images" "1" "--vllm-gpu-memory-utilization" "0.85" "--vllm-max-model-len" "4096")
                else
                    CMD+=("--backend" "transformers")
                fi

                if [ -n "$OUTDIR" ]; then
                    CMD+=("--outdir" "$OUTDIR")
                fi
                
                # 2. Append optional flags if they are not empty
                if [ -n "$REPEAT" ]; then
                    CMD+=("$REPEAT")
                fi
                
                if [ -n "$TEMP" ]; then
                    CMD+=("$TEMP")
                fi

                if $DRY_RUN_MODE; then
                    CMD+=("--dry-run")
                fi
                
                # 3. Print out what is currently running
                echo "Run [$CURRENT_RUN/$TOTAL_RUNS]: ${CMD[*]}"
                
                # 4. Execute the command
                "${CMD[@]}"
                
                # Check if the python script failed
                if [ $? -ne 0 ]; then
                    echo "WARNING: Run $CURRENT_RUN failed. Moving to the next one."
                fi
                
                echo "--------------------------------------------------"
                ((CURRENT_RUN++))
                
            done
        done
    done
done

if $DRY_RUN_MODE; then
    echo "All dry-run validations completed!"
else
    echo "All evaluations completed!"
fi