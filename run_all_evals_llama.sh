#!/bin/bash

# ===================================================
# IMPORTANT: Set CUDA_VISIBLE_DEVICES FIRST to avoid vLLM GPU conflicts
# ===================================================
USER_SET_CUDA_VISIBLE_DEVICES=false
BEST_SINGLE_GPU=""
BEST_TWO_GPUS=""

# Auto-detect available GPUs and keep track of the freest devices.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -ge 1 ]; then
        GPU_ORDER=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -nr | awk -F',' '{gsub(/ /, "", $1); print $1}')
        BEST_SINGLE_GPU=$(echo "$GPU_ORDER" | head -n 1)
        BEST_TWO_GPUS=$(echo "$GPU_ORDER" | head -n 2 | paste -sd, -)
        if [ -n "$BEST_SINGLE_GPU" ]; then
            export CUDA_VISIBLE_DEVICES="$BEST_SINGLE_GPU"
            echo "Auto-detected $GPU_COUNT GPU(s). Setting default CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        fi
        if [ -n "$BEST_TWO_GPUS" ]; then
            echo "Freest GPUs detected for multi-GPU runs: $BEST_TWO_GPUS"
        fi
    else
        echo "WARNING: No NVIDIA GPUs detected. Proceeding anyway..."
    fi
else
    USER_SET_CUDA_VISIBLE_DEVICES=true
    echo "CUDA_VISIBLE_DEVICES already set to: $CUDA_VISIBLE_DEVICES"
fi
# ===================================================

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
NO_TEMP_SUMMARY_MODE="both"  # "both", "yes", or "no"
OUTDIR=""
REQUESTED_VLLM_FALLBACK_EXIT_CODE=86

# --------------------------------------------------
# Hugging Face token (needed for gated models like Llama 3.2 Vision)
# Paste your token below before running, or export HF_TOKEN in shell.
# Example: HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# --------------------------------------------------
source .env

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
#   --no-temp-summary [yes|no] -> choose temp-summary branch (default: sweep both)
#                                 yes/true/on/1 -> only no-temp-summary
#                                 no/false/off/0 -> only with temp-summary
#                                 (omit value to sweep both)
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
            # Check if a value is provided (not another flag and not end of args)
            if [ -z "$2" ] || [[ "$2" =~ ^- ]]; then
                # No value specified, default to sweep both branches
                NO_TEMP_SUMMARY_MODE="both"
                shift
            else
                # Value specified, validate it
                case "$2" in
                    yes|true|on|1)
                        NO_TEMP_SUMMARY_MODE="yes"
                        shift 2
                        ;;
                    no|false|off|0)
                        NO_TEMP_SUMMARY_MODE="no"
                        shift 2
                        ;;
                    *)
                        echo "Invalid value for --no-temp-summary: $2"
                        echo "Use: yes, no, true, false, on, off, 1, or 0"
                        exit 1
                        ;;
                esac
            fi
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

case "$NO_TEMP_SUMMARY_MODE" in
    yes)
        TEMP_SUMMARIES=("--no-temp-summary")
        ;;
    no)
        TEMP_SUMMARIES=("")
        ;;
    both|*)
        TEMP_SUMMARIES=("" "--no-temp-summary")
        ;;
esac

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
case "$NO_TEMP_SUMMARY_MODE" in
    yes)
        echo "Temp summary: forced OFF (--no-temp-summary)"
        ;;
    no)
        echo "Temp summary: forced ON (no --no-temp-summary)"
        ;;
    both|*)
        echo "Temp summary: sweep ON/OFF"
        ;;
esac
echo "=================================================="

if [ -n "$BEST_SINGLE_GPU" ]; then
    echo "Preferred single-GPU target: $BEST_SINGLE_GPU"
fi
if [ -n "$BEST_TWO_GPUS" ]; then
    echo "Preferred two-GPU target: $BEST_TWO_GPUS"
fi
echo "=================================================="

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
                
                RUN_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
                RUN_TP_SIZE="1"
                RUN_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

                # 1. Start building the command with required arguments
                CMD=("$PYTHON_BIN" "eval_vlm_checkpoint_llama.py" "--input" "$INPUT_DIR" "--model" "$MODEL" "--input-mode" "$MODE" "--percent" "$PERCENT")

                if [ "$MODEL" = "llama3.2" ]; then
                    if ! $USER_SET_CUDA_VISIBLE_DEVICES && [ -n "$BEST_TWO_GPUS" ] && [[ "$BEST_TWO_GPUS" == *,* ]]; then
                        RUN_CUDA_VISIBLE_DEVICES="$BEST_TWO_GPUS"
                        RUN_TP_SIZE="2"
                    elif ! $USER_SET_CUDA_VISIBLE_DEVICES && [ -n "$BEST_SINGLE_GPU" ]; then
                        RUN_CUDA_VISIBLE_DEVICES="$BEST_SINGLE_GPU"
                    fi

                    CMD+=(
                        "--vllm-tensor-parallel-size" "$RUN_TP_SIZE"
                        "--vllm-batch-size" "1"
                        "--vllm-max-model-len" "8192"
                        "--vllm-gpu-memory-utilization" "0.75"
                        "--disable-vllm-prefix-caching"
                    )
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
                echo "Run [$CURRENT_RUN/$TOTAL_RUNS]: CUDA_VISIBLE_DEVICES=$RUN_CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=$RUN_PYTORCH_CUDA_ALLOC_CONF ${CMD[*]}"
                
                # 4. Execute the command
                CUDA_VISIBLE_DEVICES="$RUN_CUDA_VISIBLE_DEVICES" \
                PYTORCH_CUDA_ALLOC_CONF="$RUN_PYTORCH_CUDA_ALLOC_CONF" \
                "${CMD[@]}"
                STATUS=$?
                
                # Check if the python script failed
                if [ $STATUS -eq $REQUESTED_VLLM_FALLBACK_EXIT_CODE ]; then
                    echo "ERROR: Run $CURRENT_RUN aborted because requested vLLM would fall back to transformers. EXIT_CODE=$STATUS"
                    exit $STATUS
                fi
                if [ $STATUS -ne 0 ]; then
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