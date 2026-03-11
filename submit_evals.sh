#!/bin/bash

# Configuration
INPUT_DIR="./response_v16"
PYTHON_BIN="python"
MODELS=("llava" "qwen" "llama3.2" "internvl2" "minicpm" "pixtral")
INPUT_MODES=("rgb_thermal" "rgb" "thermal")
REPEAT_PROMPTS=("" "--repeat-prompt")
TEMP_SUMMARIES=("" "--no-temp-summary")

MAP_FILE=".job_map.txt"
> $MAP_FILE 

# Nested loops to generate the flat command list
count=0
for MODEL in "${MODELS[@]}"; do
    for MODE in "${INPUT_MODES[@]}"; do
        for REPEAT in "${REPEAT_PROMPTS[@]}"; do
            for TEMP in "${TEMP_SUMMARIES[@]}"; do
                # Build command string
                CMD="$PYTHON_BIN eval_vlm_checkpoint.py --input $INPUT_DIR --model $MODEL --input-mode $MODE $REPEAT $TEMP"
                echo "$CMD" >> $MAP_FILE
                ((count++))
            done
        done
    done
done

TOTAL_IDX=$((count - 1))

echo "Generated $count total jobs."
echo "Submitting Slurm array 0-$TOTAL_IDX..."

# Launch the array
sbatch --array=0-$TOTAL_IDX eval_array.slurm
