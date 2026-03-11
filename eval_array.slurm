#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1

# Environment stability fixes
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_EAGER=1
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=ERROR

# Ensure log directory exists
mkdir -p logs

# Retrieve the specific command for this Array Task ID
CMD_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" .job_map.txt)

echo "Task ID $SLURM_ARRAY_TASK_ID executing: $CMD_LINE"

# Execute the command
eval $CMD_LINE

if [ $? -ne 0 ]; then
    echo "ERROR: Task $SLURM_ARRAY_TASK_ID failed."
    exit 1
fi
