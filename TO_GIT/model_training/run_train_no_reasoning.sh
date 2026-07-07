#!/bin/bash
#SBATCH --job-name=brickagent
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=cis260009p
#SBATCH --output=logs/brickagent_%j.out
#SBATCH --error=logs/brickagent_%j.err

set -e

# Load modules
module purge
module load gcc/10.2.0
module load cuda/12.6.1
module load anaconda3

# Activate env
export CONDA_ENVS_PATH="/ocean/projects/cis260009p/bangarug/conda_envs"
conda activate brickagent

# Environment
export BRICK_BASE_DIR="/ocean/projects/cis260009p/bangarug/brickagent"
export HF_TOKEN="${HF_TOKEN:?set your own Hugging Face token before running}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0

# Run
echo "=============================="
echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Job ID:       $SLURM_JOB_ID"
echo "=============================="

mkdir -p /ocean/projects/cis260009p/bangarug/brickagent/logs

cd /ocean/projects/cis260009p/bangarug/brickagent
python -c "
import train_no_reasoning as b
b.STAGE_CONFIG[1]['batch_size'] = 4
b.STAGE_CONFIG[1]['grad_accum'] = 4
b.train()
"

echo "=============================="
echo "Job finished: $(date)"
echo "=============================="
