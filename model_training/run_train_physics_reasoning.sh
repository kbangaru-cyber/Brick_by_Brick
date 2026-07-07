#!/bin/bash
#SBATCH --job-name=brick_physics_reasoning
#SBATCH --account=cis260075p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis260075p/bangarug/brickagent/logs/brick_s6_full_%j.out
#SBATCH --error=/ocean/projects/cis260075p/bangarug/brickagent/logs/brick_s6_full_%j.err

echo "=== physics_reasoning (Stage 6 full) started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export BRICK_BASE_DIR=/ocean/projects/cis260075p/bangarug/brickagent
export HF_HOME=/local/hf-cache
export HF_HUB_CACHE=/local/hf-cache/hub
export TRANSFORMERS_CACHE=/local/hf-cache/hub
export TRITON_CACHE_DIR=$BRICK_BASE_DIR/triton_cache
export XDG_CACHE_HOME=$BRICK_BASE_DIR/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage model to /local NVMe for fast read
mkdir -p /local/hf-cache/hub
if [ ! -f /local/hf-cache/hub/.staged_qwen25 ]; then
    rsync -a $BRICK_BASE_DIR/hf-cache/hub/models--unsloth--Qwen2.5-32B-Instruct /local/hf-cache/hub/ 2>/dev/null || true
    touch /local/hf-cache/hub/.staged_qwen25
fi

cd $BRICK_BASE_DIR
source /ocean/projects/cis260075p/bangarug/conda_envs/rhino-agent/bin/activate 2>/dev/null ||     conda activate /ocean/projects/cis260075p/bangarug/conda_envs/rhino-agent

python train_physics_reasoning.py

echo "=== physics_reasoning (Stage 6 full) finished at $(date) ==="
