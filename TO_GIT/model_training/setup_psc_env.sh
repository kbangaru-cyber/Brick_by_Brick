#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  One-time setup for PSC BrickAgent training environment
# ═══════════════════════════════════════════════════════════════
#
#  Run this ONCE in an interactive GPU session:
#    interact -p GPU-shared --gpus=h100-80:1 -t 01:00:00 -A YOUR_CHARGE_ID
#    cd $PROJECT/brickagent
#    bash setup_psc_env.sh
#
#  Must run on a GPU node because causal-conv1d compiles
#  CUDA kernels for the H100 architecture.
# ═══════════════════════════════════════════════════════════════

set -e

module purge
module load cuda
module load anaconda3

# ── Create conda env ──────────────────────────────────────────
echo "Creating conda environment..."
conda create -n brickagent python=3.11 -y
conda activate brickagent

# ── PyTorch ───────────────────────────────────────────────────
echo "Installing PyTorch..."
pip install torch torchvision

# ── Unsloth + training stack ──────────────────────────────────
echo "Installing Unsloth + training libraries..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate bitsandbytes xformers
pip install datasets hf_transfer huggingface_hub numpy

# ── Flash-linear-attention (for GatedDeltaNet layers) ─────────
echo "Installing flash-linear-attention..."
pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git"

# ── Causal-conv1d (needs GPU to compile) ──────────────────────
echo "Compiling causal-conv1d for H100 (sm_90)..."
export TORCH_CUDA_ARCH_LIST="9.0"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=4
pip install "causal-conv1d>=1.6.0" --no-build-isolation

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Verifying installation..."
echo "═══════════════════════════════════════════════════════════"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPU:     {torch.cuda.get_device_name(0)}')
import fla
print(f'  flash-linear-attention: OK')
import causal_conv1d
print(f'  causal-conv1d: OK (v{causal_conv1d.__version__})')
import unsloth
print(f'  unsloth: OK')
print()
print('  All good!')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Edit run_train.sh — set YOUR_CHARGE_ID and HF_TOKEN"
echo "  2. mkdir -p logs"
echo "  3. sbatch run_train.sh"
echo ""
