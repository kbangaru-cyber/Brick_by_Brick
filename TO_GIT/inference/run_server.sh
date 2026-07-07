#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  run_server.sh — BrickAgent Orchestrator Launch
#
#  STEP 1 — Request an interactive GPU node from the login node:
#
#  RECOMMENDED — 1× L40S-48GB  (SM 8.9, bfloat16):
#    srun --ntasks=1 --gres=gpu:l40s-48:1 \
#         --partition=GPU-shared \
#         --mem=80G --time=4:00:00 --pty bash
#
#  BEST — 1× H100-80GB:
#    srun --ntasks=1 --gres=gpu:h100-80:1 \
#         --partition=GPU-shared \
#         --mem=80G --time=4:00:00 --pty bash
#
#  STEP 2 — From the compute node shell:
#    cd /ocean/projects/cis260075p/bangarug/brickagent
#    bash run_server.sh
#
# ═══════════════════════════════════════════════════════════════════

set -e

# ── paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# checkpoints/ at the repo root: no_reasoning (stage 1), physics_reasoning (stage 6 full)
CHECKPOINTS_DIR="${BRICKAGENT_CHECKPOINTS_DIR:-$SCRIPT_DIR/../checkpoints}"
ADAPTER_PATH="${ADAPTER_PATH:-$CHECKPOINTS_DIR/physics_reasoning}"
ADAPTER_STAGE1="${ADAPTER_STAGE1:-$CHECKPOINTS_DIR/no_reasoning}"
ADAPTER_STAGE5="${ADAPTER_STAGE5:-$CHECKPOINTS_DIR/physics_reasoning}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen3.5-27B}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/ocean/projects/cis260075p/bangarug/brickagent/hf-cache}"   # canonical (Lustre) — read directly, no NVMe staging
PORT="${PORT:-8080}"

# Models are read directly from /ocean — no rsync to node-local NVMe.
mkdir -p "${HF_CACHE_DIR}/hub"

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HF_HUB_OFFLINE=1   # don't phone home — everything is on /ocean

# Every cache lives on /ocean — one project directory, one place to clean up.
# Triton compiles GPU kernels on first run (Qwen3.5 uses fla/gated-delta-rule).
# TorchInductor only kicks in if torch.compile() is called (not used here, but
# redirected anyway for completeness). TMPDIR catches everything else that
# would otherwise spill to /tmp.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/ocean/projects/cis260075p/bangarug/brickagent/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/ocean/projects/cis260075p/bangarug/brickagent/torchinductor-cache}"
export TMPDIR="${TMPDIR:-/ocean/projects/cis260075p/bangarug/brickagent/tmp}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

echo ""
echo "============================================================"
echo "  BrickAgent Orchestrator"
echo "  Planner       : ${PLANNER_MODEL}        (VL negotiator)"
echo "  Executor      : Qwen2.5-32B (base, shared across adapters)"
echo "  Default (physics_reasoning) : ${ADAPTER_PATH}"
echo "  Stage 1  (no_reasoning)      : ${ADAPTER_STAGE1}"
echo "  Stage 6  (physics_reasoning) : ${ADAPTER_STAGE5}"
echo "  HF cache      : ${HF_CACHE_DIR}"
echo "  Triton cache  : ${TRITON_CACHE_DIR}"
echo "  Inductor cache: ${TORCHINDUCTOR_CACHE_DIR}"
echo "  TMPDIR        : ${TMPDIR}"
echo "  (all paths on Lustre — no /tmp or NVMe usage)"
echo "============================================================"
echo ""

# ── GPU check ──
echo "[0/3] Checking GPUs ..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || \
  echo "  (nvidia-smi not available — continuing)"
echo ""

# ── activate env ──
echo "[1/3] Activating conda environment ..."
module load anaconda3
conda activate /ocean/projects/cis260075p/bangarug/conda_envs/rhino-agent

# ── verify deps ──
echo "[2/3] Checking dependencies ..."
pip install --quiet fastapi uvicorn 2>/dev/null || pip install fastapi uvicorn

# ── launch ──
NODE=$(hostname -s)
echo ""
echo "[3/3] Starting server on ${NODE}:${PORT}"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Open a tunnel from your LOCAL machine:                     ║"
echo "║                                                             ║"
echo "║    ssh -L ${PORT}:${NODE}:${PORT} bangarug@bridges2.psc.edu       ║"
echo "║                                                             ║"
echo "║  Then open brickagent_ui/index.html in your browser.        ║"
echo "║  Run rhino_brick_client.py in Rhino on your local machine.  ║"
echo "║                                                             ║"
echo "║  Server URL  : http://localhost:${PORT}                          ║"
echo "║  Rhino bridge: http://localhost:8081  (local only)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python serve_brickagent.py \
    --adapter        "${ADAPTER_PATH}" \
    --adapter-stage1 "${ADAPTER_STAGE1}" \
    --adapter-stage5 "${ADAPTER_STAGE5}" \
    --planner        "${PLANNER_MODEL}" \
    --port           "${PORT}"
