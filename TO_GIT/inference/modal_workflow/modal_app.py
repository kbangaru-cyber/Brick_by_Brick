from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


APP_NAME = os.environ.get("BRICKAGENT_MODAL_APP_NAME", "brickagent-modal")
VOLUME_NAME = os.environ.get("BRICKAGENT_MODAL_VOLUME", "brickagent-model-store")
VOLUME_MOUNT = "/models"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf-cache"
HF_HUB_CACHE_DIR = HF_CACHE_DIR
ADAPTER_NAME = os.environ.get("BRICKAGENT_ADAPTER_NAME", "physics_reasoning")
ADAPTER_DIR = f"{VOLUME_MOUNT}/adapters/{ADAPTER_NAME}"
# Single-agent architecture: only Qwen2.5-32B-Instruct + one LoRA adapter
# (physics_reasoning or no_reasoning, uploaded via upload_adapter.py).
# The old Qwen3.5-27B planner is no longer loaded.
EXECUTOR_MODEL = os.environ.get("BRICKAGENT_EXECUTOR_MODEL", "unsloth/Qwen2.5-32B-Instruct")
GPU_CONFIG = os.environ.get("BRICKAGENT_MODAL_GPU", "A100-80GB:1")
MIN_CONTAINERS = int(os.environ.get("BRICKAGENT_MODAL_MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(os.environ.get("BRICKAGENT_MODAL_SCALEDOWN_WINDOW", "900"))

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SERVER_FILE = _REPO_ROOT / "serve_brickagent.py"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "fastapi>=0.115.0",
        "uvicorn>=0.34.0",
        "transformers>=4.51.0",
        "accelerate>=1.5.0",
        "peft>=0.15.0",
        "safetensors>=0.5.0",
        "huggingface_hub>=0.30.0",
    )
    .add_local_file(_SERVER_FILE, remote_path="/app/serve_brickagent.py")
)

_server_module = None


def _configure_hf_env() -> None:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")


def _import_server_module():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    import serve_brickagent as server_module

    return server_module


def _ensure_adapter_exists(adapter_dir: str, adapter_name: str) -> None:
    adapter_config = Path(adapter_dir) / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            "Adapter not found at "
            f"{adapter_config}. Upload your local checkpoint directory to "
            f"{VOLUME_NAME} under /adapters/{adapter_name} first."
        )


def _load_server():
    global _server_module
    if _server_module is not None:
        return _server_module

    model_volume.reload()
    _configure_hf_env()
    os.environ["HF_HUB_OFFLINE"] = "1"
    _ensure_adapter_exists(ADAPTER_DIR, ADAPTER_NAME)

    server_module = _import_server_module()
    server_module.EXECUTOR_BASE = EXECUTOR_MODEL

    # Single-agent: load executor only. Alias planner refs to the same model
    # so any legacy code path in serve_brickagent.py (e.g. /plan) still works.
    server_module._exec_model, server_module._exec_tok = server_module.load_executor(
        ADAPTER_DIR
    )
    server_module._planner_model = server_module._exec_model
    server_module._planner_tok = server_module._exec_tok

    _server_module = server_module
    return _server_module


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: model_volume},
)
def download_base_models() -> dict[str, str]:
    from huggingface_hub import snapshot_download

    _configure_hf_env()

    local_paths = {}
    for model_id in (EXECUTOR_MODEL,):
        local_paths[model_id] = snapshot_download(
            repo_id=model_id,
            cache_dir=HF_HUB_CACHE_DIR,
        )

    model_volume.commit()
    return local_paths


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=60 * 60,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={VOLUME_MOUNT: model_volume},
)
@modal.asgi_app()
def serve():
    server_module = _load_server()
    return server_module.app
