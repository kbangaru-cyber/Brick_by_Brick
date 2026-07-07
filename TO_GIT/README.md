# BrickAgent

BrickAgent is an LLM agent that plans and builds LEGO-style brick structures. This repo is scoped to two checkpoints from a larger multi-stage training curriculum:

- **`no_reasoning`** — the Stage 1 adapter (structure generator), trained before any chain-of-thought or physics reasoning was added to the curriculum.
- **`physics_reasoning`** — the Stage 6 "full" adapter (Model A), trained to reason about physics/stability (`<think>+<plan>`) before building (`<build>+<review>`).

Both are LoRA adapters (r=64, alpha=128) on top of `unsloth/Qwen2.5-32B-Instruct`.

## Repository layout

```
checkpoints/
  no_reasoning/        LoRA adapter — Stage 1
  physics_reasoning/   LoRA adapter — Stage 6 "full"
model_training/         the two SFT scripts that produced each checkpoint
inference/              the server that loads and switches between them, plus the Rhino bridge for live physics-validated building
dataset_loading/        utilities to inspect the training data each model was trained on
```

## checkpoints/

Only the small files (`adapter_config.json`, `tokenizer.json`, `chat_template.jinja`, `README.md`) are meant to be tracked in git — `.gitignore` excludes `*.safetensors`, so each adapter's actual weight file (~2GB) stays local-only. Drop the real `adapter_model*.safetensors` into the matching folder (or fetch it via `inference/modal_workflow/upload_adapter.py`) on any machine that clones this repo.

Don't confuse `no_reasoning` (Stage 1, the curriculum's starting point, before reasoning existed at all) with the separate Stage 6 **noreason** ablation control that this repo does not include — same word, different checkpoint from a different experiment.

## model_training/

| File | Produces | Notes |
|---|---|---|
| `train_no_reasoning.py` | `checkpoints/no_reasoning` | Fine-tunes `unsloth/Qwen2.5-32B-Instruct` from scratch on `stage1_structure_generator.jsonl`. Launch via `run_train_no_reasoning.sh` (PSC sbatch). |
| `train_physics_reasoning.py` | `checkpoints/physics_reasoning` | Continues from a Stage 5 adapter (not included in this repo) and trains on `stage6_full.jsonl` — samples with full `<think>+<plan>+<build>+<review>` reasoning. Launch via `run_train_physics_reasoning.sh` (PSC sbatch). Uses `ds_zero2.json` for multi-GPU DeepSpeed. |

Both scripts save to `checkpoints/<name>` by default, resolved relative to the script's own location — override with the `BRICKAGENT_CHECKPOINTS_DIR` env var (or `BRICKAGENT_NO_REASONING_DIR` / `BRICKAGENT_PHYSICS_REASONING_DIR` for just one). `setup_psc_env.sh` is the one-time PSC environment setup (installs deps, compiles CUDA kernels) needed before either script runs on a PSC GPU node.

`train_physics_reasoning.py` warm-starts from a Stage 5 checkpoint that isn't part of this repo — you need the full upstream curriculum to reproduce it from scratch. `checkpoints/physics_reasoning` here is the *output*, for inference and further fine-tuning, not something you can retrain from zero with only what's in this repo.

**Security note:** `run_train_no_reasoning.sh` originally had a live Hugging Face token hardcoded (`export HF_TOKEN="hf_..."`). I redacted it to `${HF_TOKEN:?set your own...}` before copying — the original script (`Training_scripts/run_train.sh` in your working folder) still has the real token in plaintext. Consider rotating that token since it's been sitting in a plaintext file.

## inference/

`serve_brickagent.py` is a two-agent server: a planner (`Qwen/Qwen3.5-27B`, VL-capable) negotiates the build with a human designer, then an executor (`Qwen2.5-32B-Instruct` + LoRA) streams brick placements. It loads **both** checkpoints at once as swappable adapters — `--adapter-stage1` (no_reasoning, used by Fast/Builder mode) and `--adapter-stage5` (physics_reasoning) — plus a `--adapter` default slot that also points at `physics_reasoning` here (the original PSC deployment used a separate Stage 3 checkpoint in that slot that isn't part of this repo).

- `rhino_brick_client.py` — bridge that runs inside Rhino 8, exposes viewport captures and brick placement over `localhost:8081`. This is what makes placement/stability checks happen in a real 3D scene rather than just in the model's own reasoning tokens. Sample captures are under `sample_captures/`.
- `run_server.sh` — PSC launch script; all three adapter paths default to `checkpoints/` in this repo (override with `BRICKAGENT_CHECKPOINTS_DIR`, or `ADAPTER_PATH` / `ADAPTER_STAGE1` / `ADAPTER_STAGE5` individually).
- `app.js` / `index.html` — the browser UI that talks to the server and displays Rhino's viewport captures to the designer.
- `modal_workflow/` — an alternative deployment path that runs a single-adapter version of the server on Modal instead of PSC, with Rhino still running locally. `ADAPTER_NAME` defaults to `physics_reasoning`; pass `--remote-name no_reasoning` to `upload_adapter.py` to use the other checkpoint instead. See its own README for the full setup (upload adapter → download base models → deploy).

## dataset_loading/

- `viewer.py`, `show_sample.py` — generic JSONL viewers, work on either model's training data.
- `view_physics_reasoning_samples.py` (was `view_samples.py`) — prints one sample per prompt tier from a `stage6_full`-style JSONL.
- `inspect_physics_reasoning_data.py` (was `inspect_stage6.py`) — summarizes/compares `stage6_full` vs `stage6_noreason` shards (useful context even though this repo only ships the `full`/physics_reasoning checkpoint).

## What's not included

- The generated training datasets themselves (`stage1_structure_generator.jsonl`, `stage6_full.jsonl`, etc.) — multi-GB JSONL files, regenerate via your own data pipeline.
- Every other stage in the curriculum (2 through 5, and the Stage 6 noreason ablation) and the dataset-creation scripts that produced them — out of scope for this repo, which is intentionally narrowed to just `no_reasoning` and `physics_reasoning`.
- The evaluation/scoring pipeline and the notebook/batch inference paths (Colab notebook, walkthrough notebook, `eval_generate_stage6.py`) — say if you want these added back; they were left out to keep this repo focused on training + the live server + data inspection.
