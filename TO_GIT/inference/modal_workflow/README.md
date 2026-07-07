# BrickAgent on Modal

This folder adds a separate Modal deployment path for BrickAgent without changing the PSC workflow.

It assumes:

- you will download the LoRA adapter checkpoint locally from PSC (or use the `checkpoints/physics_reasoning` or `checkpoints/no_reasoning` folder already in this repo)
- you want to use `physics_reasoning` for the main agent (swap to `no_reasoning` the same way if you want the pre-reasoning baseline instead)
- Rhino still runs locally via `rhino_brick_client.py` on `http://localhost:8081`
- only the model server moves to Modal

## Files

- `modal_app.py`: deploys the existing FastAPI BrickAgent server on Modal
- `upload_adapter.py`: uploads your local stage checkpoint directory into a Modal Volume
- `ui/index.html`: local browser UI for the Modal endpoint
- `ui/server-config.js`: stores the Modal URL locally
- `ui/fetch-shim.js`: rewrites the old `localhost:8080` server calls to your Modal URL

## 1. Install and authenticate Modal

```bash
pip install -r modal_workflow/requirements.txt
modal setup
```

## 2. Prepare your local adapter folders

You need the main adapter directory on your machine, for example:

```text
physics_reasoning/
  adapter_config.json
  adapter_model.safetensors
```

If the checkpoint is sharded, that is also fine:

```text
stage_n_checkpoint/
  adapter_config.json
  adapter_model-00001-of-00002.safetensors
  adapter_model-00002-of-00002.safetensors
```

## 3. Upload the adapter to a Modal Volume

```bash
python modal_workflow/upload_adapter.py --local-path /path/to/physics_reasoning --remote-name physics_reasoning
```

By default, the uploader sends only the LoRA files needed by the server:

- `adapter_config.json`
- `adapter_model*.safetensors`

This avoids uploading training artifacts like `optimizer.pt`, which are not used by the server.

If you truly need the whole directory, add `--all-files`.

By default this writes into:

```text
Volume name: brickagent-model-store
Remote paths:
  /adapters/physics_reasoning
```

If you want a different Volume name:

```bash
python modal_workflow/upload_adapter.py \
  --local-path /path/to/physics_reasoning \
  --volume my-brickagent-volume
```

## 4. Download the two base models into the same Volume

`modal_app.py` uses the same planner/executor pair as the PSC server:

- planner: `Qwen/Qwen3.5-27B`
- executor base: `unsloth/Qwen2.5-32B-Instruct`

Run:

```bash
modal run modal_workflow/modal_app.py::download_base_models
```

If you changed the Volume name, export it first:

```bash
export BRICKAGENT_MODAL_VOLUME=my-brickagent-volume
modal run modal_workflow/modal_app.py::download_base_models
```

## 5. Deploy the Modal server

The default GPU setting is:

```text
BRICKAGENT_MODAL_GPU=A100-80GB:2
```

That matches the current server design, which loads the planner and executor onto separate GPUs when two are available.

Deploy:

```bash
modal deploy modal_workflow/modal_app.py
```

Optional overrides:

```bash
export BRICKAGENT_MODAL_VOLUME=my-brickagent-volume
export BRICKAGENT_ADAPTER_NAME=physics_reasoning
export BRICKAGENT_MODAL_GPU=H100:2
export BRICKAGENT_MODAL_MIN_CONTAINERS=0
export BRICKAGENT_MODAL_SCALEDOWN_WINDOW=900
modal deploy modal_workflow/modal_app.py
```

After deploy, Modal will print the public HTTPS URL for the `serve` web endpoint.

## 6. Keep Rhino local

Run the existing local Rhino bridge exactly as before:

```text
rhino_brick_client.py
```

It must still expose:

- `http://localhost:8081/health`
- `http://localhost:8081/viewport`
- `http://localhost:8081/place`

## 7. Open the separate local UI

Open this file locally:

```text
modal_workflow/ui/index.html?server=https://YOUR-MODAL-ENDPOINT.modal.run
```

Example:

```text
file:///.../brickagent_ui/modal_workflow/ui/index.html?server=https://your-workspace--serve.modal.run
```

The UI stores the `server` query parameter in local storage, so after the first launch you can usually reopen the same file without the query string.

## Notes

- The Modal server reuses `serve_brickagent.py`; it does not modify the existing PSC files.
- The main adapter must live at `/adapters/physics_reasoning` unless you override `BRICKAGENT_ADAPTER_NAME`.
- The FastAPI app still exposes `/health`, `/plan`, `/execute`, `/execute-part`, `/execute-fast`, and `/inspect-part`.
- Fast is a separate UI toggle that can be combined with Executor, Work Along, or Inspect.
- In direct Executor mode, Fast uses `/execute-fast`; in plan-driven modes it uses the simplified prompt inside the normal execution endpoints.
- If you redeploy to a new URL, reopen `ui/index.html` with a new `?server=...` value.
