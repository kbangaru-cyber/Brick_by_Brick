from __future__ import annotations

import argparse
import time
from pathlib import Path

import modal


DEFAULT_VOLUME = "brickagent-model-store"
DEFAULT_REMOTE_ROOT = "/adapters"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a local BrickAgent LoRA checkpoint directory into a Modal Volume."
    )
    parser.add_argument(
        "--local-path",
        required=True,
        help="Local checkpoint directory, e.g. ./checkpoint-3600",
    )
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help=f"Modal Volume name (default: {DEFAULT_VOLUME})",
    )
    parser.add_argument(
        "--remote-name",
        default="checkpoint-3600",
        help="Folder name to use inside /adapters on the Modal Volume",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the destination adapter directory",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Upload the full checkpoint directory instead of only runtime files",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count per file for transient upload failures",
    )
    args = parser.parse_args()

    src = Path(args.local_path).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Local adapter directory not found: {src}")

    adapter_config = src / "adapter_config.json"
    if not adapter_config.exists():
        raise SystemExit(
            f"{src} does not look like a LoRA checkpoint directory; "
            "missing adapter_config.json"
        )

    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    remote_path = f"{DEFAULT_REMOTE_ROOT}/{args.remote_name}"
    if args.all_files:
        files_to_upload = sorted(p for p in src.iterdir() if p.is_file())
    else:
        files_to_upload = [src / "adapter_config.json"]
        files_to_upload.extend(sorted(src.glob("adapter_model*.safetensors")))

    if not files_to_upload:
        raise SystemExit(
            f"No files selected for upload from {src}. "
            "Use --all-files if you intend to upload the full checkpoint directory."
        )
    if not (src / "adapter_config.json").exists():
        raise SystemExit(f"Missing required file: {src / 'adapter_config.json'}")
    if not list(src.glob("adapter_model*.safetensors")):
        raise SystemExit(f"Missing required adapter weights: {src / 'adapter_model*.safetensors'}")

    print(f"Uploading to volume '{args.volume}' at {remote_path}")
    for file_path in files_to_upload:
        remote_file = f"{remote_path}/{file_path.name}"
        size_gb = file_path.stat().st_size / (1024 ** 3)
        print(f"  -> {file_path.name} ({size_gb:.2f} GB)")

        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                with volume.batch_upload(force=args.force) as batch:
                    batch.put_file(str(file_path), remote_file)
                print(f"     uploaded on attempt {attempt}")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"     attempt {attempt} failed: {exc}")
                if attempt < args.retries:
                    time.sleep(min(5 * attempt, 15))

        if last_error is not None:
            raise SystemExit(
                f"Upload failed for {file_path.name} after {args.retries} attempts: {last_error}"
            )

    print(f"Uploaded selected files from {src} -> volume '{args.volume}' at {remote_path}")
    print(
        "Next: run `modal run modal_workflow/modal_app.py::download_base_models` "
        "then `modal deploy modal_workflow/modal_app.py`."
    )


if __name__ == "__main__":
    main()
