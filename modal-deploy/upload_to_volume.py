"""
One-time / on-demand: download the latest checkpoint + tokenizer from
edisimon/armgpt and push them into the Modal Volume named 'armgpt-checkpoints'.

Re-run this whenever you want the deployed Modal endpoint to serve a newer
checkpoint. Modal containers will pick it up on the next cold start.

Usage:
    python upload_to_volume.py            # uploads latest step_*.pt
    python upload_to_volume.py step_30000 # uploads a specific step
"""

import re
import sys

import modal
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "edisimon/armgpt"
VOLUME_NAME = "armgpt-checkpoints"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def latest_checkpoint_name() -> str:
    files = HfApi().list_repo_files(MODEL_REPO, repo_type="model")
    pattern = re.compile(r"^checkpoints/step_(\d+)\.pt$")
    matches = [(int(pattern.match(f).group(1)), f) for f in files if pattern.match(f)]
    if not matches:
        raise RuntimeError(f"No step_*.pt files found in {MODEL_REPO}")
    matches.sort()
    return matches[-1][1]


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if not target.startswith("checkpoints/"):
            target = f"checkpoints/{target}"
        if not target.endswith(".pt"):
            target += ".pt"
    else:
        target = latest_checkpoint_name()
    print(f"Target checkpoint on HF: {target}")

    print("Downloading checkpoint from HF (cached)...")
    local_ckpt = hf_hub_download(MODEL_REPO, target, repo_type="model")
    print(f"  -> {local_ckpt}")

    print("Downloading tokenizer from HF (cached)...")
    local_tok = hf_hub_download(MODEL_REPO, "data/tokenizer.json", repo_type="model")
    print(f"  -> {local_tok}")

    print(f"Uploading to Modal Volume '{VOLUME_NAME}'...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(local_ckpt, "checkpoint.pt")
        batch.put_file(local_tok, "tokenizer.json")
    print("Done. Modal endpoint will pick up the new checkpoint on next cold start.")
    print("To force a hot reload now, run: modal app stop armgpt && modal deploy modal_app.py")


if __name__ == "__main__":
    main()
