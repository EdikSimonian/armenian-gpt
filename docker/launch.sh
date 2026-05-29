#!/usr/bin/env bash
# Run INSIDE the container (./docker/run-detached.sh handles that for the
# overnight detached case) to execute the giant-preset pipeline.
#
# The dataset at edisimon/armenian-clean-text already ships pre-tokenized
# BPE bins under tokenized/, so we skip corpus assembly + BPE training
# entirely and go straight from "fresh container" to training in ~10 min.
#
# Each step is idempotent — rerunning skips work already done. Critical for
# the restart=unless-stopped + gpu-thermal-watchdog cycle.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_DIR=data
TRAIN_BIN="$DATA_DIR/train_bpe.bin"
VAL_BIN="$DATA_DIR/val_bpe.bin"
TOKENIZER="$DATA_DIR/tokenizer_bpe.json"

mkdir -p "$DATA_DIR"

# Step 1: hydrate data from HF — tokenized bins + tokenizer.
if [ ! -f "$TRAIN_BIN" ] || [ ! -f "$VAL_BIN" ] || [ ! -f "$TOKENIZER" ]; then
    echo "[1/2] Hydrating tokenized data from edisimon/armenian-clean-text..."
    python <<'PY'
import os, shutil
import zstandard as zstd
from huggingface_hub import hf_hub_download

REPO = "edisimon/armenian-clean-text"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

# Plain JSON tokenizer (small).
src = hf_hub_download(REPO, "tokenized/tokenizer_bpe.json", repo_type="dataset")
shutil.copyfile(src, os.path.join(OUT, "tokenizer_bpe.json"))
print("  tokenizer_bpe.json ready")

# Compressed bins → decompress into data/.
for name in ("train_bpe.bin", "val_bpe.bin"):
    out_path = os.path.join(OUT, name)
    if os.path.exists(out_path):
        print(f"  {name} already present")
        continue
    src = hf_hub_download(REPO, f"tokenized/{name}.zst", repo_type="dataset")
    print(f"  decompressing {src} -> {out_path}")
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as fi, open(out_path, "wb") as fo:
        dctx.copy_stream(fi, fo)
    print(f"  {name} ready ({os.path.getsize(out_path)/1e9:.1f} GB)")
PY
else
    echo "[1/2] Tokenized data already present, skipping hydration"
fi

# Step 2: train. Resume preference: latest local > latest NAS-staged > scratch.
# Local is mounted at checkpoints/ from the host repo dir; the NAS rsync
# watcher mirrors checkpoints to /mnt/nas-checkpoints (via symlinks back to
# local OR real files if local has been purged).
LATEST_LOCAL=$(ls -t checkpoints/step_*.pt 2>/dev/null | head -1 || true)
LATEST_NAS=$(ls -t /mnt/nas-checkpoints/step_*.pt 2>/dev/null | head -1 || true)
RESUME_FROM=""
if [ -n "$LATEST_LOCAL" ]; then
    RESUME_FROM="$LATEST_LOCAL"
elif [ -n "$LATEST_NAS" ]; then
    cp "$LATEST_NAS" checkpoints/
    RESUME_FROM="checkpoints/$(basename "$LATEST_NAS")"
fi

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    echo "[2/2] Resuming from $RESUME_FROM"
    RESUME_ARGS=(--resume_from "$RESUME_FROM")
else
    echo "[2/2] Starting giant-preset training from scratch..."
fi
exec python 4_train.py --preset giant --tokenizer bpe "${RESUME_ARGS[@]}" "$@"
