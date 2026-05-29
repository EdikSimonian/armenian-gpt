#!/usr/bin/env bash
# Drop into the armgpt training container with the right mounts.
# Code is mounted live from the host so edits don't need a rebuild.
#
# Usage:
#   ./docker/run.sh                 # interactive bash
#   ./docker/run.sh python --version
#   ./docker/run.sh ./docker/launch.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

# --ipc=host + memlock/stack ulimits + shm-size are NGC's recommended runtime
# flags for DDP and torch.compile graph capture. --gpus all exposes the RTX
# PRO 6000 via NVIDIA Container Toolkit.
exec docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    -v "$REPO_DIR":/workspace/armgpt \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace/armgpt \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    --name armgpt-train \
    armgpt:giant \
    "$@"
