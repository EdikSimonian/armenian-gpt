#!/usr/bin/env bash
# Production launcher: starts the training container detached with restart
# policy that cooperates with /etc/systemd/system/gpu-thermal-watchdog.service.
#
# When the watchdog kills the docker daemon on overheat, restart=unless-stopped
# brings this container back automatically after cooldown, and launch.sh's
# auto-resume picks up from the most recent checkpoint.
#
# Usage:
#   ./docker/run-detached.sh         # start (or restart) the training run
#   docker logs -f armgpt-train      # follow output
#   docker stop armgpt-train         # graceful stop (sends SIGTERM, 120s grace)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Wipe any prior container instance so --name is reusable.
docker rm -f armgpt-train 2>/dev/null || true

docker run -d \
    --name armgpt-train \
    --gpus all \
    --restart=unless-stopped \
    --stop-timeout 120 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    -v "$REPO_DIR":/workspace/armgpt \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v /mnt/nas-checkpoints-staging:/mnt/nas-checkpoints \
    -w /workspace/armgpt \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    armgpt:giant \
    bash ./docker/launch.sh

echo "Started armgpt-train (detached). Follow with: docker logs -f armgpt-train"
