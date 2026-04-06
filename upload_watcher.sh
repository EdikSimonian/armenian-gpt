#!/bin/bash
# Watches for new checkpoints and uploads them to HF with retries
# Usage: nohup bash upload_watcher.sh > upload_watcher.log 2>&1 &

HF_REPO="edisimon/armgpt"
CKPT_DIR="/workspace/armenian-gpt/checkpoints"
LOG="/workspace/armenian-gpt/upload_watcher.log"
UPLOADED_FILE="/workspace/armenian-gpt/.uploaded_checkpoints"

touch "$UPLOADED_FILE"

echo "$(date): Upload watcher started. Monitoring $CKPT_DIR for new checkpoints."

while true; do
    for ckpt in "$CKPT_DIR"/step_*.pt; do
        [ -f "$ckpt" ] || continue
        basename=$(basename "$ckpt")
        # Skip if already uploaded
        grep -qF "$basename" "$UPLOADED_FILE" && continue
        # Extract step number, only upload every 1000 steps
        step_num=$(echo "$basename" | grep -oP '\d+')
        if [ $((step_num % 1000)) -ne 0 ]; then
            continue
        fi
        # Wait a bit to make sure file is fully written
        sleep 10
        echo "$(date): Uploading $basename..."
        for attempt in 1 2 3; do
            python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$ckpt',
    path_in_repo='checkpoints/$basename',
    repo_id='$HF_REPO',
    repo_type='model',
)
print('OK')
" 2>&1
            if [ $? -eq 0 ]; then
                echo "$(date): $basename uploaded successfully"
                echo "$basename" >> "$UPLOADED_FILE"
                break
            else
                echo "$(date): $basename upload attempt $attempt failed, retrying in 60s..."
                sleep 60
            fi
        done
    done
    sleep 30
done
