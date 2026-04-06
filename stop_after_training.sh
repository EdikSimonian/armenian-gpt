#!/bin/bash
# Monitors training and stops the pod when it completes + uploads finish
# Usage: nohup bash stop_after_training.sh &

POD_ID="rmxoq3k3dmfxs7"
HF_REPO="edisimon/armgpt"
CKPT_DIR="/workspace/armenian-gpt/checkpoints"

echo "$(date): Monitoring training. Will stop pod $POD_ID when training completes and uploads finish."

while true; do
    # Check if training process is still running
    if ! pgrep -f "train.py.*xlarge" > /dev/null 2>&1; then
        echo "$(date): Training process not found. Checking if it completed..."

        if [ -f "$CKPT_DIR/final.pt" ]; then
            echo "$(date): Training complete! final.pt found."

            # Wait for any HF upload threads to finish (train.py uploads in background threads)
            echo "$(date): Waiting 120s for any in-flight HF uploads..."
            sleep 120

            # Verify final checkpoint is on HF
            echo "$(date): Verifying final.pt is uploaded to HF..."
            python -c "
from huggingface_hub import HfApi
api = HfApi()
files = [f.rfilename for f in api.list_repo_tree('$HF_REPO', repo_type='model')]
if 'checkpoints/final.pt' in files:
    print('final.pt confirmed on HF')
else:
    print('final.pt NOT on HF, uploading now...')
    api.upload_file(
        path_or_fileobj='$CKPT_DIR/final.pt',
        path_in_repo='checkpoints/final.pt',
        repo_id='$HF_REPO',
        repo_type='model',
    )
    print('Upload complete')
" 2>&1 | while read line; do echo "$(date): $line"; done

            # Also upload metrics
            if [ -f "$CKPT_DIR/metrics.json" ]; then
                echo "$(date): Uploading metrics.json..."
                python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='$CKPT_DIR/metrics.json',
    path_in_repo='checkpoints/metrics.json',
    repo_id='$HF_REPO',
    repo_type='model',
)
print('Done')
" 2>&1 | while read line; do echo "$(date): $line"; done
            fi

            echo "$(date): All uploads confirmed. Stopping pod $POD_ID..."
            runpodctl stop pod "$POD_ID"
            echo "$(date): Pod stop command sent."
            exit 0
        else
            echo "$(date): WARNING — training process died without final.pt. Pod NOT stopped."
            echo "$(date): Check train.log for errors."
            exit 1
        fi
    fi
    sleep 60
done
