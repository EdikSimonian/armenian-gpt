#!/bin/bash
# Waits for step_16000.pt to appear on the HF model repo, then SIGTERMs train.py
# so its graceful shutdown handler saves + uploads a final checkpoint and exits.
#
# Usage: nohup bash stop_at_16000.sh > stop_at_16000.log 2>&1 &

TRAIN_PID=74245
HF_REPO="edisimon/armgpt"
TARGET="checkpoints/step_16000.pt"
LOG=/workspace/armenian-gpt/stop_at_16000.log

echo "$(date): waiting for $TARGET on $HF_REPO; will SIGTERM PID $TRAIN_PID once present"

while true; do
    found=$(python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('$HF_REPO', repo_type='model')
print('YES' if '$TARGET' in files else 'NO')
" 2>/dev/null)
    if [ "$found" = "YES" ]; then
        echo "$(date): $TARGET confirmed on HF — sending SIGTERM to PID $TRAIN_PID"
        if kill -0 $TRAIN_PID 2>/dev/null; then
            kill -TERM $TRAIN_PID
            echo "$(date): SIGTERM sent. Training will graceful-shutdown (save + upload + exit)."
        else
            echo "$(date): PID $TRAIN_PID no longer running, nothing to do."
        fi
        exit 0
    fi
    sleep 30
done
