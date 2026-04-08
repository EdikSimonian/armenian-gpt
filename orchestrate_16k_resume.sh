#!/bin/bash
# Orchestrates the stop-at-16k → cleanup → resume → re-enable-auto-stop sequence.
#
# 1. Wait for checkpoints/step_16000.pt to appear on edisimon/armgpt
# 2. SIGTERM the running train.py (PID stored below) — graceful shutdown saves+uploads
# 3. Wait for the train.py process to fully exit
# 4. Delete any duplicate checkpoint locally + on HF (any step > 16000)
# 5. Resume training with --resume_from checkpoints/step_16000.pt
# 6. Re-launch stop_after_training.sh (auto pod-stop monitor)
#
# Usage: nohup bash orchestrate_16k_resume.sh > orchestrate_16k_resume.log 2>&1 &

set -u
TRAIN_PID=74245
HF_REPO="edisimon/armgpt"
CKPT_DIR=/workspace/armenian-gpt/checkpoints
REPO_DIR=/workspace/armenian-gpt
TARGET="checkpoints/step_16000.pt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

cd "$REPO_DIR"

# ---- 1. Wait for step_16000.pt on HF ----
log "Phase 1: waiting for $TARGET to appear on $HF_REPO"
while true; do
    found=$(python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('$HF_REPO', repo_type='model')
print('YES' if '$TARGET' in files else 'NO')
" 2>/dev/null)
    if [ "$found" = "YES" ]; then
        log "  $TARGET confirmed on HF"
        break
    fi
    sleep 30
done

# ---- 2. SIGTERM train.py ----
log "Phase 2: sending SIGTERM to train.py PID $TRAIN_PID"
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID"
    log "  SIGTERM sent"
else
    log "  PID $TRAIN_PID already exited; continuing"
fi

# ---- 3. Wait for train.py to exit ----
log "Phase 3: waiting for train.py to exit (graceful shutdown saves + uploads final)"
for i in $(seq 1 60); do
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        sleep 5
    else
        log "  train.py exited after ~$((i*5))s"
        break
    fi
done
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "  WARNING: train.py still alive after 5min, sending SIGKILL"
    kill -9 "$TRAIN_PID" || true
    sleep 5
fi

# Give upload threads a beat to finish flushing
log "  Sleeping 30s to let any in-flight HF upload threads complete"
sleep 30

# ---- 4. Delete any checkpoint > step_16000 (the duplicate from graceful shutdown) ----
log "Phase 4: scanning for duplicate checkpoints (step > 16000)"
DUPES=()
for f in "$CKPT_DIR"/step_*.pt; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    step=$(echo "$base" | grep -oP '\d+')
    if [ "$step" -gt 16000 ] 2>/dev/null; then
        DUPES+=("$base")
    fi
done

if [ "${#DUPES[@]}" -eq 0 ]; then
    log "  no duplicate checkpoints found"
else
    for base in "${DUPES[@]}"; do
        log "  deleting local: $CKPT_DIR/$base"
        rm -f "$CKPT_DIR/$base"
        # Also remove from .uploaded_checkpoints so upload_watcher doesn't get confused
        if [ -f "$REPO_DIR/.uploaded_checkpoints" ]; then
            grep -vF "$base" "$REPO_DIR/.uploaded_checkpoints" > "$REPO_DIR/.uploaded_checkpoints.tmp" || true
            mv "$REPO_DIR/.uploaded_checkpoints.tmp" "$REPO_DIR/.uploaded_checkpoints"
        fi
        log "  deleting from HF: checkpoints/$base"
        python -c "
from huggingface_hub import HfApi
try:
    HfApi().delete_file(
        path_in_repo='checkpoints/$base',
        repo_id='$HF_REPO',
        repo_type='model',
        commit_message='Remove duplicate post-shutdown checkpoint',
    )
    print('  OK')
except Exception as e:
    print(f'  delete failed: {e}')
" 2>&1 | sed 's/^/    /'
    done
fi

# ---- 5. Resume training from step_16000.pt ----
log "Phase 5: resuming training from checkpoints/step_16000.pt"
cd "$REPO_DIR"
nohup python -u train.py \
    --preset xlarge \
    --tokenizer bpe \
    --hf_repo "$HF_REPO" \
    --resume_from checkpoints/step_16000.pt \
    --save_interval 1000 \
    >> train.log 2>&1 &
NEW_PID=$!
disown "$NEW_PID" 2>/dev/null || true
log "  new training PID: $NEW_PID"
sleep 10

if kill -0 "$NEW_PID" 2>/dev/null; then
    log "  training is running"
else
    log "  ERROR: training failed to start; check train.log"
    exit 1
fi

# ---- 6. Re-launch stop_after_training.sh (auto pod-stop monitor) ----
log "Phase 6: re-enabling stop_after_training.sh (auto pod-stop monitor)"
nohup bash "$REPO_DIR/stop_after_training.sh" > "$REPO_DIR/stop_monitor.log" 2>&1 &
SAT_PID=$!
disown "$SAT_PID" 2>/dev/null || true
log "  stop_after_training.sh PID: $SAT_PID"

log "Done. Training resumed from step 16000 with save_interval=1000."
