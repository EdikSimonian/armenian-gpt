# ArmGPT giant-preset training on RTX PRO 6000 (Blackwell, 96 GB)

This is the runbook for kicking off training on `gpu-pc`. Everything is already
staged at `~/armgpt/` and the NGC PyTorch 25.01 image is pre-pulled. Nothing
auto-starts — you control each step.

## What's already done

- Repo code rsync'd to `~/armgpt/`
- `docker/Dockerfile`, `docker/run.sh`, `docker/launch.sh` in place
- `nvcr.io/nvidia/pytorch:25.01-py3` pulled (~12 GB, the heavy lift)
- `armgpt:giant` image built on top with sentencepiece + huggingface_hub +
  hf_transfer + datasets + zstandard + mwxml
- Docker NVIDIA runtime confirmed: CDI mode, driver 595.71.05

## One-time auth setup (BEFORE first run)

The pre-cleaned corpus `edisimon/armenian-clean-text` is a **private** HF dataset.
You need to log in inside the container the first time so the token persists
in `~/.cache/huggingface/` (which is mounted into the container).

```bash
cd ~/armgpt
./docker/run.sh huggingface-cli login
# Paste a token with read access to edisimon/armenian-clean-text
# (and write access to edisimon/armgpt if you want checkpoint uploads).
```

Token persists across container runs because `~/.cache/huggingface` is mounted.

## Start training (the actual run)

Option A — full pipeline in one command (download → tokenize → train):

```bash
cd ~/armgpt
tmux new -s armgpt    # so you can detach and let it run
./docker/run.sh ./docker/launch.sh
```

To also push checkpoints to HF as they're saved:

```bash
./docker/run.sh ./docker/launch.sh --hf_repo edisimon/armgpt
```

Option B — step through manually (good for first run, lets you inspect each stage):

```bash
cd ~/armgpt
./docker/run.sh                  # drops you into a bash shell in the container

# Inside the container:
huggingface-cli download edisimon/armenian-clean-text clean_text.txt \
    --repo-type dataset --local-dir data/text/train
python 3_tokenize.py --tokenizer bpe
python 4_train.py --preset giant --tokenizer bpe --hf_repo edisimon/armgpt
```

## Resume from a checkpoint

```bash
./docker/run.sh python 4_train.py --preset giant --tokenizer bpe \
    --resume_from checkpoints/step_NNNNN.pt --hf_repo edisimon/armgpt
```

## Expected timing & resource use

- **Corpus download** (~31 GB from HF with hf_transfer): ~5–15 min on a 100+
  Mbps link
- **BPE tokenization**: ~30–60 min (CPU-bound, uses all cores)
- **Training (giant preset, 122 000 steps)**: ~80–110 h on the RTX PRO 6000
  - Bandwidth-bound (1.6 TB/s vs 3.35 TB/s on H100 SXM), so wall-clock is
    roughly 1.4–1.6× an H100 80GB at the same batch size
  - VRAM use: ~30–40 GB at batch 8 × ctx 2048 with BF16 AdamW (no need for
    8-bit Adam given the 96 GB headroom)
  - Disk: ~25 GB for tokenized data + ~5 GB per saved checkpoint
- **Power**: card pulls up to 600 W under load — watch your PSU and case
  airflow on multi-day runs

## Monitor

```bash
# GPU + memory in real time
ssh gpu-pc 'watch -n 2 nvidia-smi'

# Training stdout (when running inside tmux)
ssh gpu-pc 'tmux attach -t armgpt'

# Checkpoint sizes
ssh gpu-pc 'ls -lah ~/armgpt/checkpoints/'
```

## Stop / pause cleanly

Inside the tmux session, `Ctrl+C` lets train.py finish the current step and
write a final checkpoint. The next `4_train.py --resume_from` will pick up
exactly where it left off.

To kill the container entirely: `docker kill armgpt-train` from another shell.

## If something breaks

- Build fails on Blackwell sm_120 → confirm with `./docker/run.sh python -c
  "import torch; print(torch.cuda.get_device_capability(0))"` — should print
  `(12, 0)`. NGC 25.01 is the first tag with full sm_120 support; older
  PyTorch images will fall back to PTX JIT and be much slower.
- OOM at batch 8 × ctx 2048 → drop `batch_size` to 4 and bump
  `grad_accum_steps` to 32 (same effective batch of 128). Edit `core/config.py`
  giant preset or pass `--batch_size 4 --grad_accum_steps 32` on the CLI.
- HF download stalls → unset `HF_HUB_ENABLE_HF_TRANSFER` in `docker/run.sh`
  and fall back to the default downloader (slower but more reliable on
  flaky links).
