# ArmGPT — Claude Code Guide

## Project Overview

Armenian GPT language model. Transformer architecture with RMSNorm, SwiGLU, and RoPE.
Trains on Armenian text data, then fine-tunes on Q&A for chat.

## Architecture

- `model.py` — GPT model (RMSNorm, RoPE, SwiGLU MLP, weight tying)
- `config.py` — All presets: tiny, small, medium, large, xlarge, finetune
- `train.py` — Training loop (AMP, grad accum, cosine LR, checkpointing)
- `generate.py` — Text generation with temperature/top-k sampling
- `upload_to_hf.py` — Convert checkpoint and upload to HuggingFace Hub

## Tokenizers

- `tokenizers/char_tokenizer.py` — Character-level (Level 1)
- `tokenizers/bpe_tokenizer.py` — SentencePiece BPE (Level 2, recommended for xlarge)

## Data Pipeline

### Stage 1: Pretraining data

Single command downloads all sources, merges into raw_text.txt, and cleans up caches.
Uses HuggingFace streaming mode — never loads full datasets into RAM.

```bash
python data/download_all.py
```

Sources (downloaded in order): Wikipedia, CC-100, CulturaX, OSCAR, mC4, HPLT, Glot500.
Skip sources with: `python data/download_all.py --skip wiki cc100`

Then tokenize (memory-safe, processes in chunks):
```bash
python data/prepare.py --tokenizer bpe
```

Individual download scripts also exist: `data/download_*.py`

### Stage 2: Chat fine-tuning data

```bash
python data/generate_armenian_qa.py                        # Generate Q&A with Claude API
python data/prepare_chat.py --source data/armenian_qa.json # Tokenize for fine-tuning
```

## Training Presets

| Preset | Params | Layers | Dim | Context | Steps | Target GPU |
|--------|--------|--------|-----|---------|-------|------------|
| tiny | ~0.2M | 1 | 64 | 64 | 1K | CPU |
| small | ~10M | 6 | 384 | 256 | 5K | T4 / MPS |
| medium | ~30M | 8 | 512 | 512 | 10K | A100 / V100 |
| large | ~85M | 12 | 768 | 512 | 20K | RTX 4090 |
| xlarge | ~350M | 24 | 1024 | 1024 | 100K | A40 / H100 |

## RunPod Setup (Secure Cloud — REQUIRED for persistent storage)

IMPORTANT: Always use Secure Cloud, not Community Cloud.
Community Cloud does NOT support network volumes — data is lost on crash/termination.

### Pod Configuration

- Cloud: **Secure Cloud** (not Community Cloud)
- GPU: 1x A40 (48 GB VRAM) — $0.40/hr on Secure Cloud
- Container Disk: 50 GB
- Network Volume: **200 GB** (create FIRST in your target datacenter, then attach to pod)
- Mount: /workspace
- Template: RunPod PyTorch 2.4.0 / CUDA 12.1

### Network Volume Setup (do this BEFORE creating the pod)

1. Go to RunPod Dashboard > Storage > Network Volumes
2. Create a 200 GB volume in your target datacenter region
3. When creating the pod, select this volume under "Network Volume"
4. Everything in /workspace persists even if the pod crashes or is deleted

### Full Training Workflow

```bash
# Setup
cd /workspace
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install sentencepiece datasets huggingface_hub

# Download all data (streaming mode, RAM-safe, auto-cleans HF cache)
python data/download_all.py

# Tokenize with BPE (chunked processing, RAM-safe)
python data/prepare.py --tokenizer bpe

# Free raw text after tokenizing (~20-40 GB saved)
rm -f data/raw_text.txt data/clean_text.txt

# Train (A40-optimized: batch 16, accum 8 = effective 128)
tmux new -s train
python train.py --preset xlarge --tokenizer bpe --batch_size 16 --grad_accum_steps 8

# Resume if interrupted
python train.py --preset xlarge --tokenizer bpe --batch_size 16 --grad_accum_steps 8 --resume_from checkpoints/step_XXXXX.pt
```

Estimated training time on A40: ~22 hours at $0.40/hr (Secure Cloud) = ~$8.80 total.

## Upload to HuggingFace

```bash
huggingface-cli login
python upload_to_hf.py --repo YourUsername/armgpt
# With chat model:
python upload_to_hf.py --repo YourUsername/armgpt --chat_checkpoint checkpoints_chat/final.pt
```

## Key Files Not in Git

These are generated locally / on the training machine and excluded via .gitignore:

- `data/*.bin` — tokenized training data
- `data/*.txt` — raw text files
- `data/tokenizer.json` — trained tokenizer
- `data/qa_parts/` — generated Q&A data
- `data_chat/` — chat fine-tuning data
- `checkpoints/` — model checkpoints
- `data/*.bz2`, `data/*.model`, `data/*.vocab` — intermediate files
- `pipeline_log.txt` — pipeline output log

## Conventions

- `download_all.py` is the main entry point for data — downloads all sources with streaming
- `prepare.py` processes text in chunks — safe for 20+ GB files
- BPE tokenizer is recommended for xlarge; char tokenizer only for tiny/small
- Checkpoints include model + optimizer + step + config (for resuming)
- upload_to_hf.py strips optimizer state to reduce upload size
- Download progress is tracked with `.{source}_done` marker files in data/ — allows safe resume if interrupted
