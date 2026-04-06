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

Run in order — each script downloads and merges into `data/raw_text.txt`:

```bash
python data/download.py            # Armenian Wikipedia (~1.5 GB)
python data/download_cc100.py      # CC-100 (~4.9 GB)
python data/download_culturax.py   # CulturaX (~5-8 GB)
python data/download_oscar.py      # OSCAR (~5-8 GB)
python data/download_mc4.py        # mC4 (~5-15 GB)
python data/download_hplt.py       # HPLT (~2-5 GB)
python data/download_glot500.py    # Glot500 (~0.2-0.5 GB)
```

Then tokenize:
```bash
python data/prepare.py --tokenizer bpe
```

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

## RunPod A40 Setup

GPU: 1x A40 (48 GB VRAM), Container Disk: 50 GB, Volume Disk: 200 GB, Mount: /workspace

```bash
# Setup
cd /workspace
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install sentencepiece datasets huggingface_hub

# Download all data (run each one)
python data/download.py
python data/download_cc100.py
python data/download_culturax.py
python data/download_oscar.py
python data/download_mc4.py
python data/download_hplt.py
python data/download_glot500.py

# Free space after downloads
rm -rf ~/.cache/huggingface
rm -f data/cc100_hy.txt.xz data/hywiki-latest-pages-articles.xml.bz2

# Tokenize with BPE
python data/prepare.py --tokenizer bpe

# Free raw text after tokenizing
rm -f data/raw_text.txt data/cc100_hy.txt data/oscar_hy.txt data/culturax_hy.txt
rm -f data/mc4_hy.txt data/hplt_hy.txt data/glot500_hy.txt

# Train (A40-optimized: batch 16, accum 8 = effective 128)
tmux new -s train
python train.py --preset xlarge --tokenizer bpe --batch_size 16 --grad_accum_steps 8

# Resume if interrupted
python train.py --preset xlarge --tokenizer bpe --batch_size 16 --grad_accum_steps 8 --resume_from checkpoints/step_XXXXX.pt
```

Estimated training time on A40: ~22 hours at $0.20/hr = ~$4.40 total.

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
- `checkpoints/` — model checkpoints
- `data/*.bz2`, `data/*.model`, `data/*.vocab` — intermediate files

## Conventions

- All download scripts follow the same pattern: download() then merge() into raw_text.txt
- BPE tokenizer is recommended for xlarge; char tokenizer only for tiny/small
- Checkpoints include model + optimizer + step + config (for resuming)
- upload_to_hf.py strips optimizer state to reduce upload size
