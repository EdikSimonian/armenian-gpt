# ArmGPT - Armenian Language Model

A GPT language model trained on Armenian text. Transformer architecture with RMSNorm, SwiGLU, and RoPE. Trains on a 63 GB Armenian corpus (~8.3B BPE tokens), then fine-tunes on Q&A for chat.

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

**Live demo:** [huggingface.co/spaces/edisimon/armgpt-demo](https://huggingface.co/spaces/edisimon/armgpt-demo)

---

## Requirements

**Python 3.10+** and the following packages:

```bash
pip install numpy sentencepiece huggingface_hub zstandard
```

**PyTorch 2.0+** — install the right version for your hardware:

```bash
# NVIDIA GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch

# Apple Silicon (MPS is auto-detected)
pip install torch
```

> **Note:** GPUs with compute capability < 7.0 (e.g. GTX 1080 Ti) work fine but skip `torch.compile()` automatically.

**Optional** (only needed if downloading raw sources or generating Q&A):
```bash
pip install datasets requests mwxml anthropic
```

---

## Quick Start

### Fastest: Download pre-tokenized data (skip steps 1-3)

If you just want to train, download the ready-to-use BPE token bins directly:

```bash
# Requires a HuggingFace token with read access to edisimon/armenian-clean-text (private)
# Get one at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

python 1_download.py --tokenized-only
```

This gives you `data/train_bpe.bin` (14 GB, 7.5B tokens), `data/val_bpe.bin` (1.6 GB), and the BPE tokenizer. Jump to [Step 4: Train](#step-4-train-the-model).

### Full pipeline from scratch

```bash
export HF_TOKEN=hf_your_token_here

# 1. Download clean corpus + Q&A from HuggingFace (63 GB decompressed)
python 1_download.py --download

# 2. (Skip — already cleaned if using --download)

# 3. Tokenize with BPE (parallel, 16 workers)
python 3_tokenize.py --tokenizer bpe

# 4. Train
python 4_train.py --preset small --tokenizer bpe

# 5. Generate text
python 5_generate.py --prompt "Հայاstanistan"

# 6. Fine-tune for chat
python 1_download.py --download --qa          # download Q&A data
python 2_prepare.py --qa                      # merge Q&A sources
python 3_tokenize.py --qa --tokenizer bpe     # tokenize for chat
python 6_finetune.py --tokenizer bpe          # fine-tune

# 7. Chat
python 8_chat.py
```

---

## Project Structure

```
armenian-gpt/
├── 1_download.py              # Download corpus, Q&A, or pre-tokenized data from HF
├── 2_prepare.py               # Clean + dedup corpus, or merge Q&A sources
├── 3_tokenize.py              # Train BPE tokenizer and encode to .bin files
├── 4_train.py                 # Stage 1: Pretrain on corpus
├── 5_generate.py              # Generate text from a pretrained model
├── 6_finetune.py              # Stage 2: Fine-tune on Q&A for chat
├── 8_chat.py                  # Interactive chat with fine-tuned model
├── core/
│   ├── model.py               # GPT model (RMSNorm, RoPE, SwiGLU MLP)
│   ├── config.py              # Training presets and hyperparameters
│   ├── bpe_tokenizer.py       # SentencePiece BPE tokenizer
│   ├── char_tokenizer.py      # Character-level tokenizer
│   ├── prepare_chat.py        # Format Q&A pairs with chat tokens
│   └── merge_sft_sources.py   # Deduplicate and merge Q&A files
├── data/                      # Generated: .bin files, tokenizer, raw text
├── checkpoints/               # Generated: model checkpoints
└── data_chat/                 # Generated: chat fine-tuning data
```

---

## Pipeline Steps

### Step 1: Download Data

All downloads go through `1_download.py`. The corpus and Q&A data are hosted on a **private** HuggingFace dataset repo (`edisimon/armenian-clean-text`). You need an `HF_TOKEN` with read access.

```bash
# Download everything (corpus + Q&A + tokenized bins)
python 1_download.py --download --tokenized

# Download only the pre-tokenized bins (fastest, ~10 GB)
python 1_download.py --tokenized-only

# Download only corpus (63 GB) — you'll tokenize yourself
python 1_download.py --download

# Download Q&A sources for fine-tuning
python 1_download.py --download --qa

# Download from raw sources instead of HF (slow, ~100 GB download)
python 1_download.py
```

### Step 2: Prepare Data

Only needed if you downloaded raw sources (not `--download`):

```bash
python 2_prepare.py               # clean + dedup corpus -> clean_text.txt
python 2_prepare.py --qa          # merge Q&A files -> qa_merged.json
```

### Step 3: Tokenize

Only needed if you didn't use `--tokenized-only`:

```bash
python 3_tokenize.py --tokenizer bpe              # corpus -> train_bpe.bin + val_bpe.bin
python 3_tokenize.py --qa --tokenizer bpe          # Q&A -> data_chat/train_bpe.bin
```

BPE tokenizer: 16,000 tokens, trained with SentencePiece. Encoding runs in parallel (up to 16 workers).

### Step 4: Train the Model

```bash
python 4_train.py --preset small --tokenizer bpe      # 10M params, ~30 min on GPU
python 4_train.py --preset medium --tokenizer bpe     # 30M params, ~2 hrs
python 4_train.py --preset xlarge --tokenizer bpe     # 350M params, ~22 hrs on A40
```

Resume from a checkpoint:
```bash
python 4_train.py --preset xlarge --tokenizer bpe --resume_from checkpoints/step_5000.pt
```

### Step 5: Generate Text

```bash
python 5_generate.py --prompt "Հայաstanistan" --temperature 0.7 --length 300
python 5_generate.py --checkpoint checkpoints/step_5000.pt
```

### Step 6: Fine-tune for Chat

```bash
# Prepare chat data (if not done already)
python 2_prepare.py --qa
python 3_tokenize.py --qa --tokenizer bpe

# Fine-tune (loads checkpoints/final.pt automatically)
python 6_finetune.py --tokenizer bpe

# Fine-tune with HF upload enabled
python 6_finetune.py --tokenizer bpe --upload
```

### Step 7: Chat

```bash
python 8_chat.py
python 8_chat.py --temperature 0.5 --max_length 500
```

---

## Model Presets

| Preset | Params | Layers | Dim | Context | Steps | Target GPU |
|--------|--------|--------|-----|---------|-------|------------|
| `tiny` | ~0.2M | 1 | 64 | 64 | 1K | CPU |
| `small` | ~10M | 6 | 384 | 256 | 5K | GTX 1080 Ti / T4 |
| `medium` | ~30M | 8 | 512 | 512 | 10K | A100 / V100 |
| `large` | ~85M | 12 | 768 | 512 | 20K | RTX 4090 |
| `xlarge` | ~350M | 24 | 1024 | 1024 | 36K | A40 / H100 |
| `giant` | ~1B | 32 | 1536 | 2048 | 122K | H200 |

---

## Training Data

The cleaned Armenian corpus is published at [`edisimon/armenian-clean-text`](https://huggingface.co/datasets/edisimon/armenian-clean-text) (private).

| Content | HF Path | Size |
|---------|---------|------|
| Cleaned corpus (zstd) | `corpus/clean_text.txt.zst` | ~7 GB compressed, 63 GB raw |
| Train tokens (zstd) | `tokenized/train_bpe.bin.zst` | ~8.5 GB compressed, 14 GB raw |
| Val tokens (zstd) | `tokenized/val_bpe.bin.zst` | ~1 GB compressed, 1.6 GB raw |
| BPE tokenizer | `tokenized/tokenizer_bpe.json` | 1.2 MB |
| BPE model | `tokenized/bpe_model.model` | 590 KB |
| Q&A files | `finetune/*.json` | ~29K pairs |

**Corpus sources:** Wikipedia, Wikisource, Wiktionary, Wikiquote, CC-100, HPLT 3.0, ARLIS, CC-News, CulturaX, mC4, Glot500, FineTranslations.

---

## Upload to HuggingFace

```bash
# Upload model checkpoint
huggingface-cli login
python upload_to_hf.py --repo YourUsername/armgpt

# Upload corpus + tokenized data to dataset repo
python 1_download.py --upload --tokenized
```

---

## Cloud Training (RunPod)

For large models, use RunPod Secure Cloud with a network volume:

```bash
# On a RunPod A40 pod with 200 GB network volume at /workspace
cd /workspace
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install sentencepiece huggingface_hub zstandard

export HF_TOKEN=hf_your_token_here
python 1_download.py --tokenized-only    # ~10 GB, skip tokenization

python 4_train.py --preset xlarge --tokenizer bpe
# ~22 hours on A40 at $0.40/hr = ~$8.80
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Use a smaller preset or reduce `--batch_size` |
| `torch.compile` error on older GPU | Handled automatically (skips on CC < 7.0) |
| `UnicodeEncodeError` on Windows | Fixed in latest code (UTF-8 output forced) |
| `No module named 'torch'` | `pip install torch` (see Requirements for GPU version) |
| `train_bpe.bin not found` | Run `python 3_tokenize.py --tokenizer bpe` or `python 1_download.py --tokenized-only` |
| `HF_TOKEN` / 401 errors | Set `export HF_TOKEN=hf_...` (dataset repo is private) |

---

## License

MIT
