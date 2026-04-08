---
language:
- hy
license: mit
tags:
- armenian
- gpt
- text-generation
- transformer
- causal-lm
library_name: pytorch
pipeline_tag: text-generation
model-index:
- name: ArmGPT
  results: []
---

# ArmGPT — Armenian Language Model (320M)

A 320M parameter GPT language model trained from scratch on Armenian text. Built with a modern transformer architecture inspired by LLaMA.

## 🚀 Try it live

Interactive demo on HuggingFace Spaces: **[edisimon/armgpt-demo](https://huggingface.co/spaces/edisimon/armgpt-demo)**

The demo runs the latest training checkpoint on a free CPU instance. Type an Armenian prompt, get a streaming completion. The Space automatically picks up the newest `step_*.pt` from this repo on every cold start, and includes a "Reload latest checkpoint" button to hot-swap to a newer one without restarting.

> ⚠️ Free-tier CPU — first request takes 1–3 minutes to download and load the checkpoint, subsequent generations take 5–15 seconds. This is a base language model (text completion), not instruction-tuned.

## Model Details

| Parameter | Value |
|-----------|-------|
| Parameters | 320M |
| Architecture | GPT (RMSNorm, SwiGLU, RoPE) |
| Layers | 24 |
| Heads | 16 |
| Embedding dim | 1024 |
| Context window | 1024 tokens |
| Vocab size | 16,000 (BPE via SentencePiece) |
| Training data | ~3.3B tokens of Armenian text |
| Precision | fp32 with tf32 matmuls |

## Architecture

Custom GPT implementation with modern components:
- **RMSNorm** instead of LayerNorm (faster, no bias)
- **SwiGLU MLP** instead of GELU (better performance, used by LLaMA/Mistral)
- **Rotary Position Embeddings (RoPE)** instead of learned position embeddings
- **Weight tying** between embedding and output head
- **Flash Attention** via PyTorch's `scaled_dot_product_attention`

## Training Data

Trained on ~3.3B tokens from multiple Armenian text sources:

| Source | Description |
|--------|-------------|
| Armenian Wikipedia | ~325K articles |
| CC-100 | Common Crawl monolingual Armenian |
| CulturaX | Large-scale cleaned multilingual corpus |
| OSCAR-2301 | Web-crawled Armenian text |
| mC4 | Multilingual C4 Armenian subset |
| HPLT v2.0 | Common Crawl + Internet Archive |
| Glot500 | Multilingual low-resource corpus |

Data was deduplicated (paragraph-level MD5) and cleaned to retain only Armenian Unicode characters, digits, and basic punctuation. The cleaned corpus (~29 GB) is available at [edisimon/armenian-clean-text](https://huggingface.co/datasets/edisimon/armenian-clean-text).

## Training

- **GPU**: NVIDIA RTX A6000 (48 GB)
- **Effective batch size**: 144 (batch_size=24 × grad_accum=6)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1, grad_clip=1.0)
- **Schedule**: Linear warmup (2K steps) + cosine decay
- **Features**: tf32, torch.compile, AMP (fp16 forward pass)

## Usage

```python
# Clone the repo
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install torch sentencepiece

# Generate text
python generate.py --checkpoint checkpoints/step_6000.pt --tokenizer bpe
```

## Checkpoints

Intermediate checkpoints are saved during training:
- `checkpoints/step_2000.pt`
- `checkpoints/step_4000.pt`
- `checkpoints/step_6000.pt`
- *(more added as training continues)*

Each checkpoint contains model weights, optimizer state, step number, and config — allowing full training resumption.

## Limitations

- This is a base language model (text completion), not instruction-tuned
- Trained on web-crawled data which may contain biases or inaccuracies
- 1024 token context window limits long-range coherence
- Training is ongoing — model quality improves with more steps

## Links

- **Code**: [github.com/EdikSimonian/armenian-gpt](https://github.com/EdikSimonian/armenian-gpt)
- **Dataset**: [edisimon/armenian-clean-text](https://huggingface.co/datasets/edisimon/armenian-clean-text)
