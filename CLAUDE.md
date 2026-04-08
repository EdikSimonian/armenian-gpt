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

Pre-cleaned dataset available on HuggingFace (private, ~29 GB):
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='edisimon/armenian-clean-text', filename='clean_text.txt', repo_type='dataset', local_dir='data/')"
```

Then tokenize (memory-safe, processes in chunks, parallel with 16 workers):
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

# Train (A6000-optimized: batch 32, accum 4 = effective 128, tf32 + torch.compile auto-enabled)
tmux new -s train
python train.py --preset xlarge --tokenizer bpe

# Resume if interrupted
python train.py --preset xlarge --tokenizer bpe --resume_from checkpoints/step_XXXXX.pt
```

Estimated training time on A40: ~22 hours at $0.40/hr (Secure Cloud) = ~$8.80 total.

## Upload to HuggingFace

```bash
huggingface-cli login
python upload_to_hf.py --repo YourUsername/armgpt
# With chat model:
python upload_to_hf.py --repo YourUsername/armgpt --chat_checkpoint checkpoints_chat/final.pt
```

## Deploy Demo to HuggingFace Spaces

The interactive Gradio demo lives in a separate Space repo (`edisimon/armgpt-demo`). It is **not** part of this repo's git tree — its source files are kept under `/workspace/armgpt-space/` on the training machine and pushed via `HfApi.upload_folder()`.

### Space layout
```
/workspace/armgpt-space/
├── app.py                  # Gradio UI + lazy load + streaming generation
├── requirements.txt        # CPU torch wheel + sentencepiece + huggingface_hub pin
├── README.md               # HF Space frontmatter (sdk, python_version, etc.)
├── model.py                # Copy of repo's model.py (Space needs class definition)
└── tokenizers/
    ├── __init__.py
    └── bpe_tokenizer.py    # Copy of repo's BPE tokenizer
```

### Critical config (learned the hard way)
- **`sdk_version: 5.6.0`** in `README.md` frontmatter — gradio 4.x has a jinja2 dict-hash bug.
- **`python_version: "3.11"`** in `README.md` frontmatter — gradio's pydub dep imports `audioop`, removed in Python 3.13.
- **`huggingface_hub==0.25.2`** pin in requirements — gradio 5.x still imports the deprecated `HfFolder` symbol, removed in newer hub releases.
- **CPU torch wheel** via `--extra-index-url https://download.pytorch.org/whl/cpu` — default torch wheel pulls CUDA libs and OOMs the build.
- **`demo.launch(ssr_mode=False)`** — Gradio 5's SSR mode breaks event handlers (clicks don't reach the server).
- **No `gr.Progress()`** — overlays a duplicate progress bar on streaming HTML outputs and looks buggy.
- **`gr.HTML` with `<pre>` for streaming output** — `gr.Textbox` flashes on every yield because it re-renders the whole component including chrome.

### Auth for private model repo
The model repo `edisimon/armgpt` is private. The Space reads the model via an `HF_TOKEN` Space secret. **`list_repo_files` does NOT pick up the token from `HfApi(token=...)` — you MUST pass `token=HF_TOKEN` to every per-call API method too.** Same for `hf_hub_download(..., token=HF_TOKEN)`.

To set/rotate the secret programmatically:
```python
from huggingface_hub import HfApi, get_token
HfApi().add_space_secret(
    repo_id="edisimon/armgpt-demo",
    key="HF_TOKEN",
    value=get_token(),  # or a fine-grained read-only token
)
```

The Space auto-restarts on secret change. For least privilege, create a fine-grained token at `https://huggingface.co/settings/tokens` with read-only access scoped to `edisimon/armgpt`, instead of using a full-access account token.

### Auto-update to latest checkpoint
`app.py` calls `_latest_checkpoint_name()` on every cold start AND when the user clicks "Reload latest checkpoint" — it lists files in the model repo, extracts `step_NNNNN.pt` numbers, picks the highest. So as `upload_watcher.sh` pushes new checkpoints to the model repo from the training pod, the demo Space picks them up automatically (cold start) or on demand (button).

### Checkpoint compatibility gotcha
Training uses `torch.compile()`, which prefixes all state-dict keys with `_orig_mod.`. The demo's vanilla `GPT` instance doesn't have those prefixes, so `app.py` strips them on load:
```python
if any(k.startswith("_orig_mod.") for k in state.keys()):
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
```
If you train without `torch.compile()`, you can remove that block.

### Push the Space
```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/workspace/armgpt-space",
    repo_id="edisimon/armgpt-demo",
    repo_type="space",
    commit_message="Update demo",
)
```

### Freeing HF storage quota

**Critical gotcha: `super_squash_history` does NOT garbage-collect orphaned LFS objects.** Squash collapses git history but the LFS blobs from old/deleted/overwritten files stay on HF and keep counting against your storage quota. Same for `delete_file` — only removes the file from the current commit, not the underlying LFS object.

To actually free storage, you need TWO operations:

```python
from huggingface_hub import HfApi
api = HfApi()
REPO = "edisimon/armgpt"
RTYPE = "model"

# 1. (Optional) squash history first if you want to drop file history
api.super_squash_history(repo_id=REPO, repo_type=RTYPE, branch="main",
                         commit_message="Squash to free LFS storage")

# 2. Find and permanently delete LFS objects not referenced in the current tree
lfs = list(api.list_lfs_files(repo_id=REPO, repo_type=RTYPE))
referenced = set(api.list_repo_files(REPO, repo_type=RTYPE))
orphans = [f for f in lfs if f.filename not in referenced]
print(f"Freeing {sum(f.size for f in orphans)/1e9:.1f} GB across {len(orphans)} orphans")
api.permanently_delete_lfs_files(repo_id=REPO, repo_type=RTYPE, lfs_files=orphans)
```

**Watch for duplicates:** the same `filename` can appear with multiple OIDs in the LFS list — that happens when a file was re-uploaded (e.g. graceful shutdown saving a checkpoint twice, upload retries). All non-current OIDs are orphans even if the filename still exists in the tree.

**Real example from this project:** after deleting old checkpoints + train.bin/val.bin and squashing, the visible repo was 15 GB but the HF dashboard still showed 65 GB used because 13 orphan LFS blobs (49 GB) were still around. `permanently_delete_lfs_files` got rid of them. The dashboard refreshes within a few minutes.

### Debug runtime errors
Build/runtime logs from the API:
```python
from huggingface_hub import HfApi
info = HfApi().get_space_runtime("edisimon/armgpt-demo")
print(info.stage, info.raw.get("errorMessage", ""))
```

Streaming runtime logs (SSE — gives you `print()` output from `app.py`):
```bash
TOKEN=$(python -c "from huggingface_hub import get_token; print(get_token())")
curl -N -H "Authorization: Bearer $TOKEN" \
  https://huggingface.co/api/spaces/edisimon/armgpt-demo/logs/run
```

The build logs endpoint exists too (`/logs/build`) but its `errorMessage` from `get_space_runtime` is usually just cache-miss noise — the real error is in the SSE stream.

## Deploy to Modal (GPU serverless, OpenAI-compatible API)

A separate Modal deployment exists in parallel with the HF Space. Modal hosts the model on a T4 GPU with scale-to-zero; the HF Space can optionally be wired to it as a frontend, but right now **the Space loads the model from HF directly and does CPU inference**, while **Modal is kept running as a standalone API endpoint** for programmatic access (other apps, OpenAI SDK clients, etc.).

### Live deployment
- **URL:** `https://edisimon--armgpt-web.modal.run`
- **App name:** `armgpt`
- **GPU:** T4 (Modal)
- **Scale-to-zero:** 120s idle → 0 containers; cold start ~30s (image pull + weights load)
- **Endpoints exposed:**
  - `GET /` — route map
  - `GET /info` — `{model, checkpoint_step, vocab_size}`
  - `POST /generate_stream` — simple SSE: `{prompt, length, temperature, top_k, repetition_penalty}` → SSE `data: {"delta": "..."}` + `data: {"done": true}`
  - `POST /v1/chat/completions` — OpenAI-compatible, `stream=true|false`, flattens `messages` into a `User: … Assistant: …` prompt
  - `POST /v1/completions` — OpenAI-compatible raw completion
  - `GET /v1/models` — returns `[{id: "armgpt", ...}]`

### Source files
All Modal sources live at `/workspace/armgpt-modal/` on the training pod. Not in git:
```
/workspace/armgpt-modal/
├── modal_app.py         # Main: @app.cls ArmGPT + @modal.asgi_app web
├── upload_to_volume.py  # Bootstrap: pushes latest HF checkpoint into Modal Volume
├── model.py             # Verbatim copy of repo's model.py
└── bpe_tokenizer.py     # Verbatim copy of repo's tokenizer (flat import, not in tokenizers/)
```

### Resuming from a fresh machine
```bash
# 1. Clone the repo and recreate the Modal workspace
git clone https://github.com/EdikSimonian/armenian-gpt.git
mkdir -p /workspace/armgpt-modal
cp armenian-gpt/model.py /workspace/armgpt-modal/model.py
cp armenian-gpt/tokenizers/bpe_tokenizer.py /workspace/armgpt-modal/bpe_tokenizer.py
# Then copy modal_app.py and upload_to_volume.py from backups (these are custom, not in git)

# 2. Install and authenticate Modal
pip install modal
modal token set --token-id <ak-...> --token-secret <as-...>
# Or: modal token new (opens browser)

# 3. Make sure HF auth is configured (upload_to_volume.py needs it to download private model)
huggingface-cli login

# 4. Populate the Modal Volume with the latest checkpoint
cd /workspace/armgpt-modal
python upload_to_volume.py              # latest step_*.pt from edisimon/armgpt
# or: python upload_to_volume.py step_35000   # specific step

# 5. Deploy
modal deploy modal_app.py
# → returns https://edisimon--armgpt-web.modal.run
```

### Updating the deployed checkpoint (no redeploy needed)
Modal Volume is persistent across deploys. To serve a newer checkpoint:
```bash
cd /workspace/armgpt-modal
python upload_to_volume.py          # overwrites checkpoint.pt in the volume
modal app stop armgpt                # force cold start on next request
# Next request pulls the new weights
```
Or if you just wait, the next natural cold start will pick up the new volume contents.

### Key config inside `modal_app.py`
- **Image:** `debian_slim(python_version="3.11")` + `torch==2.4.0`, `sentencepiece`, `fastapi[standard]`, `huggingface_hub` + `add_local_python_source("model", "bpe_tokenizer")`
- **Volume:** `modal.Volume.from_name("armgpt-checkpoints", create_if_missing=True)` mounted at `/ckpt`
- **`@app.cls`** decorator: `gpu="T4"`, `scaledown_window=120`, `timeout=600`, `max_containers=2`
- **`@modal.enter()`** loads model + tokenizer into VRAM; strips `_orig_mod.` prefix if present (torch.compile checkpoints)
- **`@modal.method()` `generate_stream`** — generator yielding decoded text deltas (full-decode-and-diff for BPE correctness)
- **`@modal.asgi_app()` `web`** — FastAPI app with CORS, three inference routes + OpenAI shim; calls `ArmGPT().generate_stream.remote_gen(...)` from inside each request handler

### Gotchas
- **Incremental BPE decoding is unreliable** — individual SentencePiece tokens don't always form complete characters. `generate_stream` decodes the full token sequence each step and yields the string diff instead.
- **`add_local_python_source` mounts bare modules**, so `model.py` and `bpe_tokenizer.py` must be **top-level** in `/workspace/armgpt-modal/`, not inside a `tokenizers/` package. The repo's copy lives at `tokenizers/bpe_tokenizer.py` but Modal imports it as just `bpe_tokenizer`.
- **Don't deploy twice in a row in parallel** — Modal rejects "app name already taken" on the second one; no harm but creates a dead app record.
- **Cleaning up stale app records:** `modal app list` shows both active and recently stopped. There is no `modal app delete` — stopped apps age out on their own (~24–48h). Only `deployed` apps serve traffic.

### Testing the endpoint
```bash
# Cold-start info check
curl https://edisimon--armgpt-web.modal.run/info

# SSE streaming
curl -N -X POST https://edisimon--armgpt-web.modal.run/generate_stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Հայաստանի մայրաքաղաքը", "length": 100, "repetition_penalty": 1.15}'

# OpenAI SDK client
# python: OpenAI(base_url="https://edisimon--armgpt-web.modal.run/v1", api_key="sk-unused")
```

### Cost
- **Scale-to-zero**, so you only pay while a request is processing.
- T4 is ~$0.59/hr while warm. A 2-second generation costs ~$0.0003. Modal's $30/month free credits cover thousands of demo generations per month.
- Cold starts bill for the image pull + GPU spin time too (~30s once every 2+ min of idle).

## HF Space (`edisimon/armgpt-demo`) — HF-backed configuration (current)

The Space is **back to the HF-loading configuration** after a brief experiment with a Modal-backed frontend. The Modal app still runs independently, but the Space downloads the checkpoint from `edisimon/armgpt` on first request and does inference on the free CPU container — no dependency on Modal.

- **URL:** https://huggingface.co/spaces/edisimon/armgpt-demo
- **Backend:** free CPU (Python 3.11, gradio 5.6, CPU torch)
- **Secrets configured:** `HF_TOKEN` (fine-grained read on `edisimon/armgpt`). `MODAL_ENDPOINT` secret was removed when the Space was reverted.
- **Source files** at `/workspace/armgpt-space/`: `app.py`, `requirements.txt`, `README.md`, `model.py`, `tokenizers/__init__.py`, `tokenizers/bpe_tokenizer.py`
- **To push updates:** `HfApi().upload_folder(folder_path="/workspace/armgpt-space", repo_id="edisimon/armgpt-demo", repo_type="space", ...)`

### Resuming Space work from a fresh machine
```bash
# 1. Re-stage the Space files locally
mkdir -p /workspace/armgpt-space/tokenizers
# (download from HF if needed — see "Restore Space files from a specific commit" below)

# 2. Authenticate
huggingface-cli login

# 3. Push changes
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='/workspace/armgpt-space',
    repo_id='edisimon/armgpt-demo',
    repo_type='space',
    commit_message='Update demo',
)"
```

### Restore Space files from a specific commit
Useful for quickly reverting to a known-good version:
```python
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
REPO = "edisimon/armgpt-demo"
# List recent commits to find a good revision
for c in api.list_repo_commits(REPO, repo_type="space")[:10]:
    print(c.commit_id[:10], c.title)
# Then restore files from that revision
REVISION = "a871ed7889"  # example
for f in ["app.py", "requirements.txt", "model.py", "tokenizers/__init__.py", "tokenizers/bpe_tokenizer.py"]:
    local = hf_hub_download(REPO, f, repo_type="space", revision=REVISION)
    api.upload_file(path_or_fileobj=local, path_in_repo=f, repo_id=REPO, repo_type="space",
                    commit_message=f"Revert {f}")
```

## Project state summary (for resuming from another machine)

As of 2026-04-07, the following artifacts exist across services:

| Artifact | Location | Purpose |
|---|---|---|
| Training code | GitHub `EdikSimonian/armenian-gpt` | Source repo (frozen from new commits until 5pm PST 2026-04-07) |
| Model weights | HF `edisimon/armgpt` (private) | Rolling window of latest checkpoints (max 5 kept, manually trimmed) |
| Training data | HF `edisimon/armenian-clean-text` (private, ~31 GB) | Cleaned Armenian corpus, referenced by `data/prepare.py` |
| Training pod | RunPod Secure Cloud, A6000 48GB, 200GB network volume at `/workspace` | Active training run (xlarge preset, 36000 steps) |
| Demo UI | HF Space `edisimon/armgpt-demo` | Gradio CPU demo loading checkpoints from `edisimon/armgpt` |
| GPU inference API | Modal app `armgpt` at `edisimon--armgpt-web.modal.run` | Independent OpenAI-compatible API, scale-to-zero |

Background processes currently running on the training pod:
- `train.py` (main training loop) — PID varies, check with `pgrep -af train.py`
- `upload_watcher.sh` — pushes new `step_*.pt` to `edisimon/armgpt` every 1000 steps
- `watchdog.sh` — kills Python on memory pressure (reads host `free`; note: doesn't see cgroup memory)
- `stop_after_training.sh` — waits for training to complete, then `runpodctl stop pod`
- `orchestrate_16k_resume.sh` — already completed, one-shot script (historical)

### Required auth tokens to resume from another machine
1. **HF token** — `huggingface-cli login` (needs read access to private `edisimon/armgpt` and `edisimon/armenian-clean-text`; write access if updating Space or pushing checkpoints)
2. **Modal token** — `modal token set --token-id <ak-...> --token-secret <as-...>` (from Modal dashboard → Settings → Tokens)
3. **RunPod access** — only needed if you want to SSH back into the training pod or use `runpodctl`
4. **GitHub access** — only needed for the source code, which is public

### Things NOT in git that need to be preserved/recreated
- `/workspace/armgpt-modal/modal_app.py` — **the Modal deployment source** (only copy lives on the training pod right now; back it up before pod teardown!)
- `/workspace/armgpt-modal/upload_to_volume.py` — **bootstrap script for the Modal Volume**
- `/workspace/armgpt-space/app.py` — the Space's Gradio app (HF-loading version). A copy is in the Space repo's git history at `edisimon/armgpt-demo`, but if you restore it from there, make sure to also restore `model.py` and `tokenizers/`
- `.uploaded_checkpoints` in `/workspace/armenian-gpt/` — tracks which checkpoints `upload_watcher.sh` has already pushed (regenerable)
- Local checkpoints in `/workspace/armenian-gpt/checkpoints/` — the most recent 5 are mirrored to HF, older ones are local-only and not backed up

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
