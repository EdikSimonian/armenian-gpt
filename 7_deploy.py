"""
Step 7: Deploy a trained ArmGPT model to HuggingFace Hub.

Converts the PyTorch checkpoint to a HuggingFace-compatible bundle and
uploads it along with the tokenizer and a model card.

Requirements:
    pip install huggingface_hub
    huggingface-cli login   # one-time

Usage:
    python 7_deploy.py --repo your-username/armgpt
    python 7_deploy.py --repo your-username/armgpt --checkpoint checkpoints/step_50000.pt
    python 7_deploy.py --repo your-username/armgpt \
        --checkpoint checkpoints/final.pt \
        --chat_checkpoint checkpoints_chat/final.pt

Inputs:
    checkpoints/final.pt (or specified checkpoint)
    data/tokenizer_{char|bpe}.json  (picked via --tokenizer or auto-detect)

Output:
    Uploaded files at https://huggingface.co/<repo>
"""

import argparse
import json
import os
import shutil
import tempfile

import torch


def convert_checkpoint(checkpoint_path, tokenizer_path, output_dir):
    """Convert ArmGPT checkpoint to a clean format for sharing."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = checkpoint["config"]
    state_dict = checkpoint["model"]

    # Save model weights (without optimizer state — much smaller)
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(state_dict, model_path)
    model_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"  Model weights: {model_size:.0f} MB")

    # Save config
    config_to_save = {
        "architecture": "ArmGPT",
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"],
        "block_size": cfg["block_size"],
        "dropout": cfg.get("dropout", 0.2),
        "vocab_size": None,  # filled from tokenizer below
        "tokenizer_type": cfg.get("tokenizer", "bpe"),
        "training_steps": checkpoint.get("step", cfg.get("max_iters")),
    }

    # Count parameters
    n_params = sum(p.numel() for p in state_dict.values())
    config_to_save["n_params"] = n_params
    print(f"  Parameters: {n_params:,}")

    # Copy tokenizer
    print(f"  Copying tokenizer from {tokenizer_path}")
    shutil.copy2(tokenizer_path, os.path.join(output_dir, "tokenizer.json"))

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    if tok_data.get("vocab_size"):
        config_to_save["vocab_size"] = tok_data["vocab_size"]
    elif tok_data.get("itos"):
        config_to_save["vocab_size"] = len(tok_data["itos"])
    else:
        config_to_save["vocab_size"] = 0

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    return config_to_save


def generate_model_card(config, repo_id, chat_included=False):
    """Generate a README.md model card for HuggingFace."""
    n_params_m = config["n_params"] / 1_000_000
    card = f"""---
language:
  - hy
license: apache-2.0
tags:
  - armenian
  - gpt
  - text-generation
  - causal-lm
pipeline_tag: text-generation
---

# ArmGPT ({n_params_m:.0f}M)

A GPT language model trained on Armenian text data.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | {config['n_params']:,} ({n_params_m:.0f}M) |
| Architecture | Transformer (RMSNorm, SwiGLU, RoPE) |
| Layers | {config['n_layer']} |
| Heads | {config['n_head']} |
| Embedding dim | {config['n_embd']} |
| Context length | {config['block_size']} tokens |
| Vocab size | {config['vocab_size']:,} |
| Tokenizer | BPE (SentencePiece) |
| Training steps | {config['training_steps']:,} |

## Training Data

Trained on Armenian text from multiple sources:
- Armenian Wikipedia
- CC-100 Armenian
- CulturaX Armenian
- OSCAR Armenian
- HPLT Armenian
- Glot500 Armenian

## Usage

```python
import torch
import json
from core.model import GPT
from core.bpe_tokenizer import BPETokenizer

# Load model
checkpoint = torch.load("model.pt", map_location="cpu")
with open("config.json") as f:
    cfg = json.load(f)

tokenizer = BPETokenizer.load("tokenizer.json")

model = GPT(
    vocab_size=cfg["vocab_size"],
    n_layer=cfg["n_layer"],
    n_head=cfg["n_head"],
    n_embd=cfg["n_embd"],
    block_size=cfg["block_size"],
    dropout=0.0,
)
model.load_state_dict(checkpoint)
model.eval()

# Generate text
prompt = "Հայաստան"
ids = tokenizer.encode(prompt)
context = torch.tensor([ids], dtype=torch.long)
output = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
print(tokenizer.decode(output[0].tolist()))
```
"""
    if chat_included:
        card += """
## Chat Model

This repo also includes a chat-finetuned version (`chat_model.pt` + `chat_tokenizer.json`).

```python
# Load chat model
chat_checkpoint = torch.load("chat_model.pt", map_location="cpu")
chat_tokenizer = BPETokenizer.load("chat_tokenizer.json")

# Format a question
prompt = "<|user|>Ինչ է Հայաստանի մայրաքաdelays<|end|><|assistant|>"
ids = chat_tokenizer.encode(prompt)
context = torch.tensor([ids], dtype=torch.long)
output = model.generate(context, max_new_tokens=200, temperature=0.7, top_k=40)
print(chat_tokenizer.decode(output[0].tolist()))
```
"""

    card += """
## License

Apache 2.0
"""
    return card


def upload(output_dir, repo_id, private=False):
    """Upload the converted model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed!")
        print("Install it with: pip install huggingface_hub")
        print(f"\nModel files are ready in: {output_dir}")
        print("You can upload manually with:")
        print(f"  huggingface-cli upload {repo_id} {output_dir}")
        return

    api = HfApi()

    # Check authentication
    try:
        user = api.whoami()
        print(f"\nLogged in as: {user['name']}")
    except Exception:
        print("Error: Not logged in to HuggingFace!")
        print("Run: huggingface-cli login")
        print(f"\nModel files are ready in: {output_dir}")
        return

    # Create repo if needed
    print(f"\nCreating/updating repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True, private=private)

    # Upload all files
    print("Uploading files...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Upload ArmGPT model",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload ArmGPT to HuggingFace")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID (e.g. your-username/armgpt)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--chat_checkpoint", type=str, default=None,
                        help="Path to chat-finetuned checkpoint (optional)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the tokenizer file")
    parser.add_argument("--chat_data_dir", type=str, default="data_chat",
                        help="Directory containing the chat tokenizer file")
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["char", "bpe"],
                        help="Tokenizer type. If omitted, auto-detects from data_dir.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for converted files (default: temp dir)")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private")
    parser.add_argument("--no_upload", action="store_true",
                        help="Only convert, don't upload")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return

    from core import (
        detect_tokenizer_type,
        tokenizer_path as _tokenizer_path,
    )
    try:
        tok_type = args.tokenizer or detect_tokenizer_type(args.data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    tokenizer_path = _tokenizer_path(args.data_dir, tok_type)
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer not found: {tokenizer_path}")
        return

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="armgpt_hf_")

    print(f"Output directory: {output_dir}\n")

    # Convert base model
    config = convert_checkpoint(args.checkpoint, tokenizer_path, output_dir)

    # Convert chat model if provided
    chat_included = False
    if args.chat_checkpoint and os.path.exists(args.chat_checkpoint):
        print(f"\nConverting chat model: {args.chat_checkpoint}")
        chat_ckpt = torch.load(args.chat_checkpoint, map_location="cpu", weights_only=False)
        torch.save(chat_ckpt["model"], os.path.join(output_dir, "chat_model.pt"))

        chat_tok_path = _tokenizer_path(args.chat_data_dir, tok_type)
        if os.path.exists(chat_tok_path):
            shutil.copy2(chat_tok_path, os.path.join(output_dir, "chat_tokenizer.json"))
        chat_included = True
        print("  Chat model added.")

    # Copy source files so users can run the model. The pipeline uses numeric
    # prefixes at repo root, but the HF bundle is a standalone artifact — drop
    # the `5_` prefix on generate.py. The `core/` subpackage ships intact so
    # the bundled generate.py's `from core.model import GPT` still resolves.
    source_map = {
        "core/__init__.py": "core/__init__.py",
        "core/model.py": "core/model.py",
        "core/char_tokenizer.py": "core/char_tokenizer.py",
        "core/bpe_tokenizer.py": "core/bpe_tokenizer.py",
        "5_generate.py": "generate.py",
    }
    for src_file, dest_name in source_map.items():
        src_path = os.path.join(os.path.dirname(__file__), src_file)
        if os.path.exists(src_path):
            dest = os.path.join(output_dir, dest_name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src_path, dest)

    # Generate model card
    model_card = generate_model_card(config, args.repo, chat_included)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)

    print(f"\nFiles prepared in: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"  {f:30s} {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f:30s} {size / 1024:.1f} KB")

    # Upload
    if not args.no_upload:
        upload(output_dir, args.repo, args.private)
    else:
        print(f"\nSkipping upload. Files ready in: {output_dir}")
        print(f"Upload manually with: huggingface-cli upload {args.repo} {output_dir}")


if __name__ == "__main__":
    main()
