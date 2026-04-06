"""
ArmGPT Training Script

Trains the GPT model on Armenian text data.

Usage:
    python train.py                     # train with default "small" config
    python train.py --preset tiny       # quick test on CPU
    python train.py --preset medium     # better results, needs GPU
    python train.py --resume_from checkpoints/step_1000.pt  # resume training

What happens during training:
    1. Load the prepared data (data/train.bin and data/val.bin)
    2. Create the GPT model
    3. Repeatedly: grab a batch of text, predict next tokens, learn from mistakes
    4. Every N steps: check validation loss, generate sample text, save checkpoint
"""

import os
import sys

# Force unbuffered output so logs appear immediately when piped
os.environ["PYTHONUNBUFFERED"] = "1"

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

import time
import json
import math
import numpy as np
import torch

from model import GPT
from config import get_config


def load_data(data_dir, device):
    """Load pre-encoded training and validation data."""
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print("Run 'python data/download.py' and 'python data/prepare.py' first.")
        sys.exit(1)

    # Memory-map to avoid loading multi-GB datasets into RAM
    print("Loading data...")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"  Train: {len(train_data):,} tokens ({os.path.getsize(train_path)/1024/1024:.0f} MB)")
    print(f"  Val:   {len(val_data):,} tokens ({os.path.getsize(val_path)/1024/1024:.0f} MB)")
    return train_data, val_data


def load_tokenizer(data_dir, tokenizer_type):
    """Load the tokenizer that was used during data preparation."""
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"Error: {tok_path} not found! Run data/prepare.py first.")
        sys.exit(1)

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    if tok_data["type"] == "char":
        from tokenizers.char_tokenizer import CharTokenizer
        return CharTokenizer.load(tok_path)
    else:
        from tokenizers.bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(tok_path)


def get_batch(data, block_size, batch_size, device):
    """Grab a random batch of sequences from the data."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    """Estimate average loss on train and validation data."""
    model.eval()
    results = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], cfg["device"])
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[split_name] = losses.mean().item()
    model.train()
    return results


def get_lr(step, cfg):
    """Learning rate schedule: linear warmup then cosine decay."""
    # Warmup phase
    if step < cfg["warmup_iters"]:
        return cfg["learning_rate"] * step / max(cfg["warmup_iters"], 1)
    # Decay phase
    decay_ratio = (step - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    cfg = get_config()
    device = cfg["device"]
    use_amp = device == "cuda"

    # Enable tf32 for Ampere+ GPUs (A6000, A100, RTX 30xx/40xx) — ~2x faster matmuls
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Print configuration
    print(f"\n{'='*50}")
    print(f"  ArmGPT Training")
    print(f"{'='*50}")
    print(f"  Device:      {device}")
    print(f"  Model:       {cfg['n_layer']} layers, {cfg['n_head']} heads, {cfg['n_embd']} dim")
    print(f"  Block size:  {cfg['block_size']}")
    print(f"  Batch size:  {cfg['batch_size']}")
    print(f"  Max iters:   {cfg['max_iters']}")
    print(f"  Tokenizer:   {cfg['tokenizer']}")
    print(f"  Grad accum:  {cfg.get('grad_accum_steps', 1)} (eff. batch = {cfg['batch_size'] * cfg.get('grad_accum_steps', 1)})")
    print(f"  LR:          {cfg['learning_rate']}")
    print(f"  AMP:         {'enabled' if use_amp else 'disabled'}")
    print(f"{'='*50}\n")

    # Load data and tokenizer
    train_data, val_data = load_data(cfg["data_dir"], device)
    tokenizer = load_tokenizer(cfg["data_dir"], cfg["tokenizer"])
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    ).to(device)

    # Compile model for faster training (PyTorch 2.0+)
    if device == "cuda" and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Mixed precision scaler for faster training on GPU
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint if specified
    start_iter = 0
    if cfg["resume_from"] and os.path.exists(cfg["resume_from"]):
        print(f"\nResuming from {cfg['resume_from']}...")
        checkpoint = torch.load(cfg["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["step"]
        print(f"Resumed at step {start_iter}")

    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Metrics tracking
    metrics = {"steps": [], "train_loss": [], "val_loss": [],
               "perplexity": [], "tokens_per_sec": [], "accuracy": []}

    # Training loop
    grad_accum = cfg.get("grad_accum_steps", 1)
    tokens_per_step = cfg["batch_size"] * cfg["block_size"] * grad_accum
    print(f"  Grad accum:  {grad_accum} steps (effective batch = {cfg['batch_size'] * grad_accum})")
    print(f"\nStarting training from step {start_iter}...\n")
    model.train()
    t0 = time.time()
    train_start = time.time()
    running_loss = 0.0
    tps = 0

    for step in range(start_iter, cfg["max_iters"]):
        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation: run multiple micro-batches before updating
        for micro_step in range(grad_accum):
            x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits, loss = model(x, y)
                loss = loss / grad_accum  # scale loss by accumulation steps
            running_loss += loss.detach()
            scaler.scale(loss).backward()

        # Clip gradients to prevent explosions
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Print training loss (only sync GPU here, every log_interval steps)
        if step % cfg["log_interval"] == 0 and step > start_iter:
            # Now we sync to read the accumulated loss
            avg_loss = (running_loss / cfg["log_interval"]).item()
            running_loss = 0.0

            dt = time.time() - t0
            steps_done = cfg["log_interval"]
            tps = tokens_per_step * steps_done / dt
            t0 = time.time()

            # Time estimates
            elapsed = time.time() - train_start
            iters_done = step - start_iter
            iters_left = cfg["max_iters"] - step
            eta = (elapsed / iters_done) * iters_left if iters_done > 0 else 0

            print(f"step {step:5d}/{cfg['max_iters']} | loss {avg_loss:.4f} | "
                  f"lr {lr:.2e} | {tps:,.0f} tok/s | "
                  f"elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}")

        # Evaluate and generate samples
        if step > 0 and step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            perplexity = math.exp(min(losses["val"], 20))  # cap to avoid overflow

            # Calculate accuracy on a validation batch
            x_val, y_val = get_batch(val_data, cfg["block_size"], cfg["batch_size"], device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    val_logits, _ = model(x_val, y_val)
                preds = val_logits.argmax(dim=-1)
                accuracy = (preds == y_val).float().mean().item() * 100

            print(f"\n{'='*50}")
            print(f"  Step {step} Evaluation")
            print(f"  Train loss:   {losses['train']:.4f}")
            print(f"  Val loss:     {losses['val']:.4f}")
            print(f"  Perplexity:   {perplexity:.2f}")
            print(f"  Accuracy:     {accuracy:.1f}%")
            print(f"{'='*50}")

            # Log metrics
            metrics["steps"].append(step)
            metrics["train_loss"].append(losses["train"])
            metrics["val_loss"].append(losses["val"])
            metrics["perplexity"].append(perplexity)
            metrics["accuracy"].append(accuracy)
            metrics["tokens_per_sec"].append(tps if 'tps' in dir() else 0)

            # Save metrics
            metrics_path = os.path.join(cfg["checkpoint_dir"], "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Reset timer after eval (don't count eval time in tok/s)
            t0 = time.time()
            running_loss = 0.0

        # Generate sample text
        if step > 0 and step % cfg["sample_interval"] == 0:
            model.eval()
            seed_text = "Հayastan"
            seed_ids = tokenizer.encode(seed_text)
            if len(seed_ids) == 0:
                seed_ids = [0]
            context = torch.tensor([seed_ids], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=cfg["sample_length"],
                                       temperature=0.8, top_k=40)
            text = tokenizer.decode(generated[0].tolist())
            print(f"\n--- Sample (step {step}) ---")
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
            print("--- End sample ---\n")
            model.train()
            t0 = time.time()
            running_loss = 0.0

        # Save checkpoint and upload to HF
        if step > 0 and step % cfg["save_interval"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"step_{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            # Upload checkpoint to HF in background
            if cfg.get("hf_repo"):
                import threading
                def _upload_ckpt(path, repo, step_num):
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        api.upload_file(
                            path_or_fileobj=path,
                            path_in_repo=f"checkpoints/step_{step_num}.pt",
                            repo_id=repo,
                            repo_type="model",
                        )
                        print(f"  Uploaded {path} to HF")
                    except Exception as e:
                        print(f"  HF upload failed: {e}")
                threading.Thread(target=_upload_ckpt, args=(ckpt_path, cfg["hf_repo"], step), daemon=True).start()

    # Save final checkpoint
    final_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": cfg["max_iters"],
        "config": cfg,
    }, final_path)

    # Upload final checkpoint to HF
    if cfg.get("hf_repo"):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=final_path,
                path_in_repo="checkpoints/final.pt",
                repo_id=cfg["hf_repo"],
                repo_type="model",
            )
            print(f"  Final checkpoint uploaded to HF: {cfg['hf_repo']}")
        except Exception as e:
            print(f"  HF upload failed: {e}")

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data, cfg)
    perplexity = math.exp(min(losses["val"], 20))

    elapsed = time.time() - train_start
    print(f"\n{'='*50}")
    print(f"  Training Complete!")
    print(f"{'='*50}")
    print(f"  Total time:       {fmt_time(elapsed)}")
    print(f"  Final train loss: {losses['train']:.4f}")
    print(f"  Final val loss:   {losses['val']:.4f}")
    print(f"  Final perplexity: {perplexity:.2f}")
    print(f"  Checkpoint saved: {final_path}")
    print(f"  Metrics saved:    {os.path.join(cfg['checkpoint_dir'], 'metrics.json')}")
    print(f"\n  Generate text with: python generate.py --checkpoint {final_path}")


if __name__ == "__main__":
    main()
