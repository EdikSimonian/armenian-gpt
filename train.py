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

    # Load data into RAM as a contiguous torch tensor for fast batch creation
    # This avoids slow memmap access on every batch (especially on Windows)
    print("Loading data into memory...")
    train_np = np.fromfile(train_path, dtype=np.uint16)
    val_np = np.fromfile(val_path, dtype=np.uint16)
    train_data = torch.from_numpy(train_np.astype(np.int64))
    val_data = torch.from_numpy(val_np.astype(np.int64))
    print(f"  Train: {len(train_data):,} tokens ({len(train_data)*2/1024/1024:.0f} MB)")
    print(f"  Val:   {len(val_data):,} tokens ({len(val_data)*2/1024/1024:.0f} MB)")
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
    """Grab a random batch of sequences from the data. Fast, no Python loops."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)
    return x, y


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


def main():
    cfg = get_config()
    device = cfg["device"]
    use_amp = device == "cuda"

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
    tokens_per_step = cfg["batch_size"] * cfg["block_size"]
    print(f"\nStarting training from step {start_iter}...\n")
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_iter, cfg["max_iters"]):
        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get a batch and do forward + backward pass with mixed precision
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, loss = model(x, y)

        # Accumulate loss without syncing GPU (fast)
        running_loss += loss.detach()

        # Backward pass with gradient scaling
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
            elapsed = time.time() - t0
            t0 = time.time()

            # ETA calculation
            steps_remaining = cfg["max_iters"] - step
            secs_per_step = dt / steps_done
            eta_secs = int(steps_remaining * secs_per_step)
            eta_h, eta_m, eta_s = eta_secs // 3600, (eta_secs % 3600) // 60, eta_secs % 60

            print(f"step {step:5d}/{cfg['max_iters']} | loss {avg_loss:.4f} | "
                  f"lr {lr:.2e} | {tps:,.0f} tok/s | "
                  f"eta {eta_h:02d}:{eta_m:02d}:{eta_s:02d}")

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
            print(text)
            print("--- End sample ---\n")
            model.train()
            t0 = time.time()
            running_loss = 0.0

        # Save checkpoint
        if step > 0 and step % cfg["save_interval"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"step_{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": cfg["max_iters"],
        "config": cfg,
    }, final_path)

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data, cfg)
    perplexity = math.exp(min(losses["val"], 20))

    print(f"\n{'='*50}")
    print(f"  Training Complete!")
    print(f"{'='*50}")
    print(f"  Final train loss: {losses['train']:.4f}")
    print(f"  Final val loss:   {losses['val']:.4f}")
    print(f"  Final perplexity: {perplexity:.2f}")
    print(f"  Checkpoint saved: {final_path}")
    print(f"  Metrics saved:    {os.path.join(cfg['checkpoint_dir'], 'metrics.json')}")
    print(f"\n  Generate text with: python generate.py --checkpoint {final_path}")


if __name__ == "__main__":
    main()
