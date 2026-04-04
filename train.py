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


def load_data(data_dir):
    """Load pre-encoded training and validation data."""
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print("Run 'python data/download.py' and 'python data/prepare.py' first.")
        sys.exit(1)

    # Memory-map the files so we don't load everything into RAM
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
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
    # Pick random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Input: tokens from position i to i+block_size
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    # Target: tokens from position i+1 to i+block_size+1 (shifted by one)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    """Estimate average loss on train and validation data."""
    model.eval()
    results = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg["eval_iters"]):
            x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], cfg["device"])
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split_name] = sum(losses) / len(losses)
    model.train()
    return results


def get_lr(step, cfg):
    """Learning rate schedule: linear warmup then cosine decay."""
    # Warmup phase
    if step < cfg["warmup_iters"]:
        return cfg["learning_rate"] * step / cfg["warmup_iters"]
    # Decay phase
    decay_ratio = (step - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def main():
    cfg = get_config()

    # Print configuration
    print(f"\n{'='*50}")
    print(f"  ArmGPT Training")
    print(f"{'='*50}")
    print(f"  Device:      {cfg['device']}")
    print(f"  Model:       {cfg['n_layer']} layers, {cfg['n_head']} heads, {cfg['n_embd']} dim")
    print(f"  Block size:  {cfg['block_size']}")
    print(f"  Batch size:  {cfg['batch_size']}")
    print(f"  Max iters:   {cfg['max_iters']}")
    print(f"  Tokenizer:   {cfg['tokenizer']}")
    print(f"{'='*50}\n")

    # Load data and tokenizer
    train_data, val_data = load_data(cfg["data_dir"])
    tokenizer = load_tokenizer(cfg["data_dir"], cfg["tokenizer"])
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data:   {len(val_data):,} tokens")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    device = cfg["device"]
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
    print(f"\nStarting training from step {start_iter}...\n")
    model.train()
    t0 = time.time()

    for step in range(start_iter, cfg["max_iters"]):
        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get a batch and do forward + backward pass
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients to prevent explosions
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        optimizer.step()

        # Print training loss
        if step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            tokens_per_sec = cfg["batch_size"] * cfg["block_size"] / dt if dt > 0 else 0
            t0 = time.time()
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")

        # Evaluate and generate samples
        if step > 0 and step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            perplexity = math.exp(losses["val"])

            # Calculate accuracy on a validation batch
            x_val, y_val = get_batch(val_data, cfg["block_size"], cfg["batch_size"], device)
            with torch.no_grad():
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
            metrics["tokens_per_sec"].append(tokens_per_sec)

            # Save metrics
            metrics_path = os.path.join(cfg["checkpoint_dir"], "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Generate sample text
        if step > 0 and step % cfg["sample_interval"] == 0:
            model.eval()
            # Start with a few Armenian characters as seed
            seed_text = "Հայաստան"
            seed_ids = tokenizer.encode(seed_text)
            if len(seed_ids) == 0:
                seed_ids = [0]  # fallback
            context = torch.tensor([seed_ids], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=cfg["sample_length"],
                                       temperature=0.8, top_k=40)
            text = tokenizer.decode(generated[0].tolist())
            print(f"\n--- Sample (step {step}) ---")
            print(text)
            print("--- End sample ---\n")
            model.train()

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
    perplexity = math.exp(losses["val"])

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
