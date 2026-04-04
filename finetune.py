"""
Stage 2: Fine-tune ArmGPT on Conversational Data

This script takes a base model trained on Wikipedia (Stage 1) and fine-tunes
it on instruction/response pairs so it can answer questions like a chatbot.

This is how real AI assistants like ChatGPT are made:
    Stage 1: Train on lots of text (Wikipedia) -> learns language patterns
    Stage 2: Fine-tune on conversations (this script) -> learns to be helpful

Usage:
    python finetune.py
    python finetune.py --stage1_checkpoint checkpoints/final.pt
    python finetune.py --max_iters 3000 --learning_rate 1e-4

After running, use chat.py to talk to your model:
    python chat.py
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
    """Load the chat training and validation data."""
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print("Run 'python data/prepare_chat.py' first.")
        sys.exit(1)

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def load_tokenizer(data_dir):
    """Load the extended tokenizer with special chat tokens."""
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"Error: {tok_path} not found! Run data/prepare_chat.py first.")
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
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
    if step < cfg["warmup_iters"]:
        return cfg["learning_rate"] * step / max(cfg["warmup_iters"], 1)
    decay_ratio = (step - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def main():
    # Parse config — default to finetune preset
    sys.argv = sys.argv or ["finetune.py"]
    # Insert --preset finetune as default if no preset specified
    if "--preset" not in sys.argv:
        sys.argv.extend(["--preset", "finetune"])
    cfg = get_config()

    # Override data_dir for chat data
    chat_data_dir = cfg.get("data_dir", "data_chat")
    if chat_data_dir == "data":
        chat_data_dir = "data_chat"

    # Stage 1 checkpoint path
    stage1_ckpt = cfg.get("resume_from", "") or "checkpoints/final.pt"

    print(f"\n{'='*50}")
    print(f"  ArmGPT Stage 2: Fine-tuning for Chat")
    print(f"{'='*50}")
    print(f"  Device:        {cfg['device']}")
    print(f"  Stage 1 model: {stage1_ckpt}")
    print(f"  Chat data:     {chat_data_dir}/")
    print(f"  Max iters:     {cfg['max_iters']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"{'='*50}\n")

    # Load chat data and tokenizer
    train_data, val_data = load_data(chat_data_dir)
    tokenizer = load_tokenizer(chat_data_dir)
    print(f"Chat train data: {len(train_data):,} tokens")
    print(f"Chat val data:   {len(val_data):,} tokens")
    print(f"Vocab size:      {tokenizer.vocab_size} (with special tokens)")

    device = cfg["device"]

    # Load Stage 1 checkpoint
    if not os.path.exists(stage1_ckpt):
        print(f"\nError: Stage 1 checkpoint not found at {stage1_ckpt}")
        print("Train Stage 1 first with: python train.py --preset small")
        sys.exit(1)

    print(f"\nLoading Stage 1 model from {stage1_ckpt}...")
    checkpoint = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    stage1_cfg = checkpoint["config"]
    old_vocab_size = stage1_cfg.get("vocab_size", None)

    # Create new model with extended vocab size
    new_vocab_size = tokenizer.vocab_size
    model = GPT(
        vocab_size=new_vocab_size,
        n_layer=stage1_cfg["n_layer"],
        n_head=stage1_cfg["n_head"],
        n_embd=stage1_cfg["n_embd"],
        block_size=stage1_cfg["block_size"],
        dropout=cfg["dropout"],
    ).to(device)

    # Load Stage 1 weights into the new model
    # The embedding and output head are larger now (new special tokens)
    # so we copy the old weights and leave new token embeddings random
    stage1_state = checkpoint["model"]
    model_state = model.state_dict()

    for key in stage1_state:
        if key in model_state:
            old_shape = stage1_state[key].shape
            new_shape = model_state[key].shape
            if old_shape == new_shape:
                model_state[key] = stage1_state[key]
            elif len(old_shape) == 2 and old_shape[1] == new_shape[1]:
                # Embedding or output head: copy old rows, keep new rows random
                model_state[key][:old_shape[0]] = stage1_state[key]
            elif len(old_shape) == 1 and old_shape[0] < new_shape[0]:
                model_state[key][:old_shape[0]] = stage1_state[key]

    model.load_state_dict(model_state)
    print(f"  Loaded Stage 1 weights (vocab: {old_vocab_size} -> {new_vocab_size})")

    # Save vocab_size in config for later use
    cfg["vocab_size"] = new_vocab_size

    # Create optimizer (fresh — don't reuse Stage 1 optimizer state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Create checkpoint directory
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Metrics tracking
    metrics = {"steps": [], "train_loss": [], "val_loss": [],
               "perplexity": [], "tokens_per_sec": []}

    # Training loop
    print(f"\nStarting fine-tuning...\n")
    model.train()
    t0 = time.time()

    for step in range(cfg["max_iters"]):
        # Update learning rate
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

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

        # Evaluate
        if step > 0 and step % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg)
            perplexity = math.exp(losses["val"])

            print(f"\n{'='*50}")
            print(f"  Step {step} Evaluation")
            print(f"  Train loss:   {losses['train']:.4f}")
            print(f"  Val loss:     {losses['val']:.4f}")
            print(f"  Perplexity:   {perplexity:.2f}")
            print(f"{'='*50}")

            metrics["steps"].append(step)
            metrics["train_loss"].append(losses["train"])
            metrics["val_loss"].append(losses["val"])
            metrics["perplexity"].append(perplexity)
            metrics["tokens_per_sec"].append(tokens_per_sec)

            # Save metrics
            metrics_path = os.path.join(cfg["checkpoint_dir"], "chat_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Generate a sample response
            model.eval()
            sample_prompt = f"<|user|>Ի՞նdelays է Հdelays:<|end|><|assistant|>"
            prompt_ids = tokenizer.encode(sample_prompt)
            if prompt_ids:
                context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                end_token_id = tokenizer.stoi.get("<|end|>")
                stop = {end_token_id} if end_token_id is not None else None
                generated = model.generate(context, max_new_tokens=200,
                                           temperature=0.7, top_k=40,
                                           stop_tokens=stop)
                text = tokenizer.decode(generated[0].tolist())
                # Clean up special tokens for display
                text = text.replace("<|user|>", "\nUser: ")
                text = text.replace("<|assistant|>", "\nArmGPT: ")
                text = text.replace("<|end|>", "")
                print(f"\n--- Sample (step {step}) ---")
                print(text.strip())
                print("--- End sample ---\n")
            model.train()

    # Save final Stage 2 checkpoint
    final_path = os.path.join(cfg["checkpoint_dir"], "chat_final.pt")
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
    print(f"  Fine-tuning Complete!")
    print(f"{'='*50}")
    print(f"  Final train loss: {losses['train']:.4f}")
    print(f"  Final val loss:   {losses['val']:.4f}")
    print(f"  Final perplexity: {perplexity:.2f}")
    print(f"  Checkpoint saved: {final_path}")
    print(f"\n  Chat with your model: python chat.py")


if __name__ == "__main__":
    main()
