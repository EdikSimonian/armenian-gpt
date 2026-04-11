"""
Step 6: Fine-tune ArmGPT on conversational data (Stage 2 SFT).

Takes a base model produced by 4_train.py and fine-tunes it on
instruction/response pairs from data_chat/ (produced by
3_tokenize.py --qa).

Usage:
    python 6_finetune.py
    python 6_finetune.py --stage1_checkpoint checkpoints/final.pt
    python 6_finetune.py --max_iters 3000 --learning_rate 1e-4
    RESUME_CHAT_FROM=checkpoints_chat/chat_best.pt python 6_finetune.py

Inputs:
    data_chat/train_{char,bpe}.bin, val_{char,bpe}.bin, tokenizer_{char,bpe}.json
    checkpoints/final.pt   (Stage 1 base model, cold start only)

Outputs:
    checkpoints_chat/chat_step_*.pt   (periodic snapshots)
    checkpoints_chat/chat_best.pt     (best val loss so far)
    checkpoints_chat/chat_final.pt    (end-of-run)
    Uploaded to checkpoints/chat/ on the configured HF repo.

After running, use 8_chat.py to talk to your model:
    python 8_chat.py
"""

import os
import sys
import time
import json
import math
import queue
import threading
import numpy as np
import torch

from core.model import GPT
from core.config import get_config


# --- HF async upload helpers --------------------------------------------------
# Finetune checkpoints go to checkpoints/chat/ on the model repo so the HF
# Space loader can prefer them over the base pretraining checkpoints.
_HF_UPLOAD_REPO = os.environ.get("HF_UPLOAD_REPO", "edisimon/armgpt")
_HF_CHAT_TOKENIZER_UPLOADED = False  # one-shot flag (set from main thread only)

# Single-worker upload queue. Why not fire-and-forget threads?
# The original 6000-step run got SIGTERM'd by macOS jetsam at step 2000 because
# two 4 GB uploads (chat_step_02000.pt and a fresh chat_best.pt) went concurrent
# right when the model + activations + optimizer state were already eating most
# of MPS memory. Serializing through one worker caps memory at "1 upload at a
# time" no matter how often we save. The queue is unbounded; if backpressure
# becomes a problem we can add maxsize.
_upload_queue: "queue.Queue[tuple[str, str, str] | None]" = queue.Queue()
_upload_worker_started = False


def _upload_worker_loop():
    from huggingface_hub import HfApi
    api = HfApi()
    while True:
        item = _upload_queue.get()
        try:
            if item is None:
                # Sentinel — drain and exit
                return
            local_path, repo_path, commit_message = item
            try:
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=_HF_UPLOAD_REPO,
                    repo_type="model",
                    commit_message=commit_message,
                )
                dt = time.time() - t0
                size_mb = os.path.getsize(local_path) / 1e6 if os.path.exists(local_path) else 0
                print(f"  [hf-upload] OK  {repo_path}  ({size_mb:.0f} MB in {dt:.0f}s)",
                      flush=True)
            except Exception as e:
                print(f"  [hf-upload] FAIL {repo_path}: {e}", flush=True)
        finally:
            _upload_queue.task_done()


def _start_upload_worker_once():
    """Lazily spin up the upload worker on the first save."""
    global _upload_worker_started
    if _upload_worker_started:
        return
    t = threading.Thread(
        target=_upload_worker_loop,
        daemon=True,
        name="hf-upload-worker",
    )
    t.start()
    _upload_worker_started = True


def _hf_upload_bg(local_path, repo_path, commit_message):
    """Enqueue an upload for the single-worker thread. Returns immediately."""
    _start_upload_worker_once()
    _upload_queue.put((local_path, repo_path, commit_message))


def save_and_upload_chat_checkpoint(model, optimizer, step, cfg, local_name,
                                    repo_name):
    """Save a finetune checkpoint locally and queue its HF upload.

    Note: optimizer state is NOT included. SFT runs are short and restart-
    friendly, and including AdamW state doubles the file size, which was a
    contributing factor to the macOS jetsam SIGTERM at step 2000 of the
    original run (two ~4 GB files held concurrently in upload threads).

    On the FIRST call we also upload the chat tokenizer so the Space can
    decode the new <|user|>/<|assistant|>/<|end|> tokens that the base repo's
    tokenizer doesn't know about.
    """
    global _HF_CHAT_TOKENIZER_UPLOADED

    local_path = os.path.join(cfg["checkpoint_dir"], local_name)
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "config": cfg,
    }, local_path)
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"  [checkpoint] saved {local_path} ({size_gb:.2f} GB)", flush=True)

    # First upload also ships the chat tokenizer (once per run).
    if not _HF_CHAT_TOKENIZER_UPLOADED:
        from core import tokenizer_path as _tok_path
        chat_tok_path = _tok_path("data_chat", cfg["tokenizer"])
        if os.path.exists(chat_tok_path):
            _hf_upload_bg(
                chat_tok_path,
                f"data_chat/tokenizer_{cfg['tokenizer']}.json",
                f"Chat tokenizer (step {step})",
            )
            _HF_CHAT_TOKENIZER_UPLOADED = True

    _hf_upload_bg(
        local_path,
        f"checkpoints/chat/{repo_name}",
        f"Finetune checkpoint step {step}",
    )


def load_data(data_dir, tokenizer_type):
    """Load the chat training and validation data for the given tokenizer."""
    from core import bin_paths
    train_path, val_path = bin_paths(data_dir, tokenizer_type)

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print(f"Run 'python 3_tokenize.py --qa --tokenizer {tokenizer_type}' first.")
        sys.exit(1)

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    return train_data, val_data


def load_tokenizer(data_dir, tokenizer_type):
    """Load the extended tokenizer with special chat tokens."""
    from core import load_tokenizer as _load, tokenizer_path
    tok_path = tokenizer_path(data_dir, tokenizer_type)
    if not os.path.exists(tok_path):
        print(f"Error: {tok_path} not found! "
              f"Run 3_tokenize.py --qa --tokenizer {tokenizer_type} first.")
        sys.exit(1)
    return _load(data_dir, tokenizer_type)


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

    # Resume mode: if RESUME_CHAT_FROM env var points at a chat checkpoint,
    # we skip the base-model load + vocab grafting and continue from where
    # the previous run stopped. Set via env var (not CLI flag) to avoid
    # touching config.py's argparse.
    resume_chat_from = os.environ.get("RESUME_CHAT_FROM", "").strip() or None

    # Stage 1 checkpoint path (only used in non-resume mode)
    stage1_ckpt = cfg.get("resume_from", "") or "checkpoints/final.pt"

    print(f"\n{'='*50}")
    print(f"  ArmGPT Stage 2: Fine-tuning for Chat")
    print(f"{'='*50}")
    print(f"  Device:        {cfg['device']}")
    if resume_chat_from:
        print(f"  RESUMING from: {resume_chat_from}")
    else:
        print(f"  Stage 1 model: {stage1_ckpt}")
    print(f"  Chat data:     {chat_data_dir}/")
    print(f"  Max iters:     {cfg['max_iters']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"{'='*50}\n")

    # Load chat data and tokenizer
    train_data, val_data = load_data(chat_data_dir, cfg["tokenizer"])
    tokenizer = load_tokenizer(chat_data_dir, cfg["tokenizer"])
    print(f"Chat train data: {len(train_data):,} tokens")
    print(f"Chat val data:   {len(val_data):,} tokens")
    print(f"Vocab size:      {tokenizer.vocab_size} (with special tokens)")

    device = cfg["device"]
    start_step = 0

    if resume_chat_from:
        # ---- RESUME PATH ------------------------------------------------------
        # The chat checkpoint already has the right vocab and weights — just
        # rebuild the model from its state dict and pick up the loop where
        # the previous run stopped.
        if not os.path.exists(resume_chat_from):
            print(f"\nError: chat checkpoint not found at {resume_chat_from}")
            sys.exit(1)
        print(f"\nResuming from chat checkpoint {resume_chat_from}...")
        ckpt = torch.load(resume_chat_from, map_location=device, weights_only=False)
        state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
            print("  Stripped _orig_mod. prefix from chat state dict")

        # Infer architecture from state dict — don't trust ckpt config which
        # may carry the training-time block_size (256) rather than the model's
        # RoPE buffer size (1024).
        vocab_size = int(state["transformer.wte.weight"].shape[0])
        n_embd = int(state["transformer.wte.weight"].shape[1])
        n_layer = max(int(k.split(".")[2]) for k in state if k.startswith("transformer.blocks.")) + 1
        rope_cos = state["transformer.blocks.0.attn.rope_cos"]
        model_block_size = int(rope_cos.shape[0])
        head_dim = int(rope_cos.shape[1]) * 2  # RoPE uses half the head_dim
        n_head = n_embd // head_dim
        print(f"  Inferred arch: vocab={vocab_size} n_layer={n_layer} n_head={n_head} "
              f"n_embd={n_embd} model_block_size={model_block_size}")

        if vocab_size != tokenizer.vocab_size:
            print(f"  WARNING: tokenizer vocab {tokenizer.vocab_size} != ckpt vocab {vocab_size}")

        model = GPT(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=model_block_size,
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(state)

        # Resume the step counter from the checkpoint so save filenames continue
        # the same numbering. The optimizer is fresh either way — old runs may
        # have stored optimizer state but we don't load it (see save function).
        start_step = int(ckpt.get("step") or 0)
        print(f"  Resuming step counter at {start_step}")

        # Mirror cfg fields the way the non-resume path would set them so the
        # rest of main() (saves, eval, sample gen) sees consistent metadata.
        cfg["vocab_size"] = vocab_size
        cfg["n_layer"] = n_layer
        cfg["n_head"] = n_head
        cfg["n_embd"] = n_embd
        cfg["model_block_size"] = model_block_size
        if cfg["block_size"] > model_block_size:
            cfg["block_size"] = model_block_size
    else:
        # ---- COLD START PATH (load base + graft chat tokens) -----------------
        if not os.path.exists(stage1_ckpt):
            print(f"\nError: Stage 1 checkpoint not found at {stage1_ckpt}")
            print("Train Stage 1 first with: python 4_train.py --preset small")
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
        # Strip torch.compile() prefix if present.
        if any(k.startswith("_orig_mod.") for k in stage1_state):
            stage1_state = {k.removeprefix("_orig_mod."): v for k, v in stage1_state.items()}
            print("  Stripped _orig_mod. prefix from state dict (torch.compile checkpoint)")
        model_state = model.state_dict()

        for key in stage1_state:
            if key in model_state:
                old_shape = stage1_state[key].shape
                new_shape = model_state[key].shape
                if old_shape == new_shape:
                    model_state[key] = stage1_state[key]
                elif len(old_shape) == 2 and old_shape[1] == new_shape[1]:
                    model_state[key][:old_shape[0]] = stage1_state[key]
                elif len(old_shape) == 1 and old_shape[0] < new_shape[0]:
                    model_state[key][:old_shape[0]] = stage1_state[key]

        model.load_state_dict(model_state)
        print(f"  Loaded Stage 1 weights (vocab: {old_vocab_size} -> {new_vocab_size})")

        # Use Stage 1 model architecture in config so 8_chat.py can rebuild the model.
        cfg["vocab_size"] = new_vocab_size
        cfg["n_layer"] = stage1_cfg["n_layer"]
        cfg["n_head"] = stage1_cfg["n_head"]
        cfg["n_embd"] = stage1_cfg["n_embd"]
        cfg["model_block_size"] = stage1_cfg["block_size"]
        if cfg["block_size"] > stage1_cfg["block_size"]:
            cfg["block_size"] = stage1_cfg["block_size"]

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

    # How often to snapshot a chat checkpoint and kick off an HF upload.
    # Defaults to 1000 but honors cfg["save_interval"] if the preset/CLI set it.
    chat_save_interval = cfg.get("save_interval") or 1000

    # Best-checkpoint tracking: save chat_best.pt whenever val loss improves,
    # so we always have a safety net even if the periodic / final checkpoints
    # drift past the bottom of the val loss curve.
    best_val_loss = float("inf")

    # Training loop
    print(f"\nStarting fine-tuning...")
    if start_step > 0:
        print(f"  RESUMING at step {start_step}, will run through step {cfg['max_iters']}")
    print(f"  Periodic chat checkpoints every {chat_save_interval} steps → "
          f"checkpoints/chat/chat_step_NNNNN.pt on {_HF_UPLOAD_REPO}")
    print(f"  Best checkpoint tracked on val loss → "
          f"checkpoints/chat/chat_best.pt on {_HF_UPLOAD_REPO}")
    print(f"  Uploads serialized through one worker thread (no concurrent 4 GB transfers)\n")
    model.train()
    t0 = time.time()

    for step in range(start_step, cfg["max_iters"]):
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
            print(f"  Best val so far: {best_val_loss:.4f}")
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

            # Best-checkpoint save — overwrite chat_best.pt whenever val
            # improves. This is in addition to the periodic step snapshots,
            # so if training drifts past the bottom we still have a safety
            # net. The upload to HF is fire-and-forget.
            if losses["val"] < best_val_loss:
                prev = best_val_loss
                best_val_loss = losses["val"]
                print(f"  ✓ New best val loss {best_val_loss:.4f} "
                      f"(prev {prev:.4f}), saving chat_best.pt", flush=True)
                save_and_upload_chat_checkpoint(
                    model, optimizer, step, cfg,
                    local_name="chat_best.pt", repo_name="chat_best.pt",
                )

            # Generate a sample response
            model.eval()
            sample_prompt = f"<|user|>Ի՞նչ է Հայաստանը:<|end|><|assistant|>"
            prompt_ids = tokenizer.encode(sample_prompt)
            if prompt_ids:
                context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                end_ids = tokenizer.encode("<|end|>")
                end_token_id = end_ids[0] if end_ids else None
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

        # Periodic snapshot + background HF upload (after eval so we save
        # with the post-eval loss visible in logs).
        if step > 0 and step % chat_save_interval == 0:
            name = f"chat_step_{step:05d}.pt"
            save_and_upload_chat_checkpoint(
                model, optimizer, step, cfg,
                local_name=name, repo_name=name,
            )

    # Save final Stage 2 checkpoint and upload as chat_final.pt
    save_and_upload_chat_checkpoint(
        model, optimizer, cfg["max_iters"], cfg,
        local_name="chat_final.pt", repo_name="chat_final.pt",
    )
    final_path = os.path.join(cfg["checkpoint_dir"], "chat_final.pt")

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

    # Drain the upload queue before exiting so the final checkpoint actually
    # finishes its upload (otherwise the daemon worker thread dies with the
    # main process and the last checkpoint never lands on HF).
    if _upload_worker_started:
        print(f"\n  Waiting for pending HF uploads to drain...", flush=True)
        _upload_queue.put(None)  # sentinel
        # Block until the worker has popped every queued item AND processed
        # the sentinel. timeout is generous: 4 GB at 30 MB/s ≈ 130s, ×N pending.
        _upload_queue.join()
        print(f"  All uploads complete.", flush=True)

    print(f"\n  Chat with your model: python 8_chat.py")


if __name__ == "__main__":
    main()
