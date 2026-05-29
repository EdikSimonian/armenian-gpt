"""
eval_bpb.py — Tokenizer-independent bits-per-byte (bpb) evaluation.

bpb normalizes cross-entropy by the UTF-8 byte length of the held-out text, not
by tokens. That makes scores comparable across different tokenizers (BPE vocab
size, char-level, etc.) and across model versions trained with different
tokenizers. It's the right metric once val is no longer a clean held-out set —
provided the text passed in is genuinely unseen at training time.

Usage:
    # Evaluate one checkpoint:
    python eval_bpb.py --checkpoint checkpoints/step_4000.pt --text eval/heldout.txt

    # Sweep all checkpoints in a directory, log to JSON for charting:
    python eval_bpb.py --checkpoint-dir checkpoints/ --text eval/heldout.txt \
        --log eval/bpb_log.json

    # Force CPU (e.g. on the training pod, to avoid GPU contention with train.py):
    python eval_bpb.py --checkpoint checkpoints/step_4000.pt --text eval/heldout.txt \
        --device cpu

Held-out text recommendations:
    The text MUST be unseen by the model. Safe sources:
      - Armenian Wikipedia articles created/heavily-edited AFTER the training
        data was scraped (check the snapshot date of edisimon/armenian-clean-text).
      - Recent Armenian news articles from a publication not in the corpus.
      - Hand-written prose the user wrote themselves.
    Test_bpe.bin is NOT a clean source — it's sequentially adjacent to
    val_train (which is now in training pool). It still gives a useful
    in-distribution signal but bpb is most informative on truly fresh text.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F

from core import load_tokenizer
from core.model import GPT


def pick_dtype(device):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def strip_compile_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        return {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


def load_model(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, ckpt["step"], cfg


@torch.no_grad()
def compute_bpb(model, tokens, block_size, n_bytes, device, amp_dtype):
    """Score every token after the first via non-overlapping windows.

    For checkpoint-vs-checkpoint comparison the warmup bias (first ~64 tokens
    of each window have less context) is constant across runs and cancels out.
    """
    ids = torch.tensor(tokens, dtype=torch.long, device=device)
    n_tokens = ids.numel()
    if n_tokens < 2:
        raise ValueError("text too short — needs >= 2 tokens")

    total_nll_nats = 0.0
    scored = 0
    use_amp = device == "cuda"

    pos = 0
    while pos + 1 < n_tokens:
        end = min(pos + block_size, n_tokens)
        chunk = ids[pos:end]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits, _ = model(x, y)
        else:
            logits, _ = model(x, y)

        # Sum NLL, don't average. cross_entropy with reduction="sum" works in
        # fp32 internally even on bf16 logits — gather + log_softmax is
        # equivalent and a bit more explicit.
        nll = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_nll_nats += nll.item()
        scored += y.numel()
        pos = end

    total_bits = total_nll_nats / math.log(2)
    bpb = total_bits / n_bytes
    return {
        "bpb": bpb,
        "nll_nats_total": total_nll_nats,
        "tokens_scored": scored,
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "ppl_per_token": math.exp(total_nll_nats / scored),
    }


def main():
    p = argparse.ArgumentParser(description="Bits-per-byte evaluation harness")
    p.add_argument("--checkpoint", type=str, help="Single checkpoint path")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing step_*.pt — evaluates all (skipping ones already in --log)",
    )
    p.add_argument("--text", type=str, required=True, help="UTF-8 held-out text file")
    p.add_argument("--data-dir", type=str, default="data", help="Tokenizer dir")
    p.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--log",
        type=str,
        help="JSON file to append/update with one entry per (step, text_path, tokenizer)",
    )
    args = p.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    amp_dtype = pick_dtype(device)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # Load held-out text
    with open(args.text, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    n_bytes = len(raw)
    print(f"Held-out text: {args.text}  ({n_bytes:,} bytes, {len(text):,} chars)")

    # Tokenizer
    tokenizer = load_tokenizer(args.data_dir, args.tokenizer)
    print(f"Tokenizer:     {args.tokenizer} (vocab={tokenizer.vocab_size})")

    # Tokenize once — the same tokens are scored against every checkpoint
    tokens = tokenizer.encode(text)
    print(f"Tokens:        {len(tokens):,}")

    # Resolve checkpoint list
    ckpts = []
    if args.checkpoint:
        ckpts.append(args.checkpoint)
    if args.checkpoint_dir:
        ckpts.extend(
            sorted(
                glob.glob(os.path.join(args.checkpoint_dir, "step_*.pt")),
                key=lambda f: int(re.findall(r"step_(\d+)\.pt", f)[-1]),
            )
        )
    if not ckpts:
        sys.exit("Pass --checkpoint or --checkpoint-dir")

    # Load existing log so we can skip already-evaluated (step, text, tokenizer)
    results = []
    seen = set()
    if args.log and os.path.exists(args.log):
        with open(args.log) as f:
            results = json.load(f)
        seen = {(r["step"], r["text_path"], r["tokenizer"]) for r in results}

    print(f"\nDevice: {device} ({amp_dtype})\n")
    print(f"{'step':>7} | {'bpb':>7} | {'ppl/tok':>9} | {'tokens':>10} | checkpoint")
    print("-" * 80)

    for ckpt_path in ckpts:
        step = (
            int(re.findall(r"step_(\d+)\.pt", ckpt_path)[-1])
            if "step_" in ckpt_path
            else -1
        )
        key = (step, args.text, args.tokenizer)
        if key in seen:
            print(f"{step:>7} |  (cached — already in log)")
            continue

        model, model_step, _ = load_model(ckpt_path, tokenizer.vocab_size, device)
        # Trust the checkpoint's internal step over the filename
        step = model_step
        out = compute_bpb(
            model,
            tokens,
            model.block_size,
            n_bytes,
            device,
            amp_dtype,
        )
        out["step"] = step
        out["checkpoint"] = os.path.abspath(ckpt_path)
        out["text_path"] = args.text
        out["tokenizer"] = args.tokenizer

        print(
            f"{step:>7} | {out['bpb']:>7.4f} | {out['ppl_per_token']:>9.2f} | "
            f"{out['tokens_scored']:>10,} | {ckpt_path}"
        )
        results.append(out)

        # Release model before next ckpt — these are 11 GB each on giant
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.log:
        results.sort(key=lambda r: (r["text_path"], r["tokenizer"], r["step"]))
        with open(args.log, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nLogged to {args.log}")


if __name__ == "__main__":
    main()
