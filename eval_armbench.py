"""
eval_armbench.py — multiple-choice accuracy via log-probability scoring.

A capability metric that complements bits-per-byte: for each question it scores
every answer choice by the model's length-normalized log-likelihood and picks
the most probable one. Because it only reads log-probs (no generation, no chat
format), it works on BASE checkpoints as well as chat ones — so you can track
real knowledge/reasoning across pretraining, not just perplexity.

Reports two standard numbers (lm-eval-harness style):
    acc       — argmax of the summed option log-prob
    acc_norm  — argmax of the PER-TOKEN option log-prob (length-normalized;
                usually the fairer number when options differ in length)

Input: a JSON list (e.g. data/text/finetune/armbench_eval.json from
1_download.py --qa). Each item is flexible about field names:
    stem:    "question" | "instruction" | "prompt"  (+ optional "input"/"context")
    choices: "options" | "choices"  (list of strings)
    gold:    "answer" | "label" | "correct"  (int index, a letter Ա/Բ/.. or
             A/B/.., or the exact text of the correct choice)

Usage:
    python eval_armbench.py --checkpoint checkpoints/step_40000.pt \
        --data data/text/finetune/armbench_eval.json --tokenizer bpe
    python eval_armbench.py --checkpoint-dir checkpoints/ \
        --data data/text/finetune/armbench_eval.json --log eval/armbench_log.json
"""

import argparse
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from core import load_tokenizer
from core.model import GPT

# Armenian and Latin option letters -> index.
_ARM_LETTERS = "ԱԲԳԴԵԶԷԸԹԺ"
_LAT_LETTERS = "ABCDEFGHIJ"


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    """Build the model from a checkpoint, inferring arch (incl. vocab and
    qk_norm) from the weights so chat and base checkpoints both load."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    vocab_size = int(state["transformer.wte.weight"].shape[0])
    n_embd = int(state["transformer.wte.weight"].shape[1])
    n_layer = (
        max(int(k.split(".")[2]) for k in state if k.startswith("transformer.blocks."))
        + 1
    )
    rope = state["transformer.blocks.0.attn.rope_cos"]
    block_size = int(rope.shape[0])
    head_dim = int(rope.shape[1]) * 2
    n_head = n_embd // head_dim
    qk_norm = any("attn.q_norm.weight" in k for k in state)

    model = GPT(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=0.0,
        qk_norm=qk_norm,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, int(ckpt.get("step", -1)), block_size


def _stem(item):
    stem = item.get("question") or item.get("instruction") or item.get("prompt") or ""
    ctx = item.get("input") or item.get("context") or ""
    return (stem + ("\n" + ctx if ctx else "")).strip()


def _choices(item):
    return item.get("options") or item.get("choices") or []


def _gold_index(item, choices):
    """Resolve the correct-choice index from int / letter / exact text, or -1."""
    g = item.get("answer", item.get("label", item.get("correct")))
    if g is None:
        return -1
    if isinstance(g, bool):
        return -1
    if isinstance(g, int):
        return g if 0 <= g < len(choices) else -1
    s = str(g).strip()
    if len(s) == 1:
        if s in _ARM_LETTERS:
            i = _ARM_LETTERS.index(s)
            return i if i < len(choices) else -1
        if s.upper() in _LAT_LETTERS:
            i = _LAT_LETTERS.index(s.upper())
            return i if i < len(choices) else -1
    if s.isdigit():
        i = int(s)
        return i if 0 <= i < len(choices) else -1
    norm = re.sub(r"\s+", " ", s).strip().lower()
    for i, c in enumerate(choices):
        if re.sub(r"\s+", " ", str(c)).strip().lower() == norm:
            return i
    return -1


@torch.no_grad()
def _option_logprob(model, tokenizer, prompt_ids, option_text, block_size, device):
    """Summed and per-token log-prob the model assigns to `option_text`
    conditioned on the prompt. Left-truncates to the context window."""
    opt_ids = tokenizer.encode(option_text)
    if not opt_ids:
        return -1e9, -1e9
    ids = prompt_ids + opt_ids
    if len(ids) > block_size:
        # keep the option and as much trailing prompt as fits
        ids = ids[-block_size:]
        n_opt = min(len(opt_ids), block_size - 1)
    else:
        n_opt = len(opt_ids)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    tgt = torch.tensor(ids[1:], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    tok_lp = logp[torch.arange(len(tgt), device=device), tgt]
    opt_lp = tok_lp[-n_opt:]
    return float(opt_lp.sum()), float(opt_lp.mean())


@torch.no_grad()
def evaluate(model, tokenizer, items, block_size, device):
    n = 0
    correct = correct_norm = 0
    skipped = 0
    for item in items:
        choices = _choices(item)
        gold = _gold_index(item, choices)
        if len(choices) < 2 or gold < 0:
            skipped += 1
            continue
        stem = _stem(item)
        prompt_ids = tokenizer.encode(stem + "\n")
        sums, norms = [], []
        for c in choices:
            s, m = _option_logprob(
                model, tokenizer, prompt_ids, " " + str(c), block_size, device
            )
            sums.append(s)
            norms.append(m)
        pred = max(range(len(sums)), key=lambda i: sums[i])
        pred_norm = max(range(len(norms)), key=lambda i: norms[i])
        correct += int(pred == gold)
        correct_norm += int(pred_norm == gold)
        n += 1
    return {
        "n": n,
        "skipped": skipped,
        "acc": correct / n if n else 0.0,
        "acc_norm": correct_norm / n if n else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Multiple-choice log-prob accuracy")
    p.add_argument("--checkpoint", type=str)
    p.add_argument("--checkpoint-dir", type=str, help="Evaluate all step_*.pt")
    p.add_argument("--data", type=str, required=True, help="MCQ JSON file")
    p.add_argument("--data-dir", type=str, default="data", help="Tokenizer dir")
    p.add_argument("--tokenizer", type=str, default="bpe", choices=["bpe", "char"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0, help="Eval only the first N items")
    p.add_argument("--log", type=str, help="Append results to this JSON file")
    args = p.parse_args()

    device = pick_device(args.device)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    with open(args.data, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]
    print(f"MCQ items: {len(items):,}  ({args.data})")

    tokenizer = load_tokenizer(args.data_dir, args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size})")

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

    results = []
    if args.log and os.path.exists(args.log):
        with open(args.log) as f:
            results = json.load(f)

    print(f"\nDevice: {device}\n")
    print(f"{'step':>8} | {'acc':>7} | {'acc_norm':>8} | {'n':>5} | checkpoint")
    print("-" * 72)
    for ckpt_path in ckpts:
        model, step, block_size = load_model(ckpt_path, device)
        out = evaluate(model, tokenizer, items, block_size, device)
        out["step"] = step
        out["checkpoint"] = os.path.abspath(ckpt_path)
        out["data"] = args.data
        print(
            f"{step:>8} | {out['acc']:>7.3f} | {out['acc_norm']:>8.3f} | "
            f"{out['n']:>5} | {os.path.basename(ckpt_path)}"
        )
        results.append(out)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.log:
        results.sort(key=lambda r: (r.get("data", ""), r.get("step", 0)))
        with open(args.log, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nLogged to {args.log}")


if __name__ == "__main__":
    main()
