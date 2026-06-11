"""Shared training utilities used by 4_train.py (pretrain) and 6_finetune.py
(SFT): optimizer construction with correct weight-decay grouping, and the
learning-rate schedule (cosine or warmup-stable-decay).

Kept here so both stages stay in lock-step — a recipe fix applied once takes
effect everywhere instead of drifting between two copy-pasted loops.
"""

import math

import torch


def configure_optimizer(model, weight_decay, lr, betas=(0.9, 0.95), device="cpu"):
    """Build an AdamW optimizer with decay applied ONLY to matrix-shaped params.

    Weight decay on 1-D parameters (RMSNorm gains, biases) is a known
    pessimization — it pulls normalization scales toward zero for no benefit.
    The standard GPT recipe decays tensors with dim >= 2 (linear weights, the
    tied embedding) and exempts the rest. `named_parameters` already de-dupes
    the tied wte/lm_head tensor, so it is decayed once.

    betas default to (0.9, 0.95): every GPT-family recipe (GPT-2/3, LLaMA,
    OLMo) uses beta2=0.95 at this scale. The PyTorch default 0.999 adapts the
    second moment too slowly and correlates with late-training loss spikes.

    `fused=True` (CUDA only) collapses the param update into one kernel.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in no_decay)
    print(
        f"  Optimizer: AdamW betas={betas} | decayed params {n_decay:,} "
        f"in {len(decay)} tensors | no-decay {n_nodecay:,} in {len(no_decay)} tensors"
    )
    fused = device == "cuda"
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    except (RuntimeError, ValueError):
        # fused unsupported on this build/device — fall back.
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def load_optimizer_state(optimizer, state, *, where="checkpoint"):
    """Load optimizer state, tolerating a param-group-count mismatch.

    Pre-v2 checkpoints were saved with a single, ungrouped AdamW (one param
    group). The v2 optimizer has two groups (decay / no-decay), so a verbatim
    load_state_dict raises. When the shapes don't line up we keep the freshly
    built optimizer (a warm AdamW re-converges in a few hundred steps) instead
    of crashing a resume. Returns True if the state was loaded.
    """
    try:
        optimizer.load_state_dict(state)
        return True
    except (ValueError, KeyError) as e:
        print(
            f"  WARNING: could not load optimizer state from {where} "
            f"({type(e).__name__}: {e}). Continuing with a fresh optimizer "
            f"(likely a pre-v2, single-group checkpoint). Momentum re-warms "
            f"in a few hundred steps."
        )
        return False


def get_lr(step, cfg):
    """Learning-rate schedule: linear warmup, then either cosine decay or
    warmup-stable-decay (WSD), selected by cfg['lr_schedule'].

    cosine (default): warmup -> cosine from peak to min_lr across the whole run.
    wsd: warmup -> hold at peak -> decay to min_lr over only the final
        cfg['decay_frac'] (default 0.2) of steps. WSD lets you stop or extend a
        run at any point (the model is "ready" throughout the stable phase) and
        anneal on demand — useful when the horizon may change mid-run or for
        later continued-pretraining.
    """
    warmup = cfg["warmup_iters"]
    max_iters = cfg["max_iters"]
    peak = cfg["learning_rate"]
    floor = cfg.get("min_lr", 0.1 * peak)

    if step < warmup:
        return peak * step / max(warmup, 1)

    schedule = cfg.get("lr_schedule", "cosine")
    if schedule == "wsd":
        decay_frac = cfg.get("decay_frac", 0.2)
        decay_iters = max(1, int(decay_frac * max_iters))
        decay_start = max_iters - decay_iters
        if step < decay_start:
            return peak  # stable phase
        progress = (step - decay_start) / decay_iters
        progress = min(progress, 1.0)
        # 1 - sqrt decay (MiniCPM/Hu et al.): spends more steps near the floor
        # than cosine, which empirically lands a touch lower.
        return floor + (peak - floor) * (1.0 - math.sqrt(progress))

    # cosine
    decay_ratio = (step - warmup) / max(max_iters - warmup, 1)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return floor + coeff * (peak - floor)
