"""
ArmGPT Configuration
All hyperparameters in one place. Pick a preset or customize your own.
"""

import argparse

# --- Presets ---
# "tiny"   : runs on CPU in minutes, good for learning and debugging
# "small"  : default, trains in ~30 min on a single GPU (e.g. Colab T4)
# "medium" : better results, needs a good GPU (A100/V100), ~2 hours

PRESETS = {
    "tiny": dict(
        n_layer=1,
        n_head=2,
        n_embd=64,
        block_size=64,
        batch_size=32,
        max_iters=1000,
        learning_rate=1e-3,
        eval_interval=100,
    ),
    "small": dict(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        batch_size=64,
        max_iters=5000,
        learning_rate=1e-3,
        eval_interval=500,
    ),
    "medium": dict(
        n_layer=8,
        n_head=8,
        n_embd=512,
        block_size=512,
        batch_size=64,
        max_iters=10000,
        learning_rate=6e-4,
        eval_interval=500,
    ),
    "large": dict(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=512,
        batch_size=16,
        grad_accum_steps=4,  # effective batch = 16*4 = 64
        max_iters=20000,
        learning_rate=1e-4,
        eval_interval=1000,
    ),
    "xlarge": dict(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        block_size=1024,
        batch_size=24,
        grad_accum_steps=6,  # effective batch = 24*6 = 144
        max_iters=36000,
        learning_rate=3e-4,
        warmup_iters=2000,
        eval_interval=2000,
        save_interval=2000,
        sample_interval=2000,
    ),
    # Stage 2: fine-tuning on conversational data
    "finetune": dict(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        batch_size=32,
        max_iters=2000,
        learning_rate=3e-4,
        eval_interval=200,
    ),
}

# --- Default Config ---
# These are the "small" preset values. Override with --preset or CLI flags.

# model
n_layer = 6          # number of transformer blocks
n_head = 6           # number of attention heads
n_embd = 384         # embedding dimension (must be divisible by n_head)
block_size = 256     # context window length (in tokens/characters)
dropout = 0.2        # dropout rate for regularization

# training
batch_size = 64      # how many sequences to process at once
max_iters = 5000     # total training steps
learning_rate = 1e-3 # peak learning rate
warmup_iters = 100   # linear warmup steps
min_lr = 1e-4        # minimum learning rate after decay
weight_decay = 0.1   # AdamW weight decay
grad_clip = 1.0      # gradient clipping (0 = no clipping)
grad_accum_steps = 1 # gradient accumulation steps (effective batch = batch_size * this)

# evaluation and logging
eval_interval = 500  # evaluate every N steps
eval_iters = 200     # number of batches to average for eval loss
log_interval = 10    # print training loss every N steps
sample_interval = 500  # generate sample text every N steps
sample_length = 200  # how many characters/tokens to generate in samples

# checkpointing
checkpoint_dir = "checkpoints"
save_interval = 1000  # save checkpoint every N steps
resume_from = ""      # path to checkpoint to resume from

# data
data_dir = "data"
tokenizer = "char"   # "char" for Level 1, "bpe" for Level 2

# device (auto-detect)
device = "auto"      # "auto", "cpu", "cuda", or "mps"


def get_config():
    """Parse command-line arguments and return the final config as a dict."""
    parser = argparse.ArgumentParser(description="ArmGPT Training Config")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["tiny", "small", "medium", "large", "xlarge", "finetune"],
                        help="Use a preset configuration")
    # Allow overriding any config value from the command line
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["char", "bpe"])
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--sample_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace repo to upload checkpoints (e.g. edisimon/armgpt)")

    args = parser.parse_args()

    # Start with module-level defaults
    cfg = {k: v for k, v in globals().items()
           if not k.startswith("_") and isinstance(v, (int, float, str))}

    # Apply preset if specified
    if args.preset:
        cfg.update(PRESETS[args.preset])

    # Apply any explicit CLI overrides
    for key, val in vars(args).items():
        if val is not None and key != "preset":
            cfg[key] = val

    # Auto-detect device
    if cfg["device"] == "auto":
        import torch
        if torch.cuda.is_available():
            cfg["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg["device"] = "mps"
        else:
            cfg["device"] = "cpu"

    return cfg
