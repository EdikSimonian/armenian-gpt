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
        save_interval=1000,  # snapshot every 1000 steps for tighter HF upload cadence
        sample_interval=2000,
    ),
    # ~600 M params, Chinchilla-sized for ~11 B Armenian tokens (66 GB of text).
    # Tuned for a single H200 141 GB: dim 1280 × 28 layers × 2048 context fits
    # comfortably with batch 16 + grad_accum 8 (effective 128). At effective
    # 262k tokens/step × 42k steps = ~11 B tokens seen, matching the corpus.
    # Expected ~25 h on one H200 at ~40% MFU in BF16.
    "xxlarge": dict(
        n_layer=28,
        n_head=20,
        n_embd=1280,
        block_size=2048,
        batch_size=16,
        grad_accum_steps=8,  # effective batch = 16*8 = 128
        max_iters=42000,
        learning_rate=2.5e-4,
        warmup_iters=2000,
        eval_interval=2000,
        save_interval=1000,
        sample_interval=2000,
    ),
    # xxlarge trained for 4 epochs over the same 11 B-token Armenian corpus.
    # Per Muennighoff et al. (data-constrained scaling), 4 epochs yields
    # ~2.3× effective unique tokens — the practical sweet spot before
    # repeated-data returns flatten hard. Expected loss ~2.54 nats (~18%
    # lower perplexity than single-epoch xxlarge). Expected ~100 h on one
    # H200 at ~40% MFU in BF16 (~$350 at $3.50/hr).
    "xxlarge_4epoch": dict(
        n_layer=28,
        n_head=20,
        n_embd=1280,
        block_size=2048,
        batch_size=16,
        grad_accum_steps=8,
        max_iters=168000,    # 4 × 42000
        learning_rate=2.5e-4,
        warmup_iters=2000,   # 1.2% of schedule — plenty for this scale
        eval_interval=4000,
        save_interval=2000,  # 84 HF uploads across the run vs 168 at 1000
        sample_interval=4000,
    ),
    # ~1.0 B params. Bigger than xxlarge — approaches the data ceiling
    # for the ~16 B-token Armenian corpus. Chinchilla-optimal for 1 B is
    # 20 B tokens, so with max_iters=84000 × eff_batch 128 × ctx 2048 =
    # ~22 B tokens seen (~22 tokens/param), we're ~10% over Chinchilla-
    # optimal — ideal for quality without wasting compute.
    #
    # Needs ≥80 GB VRAM for training with 8-bit AdamW. Tight on 48 GB
    # (A40/A6000) even with gradient checkpointing. Sweet spot is one
    # A100 80GB or H100 80GB. Expected ~92 h on H100 at ~40% MFU in BF16
    # (~$180 on Vast.ai H100 spot at $2/hr).
    #
    # Above 1.5 B params the ratio of param-to-data gets ugly for our
    # corpus — model capacity exceeds what the data can teach. If you
    # need to go bigger, switch to continued-pretraining on a multi-
    # lingual base model (Qwen-2.5-7B) instead of from-scratch.
    "giant": dict(
        n_layer=32,
        n_head=24,
        n_embd=1536,         # 24 × 64 head_dim
        block_size=2048,
        batch_size=8,
        grad_accum_steps=16, # effective batch = 8*16 = 128
        max_iters=84000,     # ~22 B tokens seen, ~22 tok/param
        learning_rate=2e-4,  # slightly lower than xxlarge for bigger model
        warmup_iters=2000,
        eval_interval=2000,
        save_interval=2000,
        sample_interval=4000,
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
                        choices=["tiny", "small", "medium", "large", "xlarge",
                                 "xxlarge", "xxlarge_4epoch", "giant", "finetune"],
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
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Floor for cosine LR decay (must be <= learning_rate)")
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
