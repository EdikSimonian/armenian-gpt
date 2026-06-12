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
        max_iters=168000,  # 4 × 42000
        learning_rate=2.5e-4,
        warmup_iters=2000,  # 1.2% of schedule — plenty for this scale
        eval_interval=4000,
        save_interval=2000,  # 84 HF uploads across the run vs 168 at 1000
        sample_interval=4000,
    ),
    # ~1.0 B params. Bigger than xxlarge — approaches the data ceiling
    # for the ~16 B-token Armenian corpus. Chinchilla-optimal for 1 B is
    # 20 B tokens, so with max_iters=122000 × eff_batch 144 × ctx 2048 =
    # ~36 B tokens seen (~36 tokens/param), ~75% over Chinchilla-optimal
    # — fine, the second epoch buys ~10% loss improvement.
    #
    # Tuned for RTX PRO 6000 Blackwell (96 GB, ~1.6 TB/s). batch=8 / accum=16
    # is the safe config: torch.compile materializes extra fp32 activation
    # buffers in the compiled graph, so batch=24 OOMs at ~94 GB even though
    # the bf16 forward fits in ~35 GB. Eff batch 128 matches the H100 80GB
    # reference config. Expected ~90–110 h on RTX PRO 6000.
    #
    # Above 1.5 B params the ratio of param-to-data gets ugly for our
    # corpus — model capacity exceeds what the data can teach. If you
    # need to go bigger, switch to continued-pretraining on a multi-
    # lingual base model (Qwen-2.5-7B) instead of from-scratch.
    "giant": dict(
        n_layer=32,
        n_head=24,
        n_embd=1536,  # 24 × 64 head_dim
        block_size=2048,
        batch_size=8,
        grad_accum_steps=16,  # effective batch = 8*16 = 128
        max_iters=125000,  # extended from 122000; ~4 epochs over 8.2 B unique tokens (~32.8 B seen)
        learning_rate=2e-4,
        min_lr=2e-5,  # 10% of peak — modern recipe. The base default 1e-4 (50%
        # floor) prevents the model from entering a true settling phase in the
        # late cosine tail; LLaMA/Mistral/etc. decay to ~10%. Codex flagged this
        # mid-run at step 16k as the single biggest hyperparam smell.
        warmup_iters=2000,
        eval_interval=4000,
        eval_iters=50,  # 50 iters × 2 splits is enough signal; 200 wastes time
        save_interval=1000,  # dense snapshots throughout: gpu-pc has recurring
        # power crashes under load (PSU/connector — hardware), so checkpoint every
        # 1k to bound lost work to ~1k steps (~2 h) per crash instead of 4k (~9 h).
        # ~11 GB each; NAS keeps the full history, KEEP_LOCAL=5 bounds local disk.
        # (late_start=0 makes the 1k cadence uniform; was 4k pre-40k, 1k after.)
        save_interval_late=1000,
        save_interval_late_start=0,
        sample_interval=4000,
        dropout=0.0,  # 1 B params on 36 B tokens — no need for dropout
    ),
    # v2 of the giant preset — same 930M architecture, modern recipe knobs.
    # Pairs with the v2 data pipeline (32k vocab, document EOS, fixed cleaning).
    # Differences from "giant":
    #   - lr_schedule "wsd": warmup-stable-decay. The base run's horizon changed
    #     twice (122k->125k, then stopped at 120k); WSD makes the model usable
    #     throughout the stable phase and annealed only at the end, so stop/
    #     extend/continue decisions are free. (cosine bakes the endpoint in.)
    #   - qk_norm available via --qk_norm (kept OFF here so a default run doesn't
    #     depend on every checkpoint-loader detecting it; flip on if you raise LR
    #     or depth and see late-run spikes).
    #   - weight-decay grouping + AdamW betas (0.9, 0.95) + scaled residual init
    #     come from core/training.py and core/model.py automatically.
    # NOTE: with the 32k vocab the corpus tokenizes to ~15-20% FEWER tokens, so
    # recompute max_iters for the target epoch count from the printed token
    # total after 3_tokenize (eff batch = 8*16*2048 = 262144 tokens/step).
    "giant_v2": dict(
        n_layer=32,
        n_head=24,
        n_embd=1536,
        block_size=2048,
        batch_size=8,
        grad_accum_steps=16,  # effective batch = 8*16 = 128
        max_iters=125000,  # recompute from the v2 token total (see NOTE)
        learning_rate=2e-4,
        min_lr=2e-5,  # 10% of peak
        lr_schedule="wsd",
        decay_frac=0.2,  # last 20% of steps anneal peak->min_lr
        warmup_iters=2000,
        qk_norm=False,
        eval_interval=4000,
        eval_iters=50,
        save_interval=1000,
        save_interval_late=1000,
        save_interval_late_start=0,
        sample_interval=4000,
        dropout=0.0,
    ),
    # Stage 2: fine-tuning on conversational data (small base model)
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
    # Stage 2 SFT for the ~930 M "giant" base model.
    #
    # The architecture (n_layer/n_head/n_embd, RoPE buffer size) is INFERRED
    # from the base checkpoint at load time in 6_finetune.py — the arch fields
    # below are documentation only (they match the giant preset) and are
    # overridden. Only the *training* hyperparameters here actually take effect.
    #
    # Key differences from the small "finetune" preset, all driven by scale +
    # what the base-model samples showed (fluent but drifts off-prompt):
    #   - learning_rate 2e-5: SFT for a ~1 B model. The 3e-4 above is the 10 M
    #     preset value and is ~15× too hot here — it would wash out the
    #     pretrained Armenian and cause catastrophic forgetting.
    #   - block_size 1024: long QA answers (armbench averages ~500 tokens)
    #     no longer get truncated to a 256 window. Base supports 2048.
    #   - batch_size 8 × grad_accum 4 = effective batch 32: batch 8 matches the
    #     known-good pretraining microbatch (8×2048 fit in ~55 GB; 8×1024 is
    #     half that), and accumulation gives a less noisy SFT gradient than a
    #     bare batch of 8. 6_finetune.py honors grad_accum_steps + CUDA AMP.
    #   - dropout 0.05: light regularization on the small (~6 M-token) SFT set.
    # NOTE: max_iters is data-dependent — recompute for ~3 epochs after the
    # data mix is rebalanced (#3). At batch 8 × accum 4 × block 1024 = 32768
    # tok/step, one epoch over ~6 M tokens ≈ 183 steps, so ~550 steps ≈ 3 epochs
    # on the CURRENT data. 600 is a placeholder; raise it as the corpus grows.
    "finetune_giant": dict(
        n_layer=32,
        n_head=24,
        n_embd=1536,
        block_size=1024,
        batch_size=8,
        grad_accum_steps=4,  # effective batch = 8 * 4 = 32
        # The rebalanced SFT corpus is small (~1-1.4 M tokens), so a step is
        # ~37-43 tokens-worth of epochs. max_iters is a CEILING — chat_best.pt
        # tracks val loss and captures the real stopping point, so a slightly
        # high ceiling just wastes a little compute, it doesn't overfit the
        # saved model. warmup must stay well under max_iters at this scale.
        max_iters=200,
        learning_rate=2e-5,
        min_lr=2e-6,  # 10% floor
        warmup_iters=20,
        weight_decay=0.1,
        grad_clip=1.0,
        dropout=0.05,
        tokenizer="bpe",  # giant is a BPE model; pin so the preset can't fall
        # back to the module default "char" and look for char chat bins.
        # Cadences sized for the short (~200-step) run so val eval fires often
        # enough for chat_best.pt to actually capture the val-loss minimum.
        # (Sample text is generated inside the eval block, i.e. every
        # eval_interval steps — there is no separate sample_interval knob here.)
        eval_interval=25,
        eval_iters=40,
        save_interval=50,
        checkpoint_dir="checkpoints_chat",
    ),
}

# --- Default Config ---
# These are the "small" preset values. Override with --preset or CLI flags.

# model
n_layer = 6  # number of transformer blocks
n_head = 6  # number of attention heads
n_embd = 384  # embedding dimension (must be divisible by n_head)
block_size = 256  # context window length (in tokens/characters)
dropout = 0.2  # dropout rate for regularization

# training
batch_size = 64  # how many sequences to process at once
max_iters = 5000  # total training steps
learning_rate = 1e-3  # peak learning rate
warmup_iters = 100  # linear warmup steps
min_lr = 1e-4  # minimum learning rate after decay
lr_schedule = "cosine"  # "cosine" or "wsd" (warmup-stable-decay)
decay_frac = 0.2  # WSD only: fraction of steps spent annealing peak->min_lr
weight_decay = 0.1  # AdamW weight decay (applied to dim>=2 params only)
grad_clip = 1.0  # gradient clipping (0 = no clipping)
qk_norm = False  # RMSNorm on per-head q/k (stabilizer; off by default)
compile_mode = "default"  # torch.compile mode: default | max-autotune | reduce-overhead
grad_accum_steps = (
    1  # gradient accumulation steps (effective batch = batch_size * this)
)

# evaluation and logging
eval_interval = 500  # evaluate every N steps
eval_iters = 200  # number of batches to average for eval loss
log_interval = 10  # print training loss every N steps
sample_interval = 500  # generate sample text every N steps
sample_length = 200  # how many characters/tokens to generate in samples

# checkpointing
checkpoint_dir = "checkpoints"
save_interval = 1000  # save checkpoint every N steps
resume_from = ""  # path to checkpoint to resume from

# data
data_dir = "data"
tokenizer = "char"  # "char" for Level 1, "bpe" for Level 2

# device (auto-detect)
device = "auto"  # "auto", "cpu", "cuda", or "mps"


def get_config():
    """Parse command-line arguments and return the final config as a dict."""
    parser = argparse.ArgumentParser(description="ArmGPT Training Config")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[
            "tiny",
            "small",
            "medium",
            "large",
            "xlarge",
            "xxlarge",
            "xxlarge_4epoch",
            "giant",
            "giant_v2",
            "finetune",
            "finetune_giant",
        ],
        help="Use a preset configuration",
    )
    # Allow overriding any config value from the command line
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=None,
        help="Floor for cosine LR decay (must be <= learning_rate)",
    )
    parser.add_argument("--warmup_iters", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument(
        "--lr_schedule", type=str, default=None, choices=["cosine", "wsd"]
    )
    parser.add_argument(
        "--decay_frac",
        type=float,
        default=None,
        help="WSD only: fraction of steps spent annealing (e.g. 0.2)",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
        default=None,
        help="Enable per-head QK-normalization (attention-logit stabilizer)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=["default", "max-autotune", "reduce-overhead"],
        help="torch.compile mode",
    )
    parser.add_argument("--tokenizer", type=str, default=None, choices=["char", "bpe"])
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--sample_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload checkpoints (e.g. edisimon/armgpt)",
    )

    args = parser.parse_args()

    # Start with module-level defaults
    cfg = {
        k: v
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, str))
    }

    # Apply preset if specified
    if args.preset:
        cfg.update(PRESETS[args.preset])

    # Apply any explicit CLI overrides
    for key, val in vars(args).items():
        if val is not None and key != "preset":
            cfg[key] = val

    # Guard: the cosine/WSD tail decays TOWARD min_lr, so min_lr must not exceed
    # the peak — otherwise lowering --learning_rate below the default min_lr
    # would make the schedule RAISE the LR in its tail. Clamp with a warning.
    if cfg.get("min_lr") is not None and cfg["min_lr"] > cfg["learning_rate"]:
        print(
            f"  WARNING: min_lr ({cfg['min_lr']}) > learning_rate "
            f"({cfg['learning_rate']}); clamping min_lr to the peak LR."
        )
        cfg["min_lr"] = cfg["learning_rate"]

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
