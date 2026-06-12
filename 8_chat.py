"""
Step 8: Interactive chat with a fine-tuned ArmGPT model.

Talk to the chat model produced by 6_finetune.py in the terminal.
Type a question in Armenian, get a response.

Usage:
    python 8_chat.py
    python 8_chat.py --temperature 0.5
    python 8_chat.py --checkpoint checkpoints_chat/final.pt

Type 'quit' or 'exit' to stop.
"""

import argparse
import os
import sys

# Force UTF-8 stdout/stderr on Windows so Armenian text can be printed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from core.model import GPT
from core import detect_tokenizer_type, load_tokenizer as _load_tokenizer


def load_tokenizer(data_dir, tokenizer_type=None):
    """Load the extended tokenizer with special chat tokens."""
    try:
        if tokenizer_type is None:
            tokenizer_type = detect_tokenizer_type(data_dir)
        return _load_tokenizer(data_dir, tokenizer_type)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Run 'python 3_tokenize.py --qa' then 'python 6_finetune.py' first.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Chat with ArmGPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/chat_final.pt",
        help="Path to Stage 2 (chat) model checkpoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Randomness: 0.3=focused, 0.7=balanced, 1.2=creative",
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Only sample from top k tokens (0=all)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Nucleus sampling cumulative-prob cutoff (0=off, e.g. 0.9)",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Min-p sampling floor as a fraction of the top token's prob "
        "(0=off, e.g. 0.05). Robust for small models; pairs well with top_k off.",
    )
    parser.add_argument(
        "--max_length", type=int, default=300, help="Maximum response length in tokens"
    )
    parser.add_argument(
        "--single_turn",
        action="store_true",
        help="Disable conversation memory (each message answered in isolation)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Penalize already-used tokens (>1 suppresses loops; 1.0=off). "
        "Default 1.15 — the SFT set is small so the model is prone to repeat "
        "loops on out-of-distribution prompts (e.g. how-to questions).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_chat",
        help="Directory containing the chat tokenizer file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        choices=["char", "bpe"],
        help="Tokenizer type. If omitted, auto-detects from data_dir.",
    )
    args = parser.parse_args()

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        print("Fine-tune a model first:")
        print("  1. python 1_download.py --qa")
        print("  2. python 2_prepare.py --qa")
        print("  3. python 3_tokenize.py --qa --tokenizer bpe")
        print("  4. python 6_finetune.py")
        sys.exit(1)

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Infer model architecture from saved weights (config may be stale)
    state = checkpoint["model"]
    # Strip torch.compile() prefix if present.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        checkpoint["model"] = state
    cfg["n_embd"] = state["transformer.wte.weight"].shape[1]
    cfg["n_layer"] = (
        max(int(k.split(".")[2]) for k in state if k.startswith("transformer.blocks."))
        + 1
    )
    # block_size: this model uses RoPE, so the rope_cos buffer's first dim IS the
    # model's block_size. Do NOT trust cfg["block_size"] — for chat checkpoints it
    # carries the *training-window* size (e.g. 1024), not the model's RoPE buffer
    # size (e.g. 2048), which would size-mismatch on load. n_head likewise follows
    # from the RoPE head_dim. Fall back to attn.bias for older non-RoPE ckpts.
    rope_key = "transformer.blocks.0.attn.rope_cos"
    if rope_key in state:
        cfg["block_size"] = int(state[rope_key].shape[0])
        head_dim = int(state[rope_key].shape[1]) * 2  # RoPE stores half the head_dim
        if head_dim > 0 and cfg["n_embd"] % head_dim == 0:
            cfg["n_head"] = cfg["n_embd"] // head_dim
    elif "transformer.blocks.0.attn.bias" in state:
        cfg["block_size"] = state["transformer.blocks.0.attn.bias"].shape[-1]
    # Detect QK-norm from the weights (present only if trained with it).
    cfg["qk_norm"] = any("attn.q_norm.weight" in k for k in state)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load tokenizer
    tokenizer = load_tokenizer(args.data_dir, args.tokenizer)

    # Get special token IDs
    end_ids = tokenizer.encode("<|end|>")
    user_ids = tokenizer.encode("<|user|>")
    stop_tokens = set()
    if end_ids:
        stop_tokens.add(end_ids[0])
    if user_ids:
        stop_tokens.add(user_ids[0])

    # Create model and load weights
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=0.0,
        qk_norm=cfg.get("qk_norm", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None
    min_p = args.min_p if args.min_p > 0 else None

    def build_prompt_ids(history):
        """Render the conversation into prompt ids, ending with an assistant cue,
        then drop the OLDEST turns until it fits the model's context with room
        for a full response. Always keeps the latest user turn."""
        budget = cfg["block_size"] - args.max_length - 8  # 8 = special-token slack
        while True:
            parts = []
            for role, text in history:
                tag = "<|user|>" if role == "user" else "<|assistant|>"
                parts.append(f"{tag}{text}<|end|>")
            parts.append("<|assistant|>")
            ids = tokenizer.encode("".join(parts))
            if len(ids) <= budget or len(history) <= 1:
                return ids
            history.pop(0)  # evict the oldest turn and re-render

    # Chat loop
    mode = "single-turn" if args.single_turn else "multi-turn"
    print(f"\n{'=' * 50}")
    print("  ArmGPT Chat")
    print(f"  Device: {device} | Temp: {args.temperature} | {mode}")
    print("  Type 'quit' to exit, 'reset' to clear the conversation")
    print(f"{'=' * 50}\n")

    history = []  # list of (role, text)
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() in ("reset", "clear"):
            history = []
            print("(conversation cleared)\n")
            continue

        if args.single_turn:
            history = []
        history.append(("user", user_input))

        prompt_ids = build_prompt_ids(history)
        if not prompt_ids:
            print("ArmGPT: (could not encode your message)\n")
            history.pop()
            continue

        # Generate response
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            context,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            stop_tokens=stop_tokens if stop_tokens else None,
            repetition_penalty=args.repetition_penalty,
        )

        # The response is exactly the newly generated tokens after the prompt.
        new_ids = output[0].tolist()[len(prompt_ids) :]
        response = tokenizer.decode(new_ids)
        response = response.replace("<|end|>", "").replace("<|user|>", "")
        response = response.replace("<|assistant|>", "").strip()

        if response:
            print(f"ArmGPT: {response}\n")
            history.append(("assistant", response))
        else:
            print("ArmGPT: ...\n")
            history.pop()  # drop the user turn that produced nothing


if __name__ == "__main__":
    main()
