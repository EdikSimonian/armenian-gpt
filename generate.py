"""
ArmGPT Text Generation

Generate Armenian text using a trained model.

Usage:
    python generate.py
    python generate.py --prompt "Հայաստանի"
    python generate.py --temperature 0.5 --length 500
    python generate.py --checkpoint checkpoints/step_5000.pt
"""

import argparse
import os
import torch

from model import GPT
from tokenizers import detect_tokenizer_type, load_tokenizer as _load_tokenizer


def load_tokenizer(data_dir, tokenizer_type=None):
    """Load the tokenizer used during training.

    If tokenizer_type is None, auto-detects from data_dir.
    """
    if tokenizer_type is None:
        tokenizer_type = detect_tokenizer_type(data_dir)
    return _load_tokenizer(data_dir, tokenizer_type)


def main():
    parser = argparse.ArgumentParser(description="Generate Armenian text with ArmGPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Հայաստան",
                        help="Starting text (Armenian)")
    parser.add_argument("--length", type=int, default=200,
                        help="Number of tokens/characters to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Randomness: 0.1=safe, 0.8=balanced, 1.5=creative")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Only sample from top k tokens (0=all)")
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                        help="Penalty for repeating tokens already in context. "
                             "1.0=off, 1.1-1.3 typical, helps escape repetition loops")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="How many samples to generate")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing the tokenizer file")
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["char", "bpe"],
                        help="Tokenizer type. If omitted, auto-detects from data_dir.")
    args = parser.parse_args()

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        print("Train a model first with: python train.py")
        return

    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load tokenizer
    tokenizer = load_tokenizer(args.data_dir, args.tokenizer)

    # Create model and load weights
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=0.0,  # no dropout during generation
    ).to(device)
    state_dict = checkpoint["model"]
    # Strip torch.compile() prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Encode the prompt
    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) == 0:
        print("Warning: prompt produced no tokens. Using default seed.")
        prompt_ids = [0]

    print(f"\nDevice: {device}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Generating {args.length} tokens...\n")

    # Generate
    top_k = args.top_k if args.top_k > 0 else None
    for i in range(args.num_samples):
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(context, max_new_tokens=args.length,
                                temperature=args.temperature, top_k=top_k,
                                repetition_penalty=args.repetition_penalty)
        text = tokenizer.decode(output[0].tolist())

        if args.num_samples > 1:
            print(f"--- Sample {i+1} ---")
        print(text)
        if args.num_samples > 1:
            print()


if __name__ == "__main__":
    main()
