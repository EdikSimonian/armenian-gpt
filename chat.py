"""
ArmGPT Chat - Interactive Armenian Chatbot

Talk to your fine-tuned ArmGPT model in the terminal.
Type a question in Armenian, get a response. Like a mini ChatGPT!

Usage:
    python chat.py
    python chat.py --temperature 0.5
    python chat.py --checkpoint checkpoints/chat_final.pt

Type 'quit' or 'exit' to stop.
"""

import argparse
import json
import os
import sys
import torch

from model import GPT


def load_tokenizer(data_dir):
    """Load the extended tokenizer with special chat tokens."""
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"Error: {tok_path} not found!")
        print("Run 'python data/prepare_chat.py' first.")
        sys.exit(1)

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    if tok_data.get("type") == "bpe":
        from tokenizers.bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(tok_path)
    else:
        from tokenizers.char_tokenizer import CharTokenizer
        return CharTokenizer.load(tok_path)


def main():
    parser = argparse.ArgumentParser(description="Chat with ArmGPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/chat_final.pt",
                        help="Path to Stage 2 (chat) model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Randomness: 0.3=focused, 0.7=balanced, 1.2=creative")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Only sample from top k tokens (0=all)")
    parser.add_argument("--max_length", type=int, default=300,
                        help="Maximum response length in tokens")
    parser.add_argument("--data_dir", type=str, default="data_chat",
                        help="Directory with tokenizer.json")
    args = parser.parse_args()

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        print("Fine-tune a model first:")
        print("  1. python data/prepare_chat.py")
        print("  2. python finetune.py")
        sys.exit(1)

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Infer model architecture from saved weights (config may be stale)
    state = checkpoint["model"]
    cfg["n_embd"] = state["transformer.wte.weight"].shape[1]
    cfg["n_layer"] = max(int(k.split(".")[2]) for k in state if k.startswith("transformer.blocks.")) + 1
    cfg["block_size"] = state["transformer.blocks.0.attn.bias"].shape[-1]

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load tokenizer
    tokenizer = load_tokenizer(args.data_dir)

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
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    top_k = args.top_k if args.top_k > 0 else None

    # Chat loop
    print(f"\n{'='*50}")
    print(f"  ArmGPT Chat")
    print(f"  Device: {device} | Temp: {args.temperature}")
    print(f"  Type 'quit' to exit")
    print(f"{'='*50}\n")

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

        # Format as chat prompt
        prompt = f"<|user|>{user_input}<|end|><|assistant|>"
        prompt_ids = tokenizer.encode(prompt)

        if not prompt_ids:
            print("ArmGPT: (could not encode your message)\n")
            continue

        # Generate response
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            context,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_k=top_k,
            stop_tokens=stop_tokens if stop_tokens else None,
        )

        # Decode and clean up the response
        full_text = tokenizer.decode(output[0].tolist())

        # Extract just the assistant's response
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1]
        else:
            response = full_text[len(tokenizer.decode(prompt_ids)):]

        # Remove any trailing special tokens
        response = response.replace("<|end|>", "").replace("<|user|>", "").strip()

        if response:
            print(f"ArmGPT: {response}\n")
        else:
            print("ArmGPT: ...\n")


if __name__ == "__main__":
    main()
