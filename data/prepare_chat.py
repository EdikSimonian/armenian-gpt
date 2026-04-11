"""
Stage 2: Prepare Conversational Data for Fine-tuning

Formats Q&A data with special chat tokens for fine-tuning.

Usage:
    # Use hand-crafted Armenian Q&A (recommended):
    python data/prepare_chat.py --source data/armenian_qa.json

    # Use Alpaca-Armenian from HuggingFace (lower quality):
    python data/prepare_chat.py

After running, you'll have:
    data_chat/train_{char|bpe}.bin      - training data (90%)
    data_chat/val_{char|bpe}.bin        - validation data (10%)
    data_chat/tokenizer_{char|bpe}.json - extended vocabulary with chat tokens

The tokenizer type is auto-detected from data/ unless --tokenizer is passed.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = os.path.join(os.path.dirname(DATA_DIR), "data_chat")

# Special tokens for chat formatting
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

# Words to filter out for age-appropriate content (ages 12-18)
# This is a basic blocklist — extend as needed
BLOCKLIST = [
    # English terms that may appear in translated data
    "kill", "murder", "suicide", "drug", "cocaine", "heroin",
    "sex", "porn", "nude", "violent", "torture", "weapon",
    "bomb", "terrorist", "hack someone", "steal",
]


def is_appropriate(example):
    """Filter out content not suitable for students ages 12-18."""
    text = (example.get("instruction", "") + " " +
            example.get("input", "") + " " +
            example.get("output", "")).lower()
    for word in BLOCKLIST:
        if word in text:
            return False
    # Skip very short or empty responses
    if len(example.get("output", "")) < 10:
        return False
    return True


def format_chat(example):
    """
    Format an Alpaca example as a chat conversation.

    Input:  {instruction: "...", input: "...", output: "..."}
    Output: "<|user|>instruction input<|end|><|assistant|>output<|end|>"
    """
    instruction = example["instruction"].strip()
    inp = example.get("input", "").strip()
    output = example["output"].strip()

    # Combine instruction and input
    if inp:
        user_msg = f"{instruction}\n{inp}"
    else:
        user_msg = instruction

    return f"{USER_TOKEN}{user_msg}{END_TOKEN}{ASSISTANT_TOKEN}{output}{END_TOKEN}"


def load_local_json(path: str) -> list:
    """Load pre-generated Q&A pairs from a local JSON file."""
    print(f"Loading Q&A from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"  Loaded {len(examples):,} examples")
    return examples


def prepare_chat_data(source_path, tokenizer_type):
    """Build data_chat/ bins from a local SFT JSON using Stage 1 tokenizer.

    source_path: path to a {instruction, input?, output} JSON file.
    tokenizer_type: "char" or "bpe" — must match what was used in 3_tokenize.py.
    """
    from tokenizers import (
        bin_paths,
        load_tokenizer as _load_tokenizer,
        tokenizer_path,
    )

    os.makedirs(CHAT_DIR, exist_ok=True)

    all_examples = load_local_json(source_path)

    print("Filtering inappropriate content...")
    filtered = [ex for ex in all_examples if is_appropriate(ex)]
    print(f"  After filtering: {len(filtered):,} examples "
          f"({len(all_examples) - len(filtered)} removed)")

    print("Formatting as chat conversations...")
    formatted = [format_chat(ex) for ex in filtered]
    text = "\n".join(formatted)
    print(f"  Total text: {len(text):,} characters")

    # Load Stage 1 tokenizer
    stage1_tok_path = tokenizer_path(DATA_DIR, tokenizer_type)
    if not os.path.exists(stage1_tok_path):
        print(f"\nError: {stage1_tok_path} not found!")
        print("Run Stage 1 data preparation first:")
        print("  python 1_download.py")
        print("  python 2_prepare.py")
        print(f"  python 3_tokenize.py --tokenizer {tokenizer_type}")
        sys.exit(1)

    tokenizer = _load_tokenizer(DATA_DIR, tokenizer_type)
    old_vocab_size = tokenizer.vocab_size
    print(f"\nStage 1 vocabulary: {old_vocab_size} tokens (type: {tokenizer_type})")

    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN])
    print(f"Extended vocabulary: {tokenizer.vocab_size} tokens "
          f"(+{tokenizer.vocab_size - old_vocab_size} special)")

    if tokenizer_type == "char":
        new_chars = set()
        for ch in text:
            if ch not in tokenizer.stoi and len(ch) == 1:
                new_chars.add(ch)
        if new_chars:
            for ch in sorted(new_chars):
                tokenizer.stoi[ch] = len(tokenizer.itos)
                tokenizer.itos.append(ch)
            print(f"  Added {len(new_chars)} new characters from chat data")
            print(f"  Final vocabulary: {tokenizer.vocab_size} tokens")

    print("Encoding text...")
    token_ids = tokenizer.encode(text)
    token_ids = np.array(token_ids, dtype=np.uint16)
    print(f"  Total tokens: {len(token_ids):,}")

    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_path, val_path = bin_paths(CHAT_DIR, tokenizer_type)
    tok_path = tokenizer_path(CHAT_DIR, tokenizer_type)

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    tokenizer.save(tok_path)

    print(f"\n{'='*50}")
    print(f"  Chat Data Preparation Complete!")
    print(f"{'='*50}")
    print(f"  Examples:     {len(filtered):,}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")
    print(f"  Train tokens: {len(train_ids):,} ({train_path})")
    print(f"  Val tokens:   {len(val_ids):,} ({val_path})")
    print(f"  Train size:   {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val size:     {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")
    print(f"  Tokenizer:    {tok_path}")

    return len(filtered), len(train_ids), len(val_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        help="Path to pre-generated JSON file (e.g. data/qa_merged.json).")
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["char", "bpe"],
                        help="Stage 1 tokenizer type. If omitted, auto-detects from data/.")
    args = parser.parse_args()

    from tokenizers import detect_tokenizer_type

    if args.tokenizer:
        tok_type = args.tokenizer
    else:
        try:
            tok_type = detect_tokenizer_type(DATA_DIR)
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            sys.exit(1)

    source_path = args.source
    if not os.path.isabs(source_path):
        source_path = os.path.join(os.path.dirname(DATA_DIR), source_path)

    prepare_chat_data(source_path, tok_type)


if __name__ == "__main__":
    main()
