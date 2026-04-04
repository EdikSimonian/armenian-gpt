"""
Stage 2: Prepare Conversational Data for Fine-tuning

Downloads the Alpaca-Armenian dataset (52K instruction/response pairs in Armenian)
and formats it with special chat tokens for fine-tuning.

Usage:
    python data/prepare_chat.py

After running, you'll have:
    data_chat/train.bin      - training data (90%)
    data_chat/val.bin        - validation data (10%)
    data_chat/tokenizer.json - extended vocabulary with chat tokens
"""

import os
import sys
import json
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


def download_alpaca_armenian():
    """Download the Alpaca-Armenian dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print("Downloading Alpaca-Armenian dataset from HuggingFace...")
    print("  Dataset: saillab/alpaca-armenian-cleaned")
    ds = load_dataset("saillab/alpaca-armenian-cleaned")
    print(f"  Train examples: {len(ds['train']):,}")
    print(f"  Test examples:  {len(ds['test']):,}")
    return ds


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


def main():
    # Step 1: Create output directory
    os.makedirs(CHAT_DIR, exist_ok=True)

    # Step 2: Download dataset
    ds = download_alpaca_armenian()

    # Step 3: Combine train + test, then filter
    all_examples = list(ds["train"]) + list(ds["test"])
    print(f"\nTotal examples: {len(all_examples):,}")

    print("Filtering inappropriate content...")
    filtered = [ex for ex in all_examples if is_appropriate(ex)]
    print(f"  After filtering: {len(filtered):,} examples "
          f"({len(all_examples) - len(filtered)} removed)")

    # Step 4: Format as chat conversations
    print("Formatting as chat conversations...")
    formatted = [format_chat(ex) for ex in filtered]

    # Join all conversations into one long text
    text = "\n".join(formatted)
    print(f"  Total text: {len(text):,} characters")

    # Step 5: Load Stage 1 tokenizer and extend with special tokens
    stage1_tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    if not os.path.exists(stage1_tok_path):
        print(f"\nError: {stage1_tok_path} not found!")
        print("Run Stage 1 data preparation first:")
        print("  python data/download.py")
        print("  python data/prepare.py")
        sys.exit(1)

    from tokenizers.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer.load(stage1_tok_path)
    old_vocab_size = tokenizer.vocab_size
    print(f"\nStage 1 vocabulary: {old_vocab_size} tokens")

    # Add special chat tokens
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN])
    print(f"Extended vocabulary: {tokenizer.vocab_size} tokens (+{tokenizer.vocab_size - old_vocab_size} special)")

    # Also add any new characters from the Alpaca data that weren't in Wikipedia
    new_chars = set()
    for ch in text:
        if ch not in tokenizer.stoi and len(ch) == 1:
            new_chars.add(ch)
    if new_chars:
        for ch in sorted(new_chars):
            tokenizer.stoi[ch] = len(tokenizer.itos)
            tokenizer.itos.append(ch)
        print(f"  Added {len(new_chars)} new characters from Alpaca data")
        print(f"  Final vocabulary: {tokenizer.vocab_size} tokens")

    # Step 6: Encode the text
    print("Encoding text...")
    token_ids = tokenizer.encode(text)
    token_ids = np.array(token_ids, dtype=np.uint16)
    print(f"  Total tokens: {len(token_ids):,}")

    # Step 7: Split into train and validation (90/10)
    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # Step 8: Save everything
    train_path = os.path.join(CHAT_DIR, "train.bin")
    val_path = os.path.join(CHAT_DIR, "val.bin")
    tok_path = os.path.join(CHAT_DIR, "tokenizer.json")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    tokenizer.save(tok_path)

    # Print summary
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

    # Show special token IDs
    print(f"\nSpecial tokens:")
    print(f"  {USER_TOKEN:15s} -> {tokenizer.stoi[USER_TOKEN]}")
    print(f"  {ASSISTANT_TOKEN:15s} -> {tokenizer.stoi[ASSISTANT_TOKEN]}")
    print(f"  {END_TOKEN:15s} -> {tokenizer.stoi[END_TOKEN]}")

    # Show a sample
    sample_text = formatted[0][:200]
    sample_ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(sample_ids)
    print(f"\nSample conversation:")
    print(f"  {sample_text[:150]}...")
    print(f"\nRound-trip test: {'PASS' if sample_text == decoded else 'FAIL'}")

    print(f"\nNext step: python finetune.py")


if __name__ == "__main__":
    main()
