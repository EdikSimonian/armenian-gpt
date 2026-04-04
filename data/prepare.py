"""
Prepare Armenian text data for training.

This script reads the raw text, builds a character vocabulary,
encodes everything as integers, and saves train/val binary files.

Usage:
    python data/prepare.py
    python data/prepare.py --tokenizer bpe   # for Level 2

After running, you'll have:
    data/train.bin  - training data (90%)
    data/val.bin    - validation data (10%)
    data/vocab.json - character-to-integer mapping
"""

import os
import sys
import json
import unicodedata
import numpy as np

# Add project root to path so we can import tokenizers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")


def clean_text(text):
    """
    Clean and normalize Armenian text.
    Keeps Armenian letters, common punctuation, digits, and whitespace.
    """
    # Normalize Unicode (combine accents, standardize characters)
    text = unicodedata.normalize("NFC", text)

    # Keep only allowed characters
    allowed = set()
    cleaned = []
    for ch in text:
        # Armenian Unicode block: U+0530 to U+058F
        # Armenian ligatures: U+FB13 to U+FB17
        if "\u0530" <= ch <= "\u058F" or "\uFB13" <= ch <= "\uFB17":
            cleaned.append(ch)
            allowed.add(ch)
        elif ch in " \n\t":
            cleaned.append(ch)
        elif ch in ".,;:!?-()\"'0123456789":
            cleaned.append(ch)
            allowed.add(ch)
        # Skip everything else (Latin, Cyrillic, emojis, etc.)

    text = "".join(cleaned)

    # Collapse multiple spaces/newlines
    import re
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="char",
                        choices=["char", "bpe"],
                        help="Tokenizer type: 'char' (Level 1) or 'bpe' (Level 2)")
    args = parser.parse_args()

    # Step 1: Read raw text
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found!")
        print("Run 'python data/download.py' first to download the data.")
        sys.exit(1)

    print(f"Reading {RAW_FILE}...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"  Raw text: {len(raw_text):,} characters")

    # Step 2: Clean the text
    print("Cleaning text...")
    text = clean_text(raw_text)
    print(f"  Cleaned text: {len(text):,} characters")

    # Step 3: Initialize tokenizer
    if args.tokenizer == "char":
        from tokenizers.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(text)
    else:
        from tokenizers.bpe_tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        # Save text temporarily for SentencePiece training
        tmp_file = os.path.join(DATA_DIR, "clean_text.txt")
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(text)
        tokenizer.train(tmp_file, model_prefix=os.path.join(DATA_DIR, "bpe_model"))

    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Step 4: Encode the text
    print("Encoding text...")
    token_ids = tokenizer.encode(text)
    token_ids = np.array(token_ids, dtype=np.uint16)
    print(f"  Total tokens: {len(token_ids):,}")

    # Step 5: Split into train and validation (90/10)
    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # Step 6: Save binary files
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    # Step 7: Save tokenizer info
    tokenizer.save(os.path.join(DATA_DIR, "tokenizer.json"))

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Data Preparation Complete!")
    print(f"{'='*50}")
    print(f"  Tokenizer:    {args.tokenizer}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")
    print(f"  Train tokens: {len(train_ids):,} ({train_path})")
    print(f"  Val tokens:   {len(val_ids):,} ({val_path})")
    print(f"  Train size:   {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val size:     {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")
    print()

    # Show a sample of the vocabulary
    print("Sample vocabulary (first 20 tokens):")
    if args.tokenizer == "char":
        for i, ch in enumerate(tokenizer.itos[:20]):
            display = repr(ch) if ch in (" ", "\n", "\t") else ch
            print(f"  {i:3d} -> {display}")
    print()

    # Show a sample encode/decode round-trip
    sample = text[:100]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print("Sample round-trip test:")
    print(f"  Original:  {sample[:60]}...")
    print(f"  Encoded:   {encoded[:20]}...")
    print(f"  Decoded:   {decoded[:60]}...")
    print(f"  Match: {'YES' if sample == decoded else 'NO'}")


if __name__ == "__main__":
    main()
