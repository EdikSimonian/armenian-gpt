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
    Uses regex for speed — handles 900M+ characters in under a minute.
    """
    import re

    # Normalize Unicode (combine accents, standardize characters)
    text = unicodedata.normalize("NFC", text)

    # Remove everything that is NOT Armenian, punctuation, digits, or whitespace
    # Armenian Unicode block: U+0530-U+058F, ligatures: U+FB13-U+FB17
    print("  Filtering characters...")
    text = re.sub(r"[^\u0530-\u058F\uFB13-\uFB17 \n\t.,;:!?\-()\"'0-9]", "", text)

    # Collapse multiple spaces/tabs into one space
    print("  Collapsing whitespace...")
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
    if args.tokenizer == "char":
        # Fast path: use numpy for bulk encoding instead of Python loop
        # Build a lookup table: Unicode codepoint -> token ID
        max_cp = max(ord(ch) for ch in tokenizer.stoi) + 1
        lookup = np.full(max_cp, -1, dtype=np.int32)
        for ch, idx in tokenizer.stoi.items():
            lookup[ord(ch)] = idx
        # Convert text to array of codepoints, then map to token IDs
        codepoints = np.array([ord(ch) for ch in text], dtype=np.int32)
        token_ids = lookup[codepoints]
        # Remove characters not in vocab (mapped to -1)
        token_ids = token_ids[token_ids >= 0].astype(np.uint16)
    else:
        token_ids = np.array(tokenizer.encode(text), dtype=np.uint16)
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
