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
import multiprocessing as mp
from functools import partial
import numpy as np

# Add project root to path so we can import tokenizers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")


def _clean_chunk(chunk):
    """Worker function for parallel text cleaning."""
    import re
    chunk = unicodedata.normalize("NFC", chunk)
    chunk = re.sub(r"[^\u0530-\u058F\uFB13-\uFB17 \n\t.,;:!?\-()\"'0-9]", "", chunk)
    chunk = re.sub(r"[ \t]+", " ", chunk)
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    return chunk


def clean_text(text):
    """
    Clean and normalize Armenian text in parallel.
    Keeps Armenian letters, common punctuation, digits, and whitespace.
    """
    CHUNK_SIZE = 5_000_000  # 5M chars per chunk
    # Split on newlines to avoid breaking mid-line
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunks.append(text[i:i + CHUNK_SIZE])

    num_workers = mp.cpu_count()
    print(f"  Cleaning {len(chunks)} chunks across {num_workers} CPUs...")
    with mp.Pool(num_workers) as pool:
        cleaned = pool.map(_clean_chunk, chunks)

    text = "".join(cleaned)
    text = text.strip()
    return text


def _encode_chunk(chunk, model_file):
    """Worker function for parallel BPE encoding."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_file)
    return sp.encode(chunk)


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
        # Parallel encode with streaming results to avoid memory explosion
        CHUNK_SIZE = 1_000_000  # characters per chunk
        total_chars = len(text)
        text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, total_chars, CHUNK_SIZE)]
        num_chunks = len(text_chunks)
        num_workers = mp.cpu_count()
        print(f"  Encoding {num_chunks} chunks across {num_workers} CPUs...")

        model_file = os.path.join(DATA_DIR, "bpe_model.model")
        all_ids = []
        with mp.Pool(num_workers) as pool:
            for i, ids in enumerate(pool.imap(partial(_encode_chunk, model_file=model_file), text_chunks)):
                all_ids.append(np.array(ids, dtype=np.uint16))
                if i % 100 == 0:
                    done = min((i + 1) * CHUNK_SIZE, total_chars)
                    print(f"  {done:,}/{total_chars:,} chars ({100*done/total_chars:.0f}%)")

        token_ids = np.concatenate(all_ids)
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
    try:
        print(f"  Original:  {sample[:60]}...")
        print(f"  Decoded:   {decoded[:60]}...")
    except UnicodeEncodeError:
        print(f"  Original:  (contains non-ASCII characters)")
        print(f"  Decoded:   (contains non-ASCII characters)")
    print(f"  Encoded:   {encoded[:20]}...")
    print(f"  Match: {'YES' if sample == decoded else 'NO'}")


if __name__ == "__main__":
    main()
