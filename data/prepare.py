"""
Prepare Armenian text data for training.

This script reads the raw text, builds a character vocabulary,
encodes everything as integers, and saves train/val binary files.

Memory-safe: processes text in chunks to handle 20+ GB files
without loading everything into RAM.

Usage:
    python data/prepare.py
    python data/prepare.py --tokenizer bpe   # for Level 2

After running, you'll have:
    data/train.bin  - training data (90%)
    data/val.bin    - validation data (10%)
    data/tokenizer.json - tokenizer metadata
"""

import os
import sys
import re
import json
import unicodedata
import numpy as np

# Add project root to path so we can import tokenizers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")

# Regex patterns compiled once
_RE_NON_ARMENIAN = re.compile(r"[^\u0530-\u058F\uFB13-\uFB17 \n\t.,;:!?\-()\"'0-9]")
_RE_SPACES = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")


def clean_chunk(chunk):
    """Clean a chunk of text. Keeps Armenian letters, punctuation, digits, whitespace."""
    chunk = unicodedata.normalize("NFC", chunk)
    chunk = _RE_NON_ARMENIAN.sub("", chunk)
    chunk = _RE_SPACES.sub(" ", chunk)
    chunk = _RE_NEWLINES.sub("\n\n", chunk)
    return chunk


def get_file_size(path):
    """Get file size in bytes."""
    return os.path.getsize(path) if os.path.exists(path) else 0


def _find_segment_boundaries(path, num_segments):
    """Find byte offsets that split a file into segments at newline boundaries."""
    file_size = os.path.getsize(path)
    boundaries = [0]
    with open(path, "rb") as f:
        for i in range(1, num_segments):
            target = (file_size * i) // num_segments
            f.seek(target)
            # Read ahead to find next newline
            chunk = f.read(8192)
            nl = chunk.find(b"\n")
            if nl != -1:
                boundaries.append(target + nl + 1)
            else:
                boundaries.append(target)
    boundaries.append(file_size)
    return boundaries


def _clean_segment(args):
    """Clean one segment of the file in small chunks (for multiprocessing)."""
    input_path, start_byte, end_byte, segment_id = args
    chunk_size = 50_000_000  # 50 MB at a time
    out_path = os.path.join(DATA_DIR, f"clean_seg_{segment_id}.txt")
    out_chars = 0
    with open(input_path, "rb") as fin, open(out_path, "w", encoding="utf-8") as fout:
        fin.seek(start_byte)
        while fin.tell() < end_byte:
            remaining = end_byte - fin.tell()
            read_size = min(chunk_size, remaining)
            raw_bytes = fin.read(read_size)
            if not raw_bytes:
                break
            # If not at end of segment, extend to next newline to avoid mid-line split
            if fin.tell() < end_byte:
                extra = fin.read(8192)
                if extra:
                    nl = extra.find(b"\n")
                    if nl != -1:
                        raw_bytes += extra[:nl + 1]
                        fin.seek(-(len(extra) - nl - 1), 1)
                    else:
                        raw_bytes += extra
            text = raw_bytes.decode("utf-8", errors="ignore")
            cleaned = clean_chunk(text)
            fout.write(cleaned)
            out_chars += len(cleaned)
    seg_mb = (end_byte - start_byte) / (1024 * 1024)
    print(f"  Segment {segment_id}: {seg_mb:.0f} MB -> {out_chars / 1_000_000:.0f}M clean chars")
    return out_path, out_chars


def clean_file_chunked(input_path, output_path, chunk_bytes=50_000_000, parallel=True):
    """
    Clean the raw text file, writing cleaned output to a new file.
    If parallel=True, uses multiprocessing to clean segments concurrently.
    Each worker processes its segment in 50 MB chunks to limit RAM usage.
    """
    from multiprocessing import Pool, cpu_count
    total_bytes = get_file_size(input_path)
    total_mb = total_bytes / (1024 * 1024)

    if parallel:
        num_workers = min(cpu_count(), 16)
        print(f"  Cleaning {total_mb:.0f} MB in parallel with {num_workers} workers...")
        boundaries = _find_segment_boundaries(input_path, num_workers)
        args = [
            (input_path, boundaries[i], boundaries[i + 1], i)
            for i in range(len(boundaries) - 1)
        ]
        with Pool(num_workers) as pool:
            results = pool.map(_clean_segment, args)

        # Concatenate segment outputs in order
        out_chars = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for seg_path, seg_chars in results:
                out_chars += seg_chars
                with open(seg_path, "r", encoding="utf-8") as fin:
                    while True:
                        chunk = fin.read(100_000_000)
                        if not chunk:
                            break
                        fout.write(chunk)
                os.remove(seg_path)

        print(f"  Cleaned: {out_chars:,} characters")
        return out_chars

    # Fallback: sequential mode
    processed = 0
    out_chars = 0

    print(f"  Cleaning {total_mb:.0f} MB in ~{chunk_bytes // 1_000_000} MB chunks...")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        buffer = ""
        while True:
            raw = fin.read(chunk_bytes)
            if not raw:
                # Process remaining buffer
                if buffer:
                    cleaned = clean_chunk(buffer)
                    fout.write(cleaned)
                    out_chars += len(cleaned)
                break

            buffer += raw
            processed += len(raw.encode("utf-8"))

            # Find last newline to avoid splitting mid-line
            last_nl = buffer.rfind("\n")
            if last_nl == -1:
                if len(buffer) > 3 * chunk_bytes:
                    # Hard cap: force flush to prevent OOM on pathological input
                    last_nl = len(buffer) - 1
                else:
                    continue

            # Clean everything up to the last newline
            to_clean = buffer[:last_nl + 1]
            buffer = buffer[last_nl + 1:]

            cleaned = clean_chunk(to_clean)
            fout.write(cleaned)
            out_chars += len(cleaned)

            pct = min(100, processed * 100 / total_bytes) if total_bytes > 0 else 0
            print(f"  {pct:.0f}% ({processed / (1024**2):.0f}/{total_mb:.0f} MB) — "
                  f"{out_chars / 1_000_000:.0f}M clean chars")

    print(f"  Cleaned: {out_chars:,} characters")
    return out_chars


def build_char_vocab(clean_path, chunk_bytes=50_000_000):
    """Scan the clean file to build character vocabulary without loading it all."""
    from tokenizers.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    chars = set()

    with open(clean_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            chars.update(chunk)

    # Build vocab from sorted unique characters
    tokenizer.itos = sorted(chars)
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.itos)}
    return tokenizer


def encode_char_chunked(clean_path, tokenizer, output_path, chunk_bytes=50_000_000):
    """Encode text file to token IDs in chunks, appending to output binary file."""
    max_cp = max(ord(ch) for ch in tokenizer.stoi) + 1
    lookup = np.full(max_cp, -1, dtype=np.int32)
    for ch, idx in tokenizer.stoi.items():
        lookup[ord(ch)] = idx

    total_tokens = 0
    with open(clean_path, "r", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:
        while True:
            chunk = fin.read(chunk_bytes)
            if not chunk:
                break
            codepoints = np.array([ord(ch) for ch in chunk], dtype=np.int32)
            # Filter out characters with ordinals beyond our lookup table
            valid = codepoints[codepoints < max_cp]
            token_ids = lookup[valid]
            token_ids = token_ids[token_ids >= 0].astype(np.uint16)
            token_ids.tofile(fout)
            total_tokens += len(token_ids)

    return total_tokens


def _encode_bpe_segment(args):
    """Encode one segment of clean text with BPE (for multiprocessing)."""
    clean_path, start_byte, end_byte, segment_id, model_proto_hex = args
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(bytes.fromhex(model_proto_hex))

    out_path = os.path.join(DATA_DIR, f"encode_seg_{segment_id}.bin")
    chunk_size = 10_000_000
    total_tokens = 0

    with open(clean_path, "rb") as fin, open(out_path, "wb") as fout:
        fin.seek(start_byte)
        buffer = ""
        while fin.tell() < end_byte:
            remaining = end_byte - fin.tell()
            raw_bytes = fin.read(min(chunk_size, remaining))
            if not raw_bytes:
                break
            # Extend to newline if not at segment end
            if fin.tell() < end_byte:
                extra = fin.read(8192)
                if extra:
                    nl = extra.find(b"\n")
                    if nl != -1:
                        raw_bytes += extra[:nl + 1]
                        fin.seek(-(len(extra) - nl - 1), 1)
                    else:
                        raw_bytes += extra
            buffer += raw_bytes.decode("utf-8", errors="ignore")
            # Split on paragraph boundary
            last_para = buffer.rfind("\n\n")
            if last_para != -1:
                split_at = last_para + 2
            else:
                last_nl = buffer.rfind("\n")
                if last_nl != -1:
                    split_at = last_nl + 1
                elif len(buffer) > 3 * chunk_size:
                    split_at = len(buffer)
                else:
                    continue
            to_encode = buffer[:split_at]
            buffer = buffer[split_at:]
            ids = sp.encode(to_encode)
            np.array(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)

        if buffer:
            ids = sp.encode(buffer)
            np.array(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)

    seg_mb = (end_byte - start_byte) / (1024 * 1024)
    print(f"  Segment {segment_id}: {seg_mb:.0f} MB -> {total_tokens:,} tokens")
    return out_path, total_tokens


def encode_bpe_chunked(clean_path, tokenizer, output_path, chunk_bytes=10_000_000):
    """Encode text file with BPE, using parallel workers."""
    from multiprocessing import Pool, cpu_count
    total_bytes = get_file_size(clean_path)
    num_workers = min(cpu_count(), 16)

    print(f"  Encoding {total_bytes / 1024 / 1024:.0f} MB with {num_workers} parallel workers...")
    boundaries = _find_segment_boundaries(clean_path, num_workers)
    model_proto_hex = tokenizer.sp.serialized_model_proto().hex()
    args = [
        (clean_path, boundaries[i], boundaries[i + 1], i, model_proto_hex)
        for i in range(len(boundaries) - 1)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_encode_bpe_segment, args)

    # Concatenate segment bins in order
    total_tokens = 0
    with open(output_path, "wb") as fout:
        for seg_path, seg_tokens in results:
            total_tokens += seg_tokens
            with open(seg_path, "rb") as fin:
                while True:
                    chunk = fin.read(100 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.remove(seg_path)

    return total_tokens


def split_bin_file(all_tokens_path, train_path, val_path, val_ratio=0.1):
    """Split a single .bin file into train and val without loading into RAM."""
    total_bytes = get_file_size(all_tokens_path)
    total_tokens = total_bytes // 2  # uint16 = 2 bytes per token
    split_token = int(total_tokens * (1 - val_ratio))
    split_byte = split_token * 2

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Train: {split_token:,} tokens, Val: {total_tokens - split_token:,} tokens")

    # Copy chunks to train file
    chunk_size = 100 * 1024 * 1024  # 100 MB
    with open(all_tokens_path, "rb") as fin:
        with open(train_path, "wb") as fout:
            remaining = split_byte
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = fin.read(to_read)
                if not data:
                    break
                fout.write(data)
                remaining -= len(data)

        with open(val_path, "wb") as fout:
            while True:
                data = fin.read(chunk_size)
                if not data:
                    break
                fout.write(data)

    print(f"  Train: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val:   {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="char",
                        choices=["char", "bpe"],
                        help="Tokenizer type: 'char' (Level 1) or 'bpe' (Level 2)")
    args = parser.parse_args()

    # Step 1: Check raw text exists
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found!")
        print("Run 'python data/download_all.py' first to download the data.")
        sys.exit(1)

    raw_size = get_file_size(RAW_FILE)
    print(f"\n{'='*50}")
    print(f"  ArmGPT Data Preparation (memory-safe)")
    print(f"{'='*50}")
    print(f"  Raw text: {RAW_FILE} ({raw_size / 1024 / 1024:.0f} MB)")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"{'='*50}\n")

    # Step 2: Clean text in chunks -> clean_text.txt
    clean_path = os.path.join(DATA_DIR, "clean_text.txt")
    if os.path.exists(clean_path) and os.path.getmtime(clean_path) > os.path.getmtime(RAW_FILE):
        print("Step 1: Skipping cleaning (clean_text.txt is up to date)")
    else:
        print("Step 1: Cleaning text...")
        clean_file_chunked(RAW_FILE, clean_path)

    # Step 3: Build tokenizer
    print("\nStep 2: Building tokenizer...")
    all_tokens_path = os.path.join(DATA_DIR, "all_tokens.bin")

    if args.tokenizer == "char":
        tokenizer = build_char_vocab(clean_path)
        print(f"  Vocabulary: {tokenizer.vocab_size} characters")

        # Step 4: Encode in chunks
        print("\nStep 3: Encoding text...")
        total_tokens = encode_char_chunked(clean_path, tokenizer, all_tokens_path)
    else:
        from tokenizers.bpe_tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.train(clean_path, model_prefix=os.path.join(DATA_DIR, "bpe_model"))
        print(f"  Vocabulary: {tokenizer.vocab_size} tokens")

        # Step 4: Encode in chunks
        print("\nStep 3: Encoding text...")
        total_tokens = encode_bpe_chunked(clean_path, tokenizer, all_tokens_path)

    print(f"  Total tokens: {total_tokens:,}")

    # Step 5: Split into train/val
    print("\nStep 4: Splitting train/val (90/10)...")
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    split_bin_file(all_tokens_path, train_path, val_path)

    # Clean up temporary files
    os.remove(all_tokens_path)
    print(f"\n  Removed temporary file: all_tokens.bin")

    # Step 6: Save tokenizer
    tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    tokenizer.save(tok_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Data Preparation Complete!")
    print(f"{'='*50}")
    print(f"  Tokenizer:    {args.tokenizer}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")
    print(f"  Train tokens: {os.path.getsize(train_path) // 2:,} ({train_path})")
    print(f"  Val tokens:   {os.path.getsize(val_path) // 2:,} ({val_path})")
    print(f"  Train size:   {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val size:     {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")
    print(f"  Tokenizer:    {tok_path}")

    # Optional: delete clean_text.txt to save space
    clean_size = get_file_size(clean_path)
    print(f"\n  Note: clean_text.txt ({clean_size / 1024 / 1024:.0f} MB) can be deleted to save space:")
    print(f"    rm {clean_path}")

    # Show a sample encode/decode round-trip
    with open(clean_path, "r", encoding="utf-8") as f:
        sample = f.read(200)
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nSample round-trip test:")
    try:
        print(f"  Original:  {sample[:80]}...")
        print(f"  Decoded:   {decoded[:80]}...")
    except UnicodeEncodeError:
        print(f"  (contains non-ASCII characters)")
    print(f"  Encoded:   {encoded[:20]}...")
    print(f"  Match: {'YES' if sample == decoded else 'NO'}")


if __name__ == "__main__":
    main()
