"""
Step 2: Clean raw Armenian text.

Reads data/raw_text.txt (produced by 1_download.py) and writes a cleaned
version to data/clean_text.txt. Cleaning keeps only Armenian letters,
ASCII punctuation, digits, and whitespace, and normalizes Unicode to NFC.

Memory-safe: processes text in parallel segments, each worker handling its
segment in 50 MB chunks. Safe for 20+ GB inputs.

Usage:
    python 2_prepare.py

After running, you'll have:
    data/clean_text.txt  — normalized Armenian-only text ready for 3_tokenize.py
"""

import os
import re
import sys
import unicodedata
from multiprocessing import Pool, cpu_count

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
CLEAN_FILE = os.path.join(DATA_DIR, "clean_text.txt")

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


def _find_segment_boundaries(path, num_segments):
    """Find byte offsets that split a file into segments at newline boundaries."""
    file_size = os.path.getsize(path)
    boundaries = [0]
    with open(path, "rb") as f:
        for i in range(1, num_segments):
            target = (file_size * i) // num_segments
            f.seek(target)
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
    chunk_size = 50_000_000
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


def clean_file_chunked(input_path, output_path):
    """Clean the raw text file in parallel, writing concatenated output."""
    total_bytes = os.path.getsize(input_path)
    total_mb = total_bytes / (1024 * 1024)
    num_workers = min(cpu_count(), 16)
    print(f"  Cleaning {total_mb:.0f} MB in parallel with {num_workers} workers...")

    boundaries = _find_segment_boundaries(input_path, num_workers)
    args = [
        (input_path, boundaries[i], boundaries[i + 1], i)
        for i in range(len(boundaries) - 1)
    ]
    with Pool(num_workers) as pool:
        results = pool.map(_clean_segment, args)

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


def main():
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found!")
        print("Run 'python 1_download.py' first to download the data.")
        sys.exit(1)

    raw_size = os.path.getsize(RAW_FILE)
    print(f"\n{'='*50}")
    print(f"  Step 2: Clean raw text")
    print(f"{'='*50}")
    print(f"  Input:  {RAW_FILE} ({raw_size / 1024 / 1024:.0f} MB)")
    print(f"  Output: {CLEAN_FILE}")
    print(f"{'='*50}\n")

    if os.path.exists(CLEAN_FILE) and os.path.getmtime(CLEAN_FILE) > os.path.getmtime(RAW_FILE):
        clean_size = os.path.getsize(CLEAN_FILE)
        print(f"Skipping — {CLEAN_FILE} is already up to date "
              f"({clean_size / 1024 / 1024:.0f} MB).")
        print("Next step: python 3_tokenize.py --tokenizer bpe")
        return

    clean_file_chunked(RAW_FILE, CLEAN_FILE)

    clean_size = os.path.getsize(CLEAN_FILE)
    print(f"\n{'='*50}")
    print(f"  Step 2 complete")
    print(f"  {CLEAN_FILE}: {clean_size / 1024 / 1024:.0f} MB")
    print(f"{'='*50}")
    print("Next step: python 3_tokenize.py --tokenizer bpe")


if __name__ == "__main__":
    main()
