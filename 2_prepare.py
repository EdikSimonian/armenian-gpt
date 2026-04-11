"""
Step 2: Prepare the data for tokenization.

Default mode (corpus): Cleans data/text/train/raw_text.txt (produced by
1_download.py) and writes a normalized version to data/text/train/clean_text.txt.
Cleaning keeps only Armenian letters, ASCII punctuation, digits, and
whitespace (NFC normalized).

Memory-safe: processes text in parallel segments, each worker handling its
segment in 50 MB chunks. Safe for 20+ GB inputs.

--qa mode: Merges the SFT JSON files under data/text/finetune/ produced by
1_download.py --qa (armbench_{train,eval}.json + aya_armenian.json, plus
an optional Claude-generated armenian_qa.json) into a single deduplicated
data/text/finetune/qa_merged.json.

Usage:
    python 2_prepare.py           # corpus mode
    python 2_prepare.py --qa      # Q&A mode

Outputs:
    corpus: data/text/train/clean_text.txt
    qa:     data/text/finetune/qa_merged.json
"""

import os
import re
import sys
import unicodedata
from multiprocessing import Pool, cpu_count

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
TEXT_TRAIN_DIR = os.path.join(DATA_DIR, "text", "train")
TEXT_FINETUNE_DIR = os.path.join(DATA_DIR, "text", "finetune")
RAW_FILE = os.path.join(TEXT_TRAIN_DIR, "raw_text.txt")
CLEAN_FILE = os.path.join(TEXT_TRAIN_DIR, "clean_text.txt")

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
    out_path = os.path.join(TEXT_TRAIN_DIR, f"clean_seg_{segment_id}.txt")
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


def prepare_corpus():
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


def prepare_qa():
    """Merge the SFT source JSONs into data/text/finetune/qa_merged.json."""
    from core.merge_sft_sources import merge_sft_sources

    # Inputs in priority order — earlier sources win dedup ties.
    # armenian_qa.json is only present if the user ran the optional Claude
    # generator (core/generate_armenian_qa.py); it's merged in first so its
    # hand-crafted pairs take priority over the larger translated sets.
    input_paths = [
        os.path.join(TEXT_FINETUNE_DIR, "armenian_qa.json"),    # optional, Claude
        os.path.join(TEXT_FINETUNE_DIR, "armbench_train.json"), # native exam QA
        os.path.join(TEXT_FINETUNE_DIR, "aya_armenian.json"),   # filtered Aya
    ]
    output_path = os.path.join(TEXT_FINETUNE_DIR, "qa_merged.json")

    print(f"\n{'='*50}")
    print(f"  Step 2: Merge SFT sources (--qa)")
    print(f"{'='*50}")

    n = merge_sft_sources(input_paths, output_path)
    if n == 0:
        print(f"\nError: no Q&A sources found in {TEXT_FINETUNE_DIR}.")
        print("Run 'python 1_download.py --qa' first.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Step 2 (--qa) complete: {n:,} unique pairs")
    print(f"{'='*50}")
    print("Next step: python 3_tokenize.py --qa --tokenizer bpe")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", action="store_true",
                        help="Merge SFT Q&A JSON files instead of cleaning raw text")
    args = parser.parse_args()

    if args.qa:
        prepare_qa()
    else:
        prepare_corpus()


if __name__ == "__main__":
    main()
