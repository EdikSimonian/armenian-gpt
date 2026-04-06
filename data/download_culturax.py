"""
Download CulturaX Armenian data and merge with existing training text.

CulturaX is a cleaned, deduplicated combination of mC4 and OSCAR data.
Higher quality than using either source alone.

Requires: pip install datasets

Usage:
    python data/download_culturax.py
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
CULTURAX_TXT = os.path.join(DATA_DIR, "culturax_hy.txt")


def download():
    """Download CulturaX Armenian via HuggingFace datasets."""
    if os.path.exists(CULTURAX_TXT):
        print(f"CulturaX already downloaded: {CULTURAX_TXT}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print("Downloading CulturaX Armenian from HuggingFace...")
    print("  Dataset: uonlp/CulturaX")
    print("  This may take a while (~5-8 GB of text)...")

    ds = load_dataset("uonlp/CulturaX", "hy",
                      split="train", trust_remote_code=True)

    print(f"  Downloaded {len(ds):,} documents")
    print(f"  Writing to {CULTURAX_TXT}...")

    chars = 0
    with open(CULTURAX_TXT, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            text = example["text"].strip()
            if len(text) < 50:
                continue
            f.write(text)
            f.write("\n\n")
            chars += len(text)
            if (i + 1) % 100_000 == 0:
                print(f"  {i + 1:,} documents, {chars / 1_000_000:.0f}M characters...")

    print(f"  Done: {chars:,} characters ({chars / 1024 / 1024:.0f} MB)")


def merge():
    """Append CulturaX data to raw_text.txt."""
    if not os.path.exists(CULTURAX_TXT):
        print("Error: CulturaX text not found. Run download first.")
        sys.exit(1)

    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found. Run download.py first.")
        sys.exit(1)

    culturax_size = os.path.getsize(CULTURAX_TXT)
    raw_size = os.path.getsize(RAW_FILE)
    print(f"\nExisting data:   {raw_size / 1024 / 1024:.0f} MB")
    print(f"CulturaX data:   {culturax_size / 1024 / 1024:.0f} MB")

    print("Appending CulturaX to raw_text.txt...")
    with open(RAW_FILE, "a", encoding="utf-8") as out, \
         open(CULTURAX_TXT, "r", encoding="utf-8") as inp:
        out.write("\n\n")
        chars = 0
        for line in inp:
            out.write(line)
            chars += len(line)
            if chars % 100_000_000 == 0:
                print(f"  {chars / 1_000_000:.0f}M characters appended...")

    final_size = os.path.getsize(RAW_FILE)
    print(f"\nMerged! raw_text.txt is now {final_size / 1024 / 1024:.0f} MB")


def main():
    download()
    merge()
    print("\nNext step: python data/prepare.py --tokenizer bpe")


if __name__ == "__main__":
    main()
