"""
Download HPLT (High Performance Language Technologies) Armenian data.

HPLT is an EU-funded project providing cleaned web-crawled data
for under-resourced languages. Armenian subset is ~2-5 GB.

Requires: pip install datasets

Usage:
    python data/download_hplt.py
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
HPLT_TXT = os.path.join(DATA_DIR, "hplt_hy.txt")


def download():
    """Download HPLT Armenian via HuggingFace datasets."""
    if os.path.exists(HPLT_TXT):
        print(f"HPLT already downloaded: {HPLT_TXT}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print("Downloading HPLT Armenian from HuggingFace...")
    print("  Dataset: HPLT/HPLT2.0_cleaned")
    print("  This may take a while (~2-5 GB of text)...")

    ds = load_dataset("HPLT/HPLT2.0_cleaned", "hye_Armn",
                      split="train")

    print(f"  Downloaded {len(ds):,} documents")
    print(f"  Writing to {HPLT_TXT}...")

    chars = 0
    with open(HPLT_TXT, "w", encoding="utf-8") as f:
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
    """Append HPLT data to raw_text.txt."""
    if not os.path.exists(HPLT_TXT):
        print("Error: HPLT text not found. Run download first.")
        sys.exit(1)

    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found. Run download.py first.")
        sys.exit(1)

    hplt_size = os.path.getsize(HPLT_TXT)
    raw_size = os.path.getsize(RAW_FILE)
    print(f"\nExisting data:   {raw_size / 1024 / 1024:.0f} MB")
    print(f"HPLT data:       {hplt_size / 1024 / 1024:.0f} MB")

    print("Appending HPLT to raw_text.txt...")
    with open(RAW_FILE, "a", encoding="utf-8") as out, \
         open(HPLT_TXT, "r", encoding="utf-8") as inp:
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
