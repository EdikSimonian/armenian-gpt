"""
Download Glot500 Armenian data and merge with existing training text.

Glot500 is an academic multilingual dataset covering 500+ languages.
Armenian subset is ~200-500 MB — smaller but curated quality.

Requires: pip install datasets

Usage:
    python data/download_glot500.py
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
GLOT_TXT = os.path.join(DATA_DIR, "glot500_hy.txt")


def download():
    """Download Glot500 Armenian via HuggingFace datasets."""
    if os.path.exists(GLOT_TXT):
        print(f"Glot500 already downloaded: {GLOT_TXT}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print("Downloading Glot500 Armenian from HuggingFace...")
    print("  Dataset: cis-lmu/Glot500")
    print("  Filtering for Armenian (hye_Armn)...")

    ds = load_dataset("cis-lmu/Glot500", "hye_Armn",
                      split="train", trust_remote_code=True)

    print(f"  Downloaded {len(ds):,} documents")
    print(f"  Writing to {GLOT_TXT}...")

    chars = 0
    with open(GLOT_TXT, "w", encoding="utf-8") as f:
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
    """Append Glot500 data to raw_text.txt."""
    if not os.path.exists(GLOT_TXT):
        print("Error: Glot500 text not found. Run download first.")
        sys.exit(1)

    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found. Run download.py first.")
        sys.exit(1)

    glot_size = os.path.getsize(GLOT_TXT)
    raw_size = os.path.getsize(RAW_FILE)
    print(f"\nExisting data:   {raw_size / 1024 / 1024:.0f} MB")
    print(f"Glot500 data:    {glot_size / 1024 / 1024:.0f} MB")

    print("Appending Glot500 to raw_text.txt...")
    with open(RAW_FILE, "a", encoding="utf-8") as out, \
         open(GLOT_TXT, "r", encoding="utf-8") as inp:
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
