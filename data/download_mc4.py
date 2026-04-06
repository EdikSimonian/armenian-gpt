"""
Download mC4 Armenian data and merge with existing training text.

mC4 is Google's multilingual cleaned Common Crawl dataset.
Armenian subset is ~5-15 GB of web text.

Requires: pip install datasets

Usage:
    python data/download_mc4.py
"""

import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
MC4_TXT = os.path.join(DATA_DIR, "mc4_hy.txt")


def download():
    """Download mC4 Armenian via HuggingFace datasets."""
    if os.path.exists(MC4_TXT):
        print(f"mC4 already downloaded: {MC4_TXT}")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print("Downloading mC4 Armenian from HuggingFace...")
    print("  Dataset: allenai/c4 (multilingual, hy)")
    print("  This may take a while (~5-15 GB of text)...")

    ds = load_dataset("allenai/c4", "hy",
                      split="train", trust_remote_code=True)

    print(f"  Downloaded {len(ds):,} documents")
    print(f"  Writing to {MC4_TXT}...")

    chars = 0
    with open(MC4_TXT, "w", encoding="utf-8") as f:
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
    """Append mC4 data to raw_text.txt."""
    if not os.path.exists(MC4_TXT):
        print("Error: mC4 text not found. Run download first.")
        sys.exit(1)

    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found. Run download.py first.")
        sys.exit(1)

    mc4_size = os.path.getsize(MC4_TXT)
    raw_size = os.path.getsize(RAW_FILE)
    print(f"\nExisting data:   {raw_size / 1024 / 1024:.0f} MB")
    print(f"mC4 data:        {mc4_size / 1024 / 1024:.0f} MB")

    print("Appending mC4 to raw_text.txt...")
    with open(RAW_FILE, "a", encoding="utf-8") as out, \
         open(MC4_TXT, "r", encoding="utf-8") as inp:
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
