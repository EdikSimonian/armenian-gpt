"""
Download and merge CC-100 Armenian data with existing Wikipedia data.

CC-100 is a large web-crawled dataset (~776 MB compressed, ~4.9 GB text).
Combined with Wikipedia, this gives much more training data for better quality.

Usage:
    python data/download_cc100.py
"""

import os
import sys
import lzma
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CC100_URL = "https://data.statmt.org/cc-100/hy.txt.xz"
CC100_XZ = os.path.join(DATA_DIR, "cc100_hy.txt.xz")
CC100_TXT = os.path.join(DATA_DIR, "cc100_hy.txt")
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")


def download():
    """Download CC-100 Armenian if not present."""
    if os.path.exists(CC100_TXT):
        print(f"CC-100 already extracted: {CC100_TXT}")
        return
    if os.path.exists(CC100_XZ):
        print(f"CC-100 already downloaded: {CC100_XZ}")
    else:
        print(f"Downloading CC-100 Armenian (~776 MB)...")
        print(f"URL: {CC100_URL}")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(CC100_URL, CC100_XZ, reporthook=progress)
        print("\nDownload complete!")

    # Decompress
    print("Decompressing (this takes a few minutes)...")
    with lzma.open(CC100_XZ, "rt", encoding="utf-8") as f_in, \
         open(CC100_TXT, "w", encoding="utf-8") as f_out:
        chars = 0
        for line in f_in:
            f_out.write(line)
            chars += len(line)
            if chars % 100_000_000 == 0:
                print(f"  {chars / 1_000_000:.0f}M characters...")
    print(f"  Decompressed: {chars:,} characters")


def merge():
    """Append CC-100 data to raw_text.txt."""
    if not os.path.exists(CC100_TXT):
        print("Error: CC-100 text not found. Run download first.")
        sys.exit(1)

    wiki_size = os.path.getsize(RAW_FILE) if os.path.exists(RAW_FILE) else 0
    cc100_size = os.path.getsize(CC100_TXT)
    print(f"\nWikipedia data:  {wiki_size / 1024 / 1024:.0f} MB")
    print(f"CC-100 data:     {cc100_size / 1024 / 1024:.0f} MB")

    # Check if already merged (raw_text.txt much larger than wiki alone)
    if wiki_size > 3_000_000_000:  # > 3 GB means likely already merged
        print("Data appears already merged (raw_text.txt > 3 GB). Skipping.")
        return

    print("Appending CC-100 to raw_text.txt...")
    with open(RAW_FILE, "a", encoding="utf-8") as out, \
         open(CC100_TXT, "r", encoding="utf-8") as inp:
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
