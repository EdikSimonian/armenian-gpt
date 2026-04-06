"""
Download all Armenian text data sources and merge into raw_text.txt.

Memory-safe: uses HuggingFace streaming to avoid loading datasets into RAM.
Cleans up HuggingFace cache after each source to save disk space.

Requirements:
    pip install datasets requests mwxml

Usage:
    python data/download_all.py              # download all sources
    python data/download_all.py --skip wiki  # skip specific sources
"""

import os
import sys
import gc
import shutil
import argparse

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")

# HuggingFace cache location
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def clear_hf_cache():
    """Delete HuggingFace cache to free disk space."""
    if os.path.exists(HF_CACHE):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(HF_CACHE)
            for f in fnames
        ) / (1024 ** 3)
        print(f"  Clearing HuggingFace cache ({size:.1f} GB)...")
        shutil.rmtree(HF_CACHE, ignore_errors=True)
    gc.collect()


def get_raw_size():
    """Get current size of raw_text.txt in MB."""
    if os.path.exists(RAW_FILE):
        return os.path.getsize(RAW_FILE) / (1024 * 1024)
    return 0


def download_wikipedia():
    """Download Armenian Wikipedia and extract articles."""
    marker = os.path.join(DATA_DIR, ".wiki_done")
    if os.path.exists(marker):
        print("  Wikipedia: already downloaded, skipping.")
        return

    # Use existing download.py logic
    sys.path.insert(0, DATA_DIR)
    from download import download_dump, extract_articles, DUMP_FILE

    download_dump()

    print("  Extracting articles to raw_text.txt...")
    chars = 0
    with open(RAW_FILE, "w", encoding="utf-8") as out:
        for article_text in extract_articles(DUMP_FILE):
            out.write(article_text)
            out.write("\n\n")
            chars += len(article_text)

    print(f"  Wikipedia: {chars:,} characters ({chars / 1024 / 1024:.0f} MB)")

    # Mark done and clean up dump
    with open(marker, "w") as f:
        f.write("done")


def download_cc100():
    """Download CC-100 Armenian data."""
    marker = os.path.join(DATA_DIR, ".cc100_done")
    if os.path.exists(marker):
        print("  CC-100: already downloaded, skipping.")
        return

    import lzma
    import urllib.request

    cc100_url = "https://data.statmt.org/cc-100/hy.txt.xz"
    cc100_xz = os.path.join(DATA_DIR, "cc100_hy.txt.xz")

    # Download compressed file
    if not os.path.exists(cc100_xz):
        print(f"  Downloading CC-100 Armenian (~776 MB)...")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  {pct:.1f}% ({mb:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(cc100_url, cc100_xz, reporthook=progress)
        print("\n  Download complete!")

    # Decompress and append line by line (memory-safe)
    print("  Decompressing and appending to raw_text.txt...")
    chars = 0
    with lzma.open(cc100_xz, "rt", encoding="utf-8") as f_in, \
         open(RAW_FILE, "a", encoding="utf-8") as f_out:
        f_out.write("\n\n")
        for line in f_in:
            f_out.write(line)
            chars += len(line)
            if chars % 500_000_000 == 0:
                print(f"  {chars / 1_000_000_000:.1f}G characters...")

    print(f"  CC-100: {chars:,} characters appended")

    # Clean up compressed file
    os.remove(cc100_xz)
    with open(marker, "w") as f:
        f.write("done")


def download_hf_streaming(name, dataset_id, lang_config, text_field="text"):
    """
    Download a HuggingFace dataset using streaming (no RAM explosion).
    Appends directly to raw_text.txt line by line.
    """
    marker = os.path.join(DATA_DIR, f".{name}_done")
    if os.path.exists(marker):
        print(f"  {name}: already downloaded, skipping.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed!")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print(f"  Downloading {name} ({dataset_id}, lang={lang_config})...")
    print(f"  Using streaming mode (low RAM usage)...")

    ds = load_dataset(dataset_id, lang_config,
                      split="train", streaming=True, trust_remote_code=True)

    chars = 0
    docs = 0
    with open(RAW_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n")
        for example in ds:
            text = example[text_field].strip()
            if len(text) < 50:
                continue
            f.write(text)
            f.write("\n\n")
            chars += len(text)
            docs += 1
            if docs % 100_000 == 0:
                print(f"  {name}: {docs:,} docs, {chars / 1_000_000:.0f}M characters...")
                # Flush to disk periodically
                f.flush()

    print(f"  {name}: {docs:,} docs, {chars:,} characters ({chars / 1024 / 1024:.0f} MB)")

    # Clean up HF cache after each source
    clear_hf_cache()
    gc.collect()

    with open(marker, "w") as f:
        f.write("done")


def main():
    parser = argparse.ArgumentParser(description="Download all Armenian text data")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Sources to skip (e.g. --skip wiki cc100)")
    args = parser.parse_args()

    skip = set(s.lower() for s in args.skip)

    sources = [
        ("wiki", "Armenian Wikipedia (~1.5 GB)", download_wikipedia),
        ("cc100", "CC-100 Armenian (~4.9 GB)", download_cc100),
        ("culturax", "CulturaX Armenian (~5-8 GB)", None),
        ("oscar", "OSCAR Armenian (~5-8 GB)", None),
        ("mc4", "mC4 Armenian (~5-15 GB)", None),
        ("hplt", "HPLT Armenian (~2-5 GB)", None),
        ("glot500", "Glot500 Armenian (~0.2-0.5 GB)", None),
    ]

    # HuggingFace streaming sources
    hf_sources = {
        "culturax": ("uonlp/CulturaX", "hy", "text"),
        "oscar": ("oscar-corpus/OSCAR-2301", "hy", "text"),
        "mc4": ("allenai/c4", "hy", "text"),
        "hplt": ("HPLT/HPLT2.0_cleaned", "hy", "text"),
        "glot500": ("cis-lmu/Glot500", "hye_Armn", "text"),
    }

    print(f"{'='*60}")
    print(f"  ArmGPT Data Download — All Sources")
    print(f"{'='*60}")
    print(f"  Output: {RAW_FILE}")
    print(f"  Current size: {get_raw_size():.0f} MB")
    if skip:
        print(f"  Skipping: {', '.join(skip)}")
    print(f"{'='*60}\n")

    for key, desc, func in sources:
        if key in skip:
            print(f"[SKIP] {desc}")
            continue

        print(f"\n[{key.upper()}] {desc}")

        if func:
            # Custom download function (wiki, cc100)
            func()
        elif key in hf_sources:
            # HuggingFace streaming download
            dataset_id, lang, text_field = hf_sources[key]
            download_hf_streaming(key, dataset_id, lang, text_field)

        print(f"  raw_text.txt is now {get_raw_size():.0f} MB")

    # Final cleanup
    print(f"\n{'='*60}")
    print(f"  Final cleanup")
    print(f"{'='*60}")
    clear_hf_cache()

    # Remove any leftover compressed/intermediate files
    for f in os.listdir(DATA_DIR):
        if f.endswith((".xz", ".bz2")) and os.path.isfile(os.path.join(DATA_DIR, f)):
            path = os.path.join(DATA_DIR, f)
            print(f"  Removing {f} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")
            os.remove(path)

    final_size = get_raw_size()
    print(f"\n{'='*60}")
    print(f"  Download Complete!")
    print(f"{'='*60}")
    print(f"  raw_text.txt: {final_size:.0f} MB ({final_size / 1024:.1f} GB)")
    print(f"\n  Next step: python data/prepare.py --tokenizer bpe")


if __name__ == "__main__":
    main()
