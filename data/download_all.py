"""
Download all Armenian text data sources and merge into raw_text.txt.

Optimized for fast parallel downloads on machines with ample RAM (32+ GB).
HuggingFace sources download in parallel to separate files, then merge.

Requirements:
    pip install datasets requests mwxml

Usage:
    python data/download_all.py              # download all sources
    python data/download_all.py --skip wiki  # skip specific sources
    python data/download_all.py --workers 3  # limit parallel HF downloads
"""

import os
import sys
import gc
import time
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")

# HuggingFace cache location
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

# I/O buffer size — large buffers = fewer syscalls = faster merging
IO_BUFFER = 16 * 1024 * 1024  # 16 MB


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


def fmt_time(seconds):
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def get_file_size_mb(path):
    """Get file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def download_wikipedia():
    """Download Armenian Wikipedia and extract articles."""
    marker = os.path.join(DATA_DIR, ".wiki_done")
    out_file = os.path.join(DATA_DIR, "wiki_hy.txt")
    if os.path.exists(marker):
        print("  Wikipedia: already done, skipping.")
        return out_file

    sys.path.insert(0, DATA_DIR)
    from download import download_dump, extract_articles, DUMP_FILE

    download_dump()

    print("  Extracting articles...")
    chars = 0
    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as out:
        for article_text in extract_articles(DUMP_FILE):
            out.write(article_text)
            out.write("\n\n")
            chars += len(article_text)

    print(f"  Wikipedia: {chars:,} chars ({chars / 1024 / 1024:.0f} MB)")
    with open(marker, "w") as f:
        f.write("done")
    return out_file


# ---------------------------------------------------------------------------
# CC-100 (direct download, not HuggingFace)
# ---------------------------------------------------------------------------

def download_cc100():
    """Download and decompress CC-100 Armenian data."""
    marker = os.path.join(DATA_DIR, ".cc100_done")
    out_file = os.path.join(DATA_DIR, "cc100_hy.txt")
    if os.path.exists(marker):
        print("  CC-100: already done, skipping.")
        return out_file

    import lzma
    import urllib.request

    cc100_url = "https://data.statmt.org/cc-100/hy.txt.xz"
    cc100_xz = os.path.join(DATA_DIR, "cc100_hy.txt.xz")

    if not os.path.exists(cc100_xz) and not os.path.exists(out_file):
        print("  Downloading CC-100 (~776 MB)...")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  {pct:.1f}% ({mb:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(cc100_url, cc100_xz, reporthook=progress)
        print("\n  Download complete!")

    if not os.path.exists(out_file):
        print("  Decompressing CC-100 (this takes a few minutes)...")
        chars = 0
        # Use large read buffer for faster decompression
        with lzma.open(cc100_xz, "rt", encoding="utf-8") as f_in, \
             open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as f_out:
            while True:
                chunk = f_in.read(IO_BUFFER)
                if not chunk:
                    break
                f_out.write(chunk)
                chars += len(chunk)
                if chars % 500_000_000 == 0:
                    print(f"  {chars / 1_000_000_000:.1f}G chars...")

        print(f"  CC-100: {chars:,} chars decompressed")

        # Remove compressed file
        if os.path.exists(cc100_xz):
            os.remove(cc100_xz)

    with open(marker, "w") as f:
        f.write("done")
    return out_file


# ---------------------------------------------------------------------------
# HuggingFace streaming download (runs as subprocess for parallelism)
# ---------------------------------------------------------------------------

def _download_hf_worker(args):
    """
    Worker function for parallel HF downloads.
    Runs in a separate process. Downloads one dataset via streaming
    and writes to its own text file.
    """
    name, dataset_id, lang_config, text_field, out_file, marker_file = args

    if os.path.exists(marker_file):
        return name, out_file, 0, 0, True

    # Reimport in subprocess
    from datasets import load_dataset

    print(f"  [{name}] Starting download ({dataset_id})...")
    t0 = time.time()

    ds = load_dataset(dataset_id, lang_config,
                      split="train", streaming=True, trust_remote_code=True)

    chars = 0
    docs = 0
    # Buffer writes — collect ~5 MB before flushing to disk
    write_buf = []
    buf_size = 0
    FLUSH_THRESHOLD = 5 * 1024 * 1024  # 5 MB

    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as f:
        for example in ds:
            text = example[text_field].strip()
            if len(text) < 50:
                continue
            write_buf.append(text)
            write_buf.append("\n\n")
            buf_size += len(text) + 2
            chars += len(text)
            docs += 1

            if buf_size >= FLUSH_THRESHOLD:
                f.write("".join(write_buf))
                write_buf.clear()
                buf_size = 0

            if docs % 100_000 == 0:
                elapsed = time.time() - t0
                rate = chars / elapsed / 1_000_000 if elapsed > 0 else 0
                print(f"  [{name}] {docs:,} docs, {chars / 1_000_000:.0f}M chars "
                      f"({rate:.1f} MB/s)")

        # Flush remaining
        if write_buf:
            f.write("".join(write_buf))

    elapsed = time.time() - t0
    print(f"  [{name}] Done: {docs:,} docs, {chars / 1024 / 1024:.0f} MB "
          f"in {fmt_time(elapsed)}")

    with open(marker_file, "w") as f:
        f.write("done")

    return name, out_file, docs, chars, False


# ---------------------------------------------------------------------------
# Merge files into raw_text.txt
# ---------------------------------------------------------------------------

def merge_files(file_list, output_path):
    """Merge multiple text files into one, using large I/O buffers."""
    print(f"\nMerging {len(file_list)} files into {os.path.basename(output_path)}...")
    t0 = time.time()
    total_bytes = 0

    with open(output_path, "wb", buffering=IO_BUFFER) as fout:
        for i, src_path in enumerate(file_list):
            if not os.path.exists(src_path):
                print(f"  Warning: {src_path} not found, skipping")
                continue

            src_size = os.path.getsize(src_path) / (1024 * 1024)
            print(f"  [{i+1}/{len(file_list)}] {os.path.basename(src_path)} "
                  f"({src_size:.0f} MB)...")

            with open(src_path, "rb", buffering=IO_BUFFER) as fin:
                while True:
                    chunk = fin.read(IO_BUFFER)
                    if not chunk:
                        break
                    fout.write(chunk)
                    total_bytes += len(chunk)

            # Add separator between sources
            sep = b"\n\n"
            fout.write(sep)
            total_bytes += len(sep)

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)
    rate = total_mb / elapsed if elapsed > 0 else 0
    print(f"  Merged {total_mb:.0f} MB in {fmt_time(elapsed)} ({rate:.0f} MB/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download all Armenian text data")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Sources to skip (e.g. --skip wiki cc100)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Max parallel HF downloads (default: 5)")
    args = parser.parse_args()

    skip = set(s.lower() for s in args.skip)

    # HuggingFace sources — all downloaded in parallel
    hf_sources = {
        "culturax": ("uonlp/CulturaX", "hy", "text"),
        "oscar": ("oscar-corpus/OSCAR-2301", "hy", "text"),
        "mc4": ("allenai/c4", "hy", "text"),
        "hplt": ("HPLT/HPLT2.0_cleaned", "hy", "text"),
        "glot500": ("cis-lmu/Glot500", "hye_Armn", "text"),
    }

    print(f"{'='*60}")
    print(f"  ArmGPT Data Download — Parallel Mode")
    print(f"{'='*60}")
    print(f"  Output: {RAW_FILE}")
    print(f"  HF workers: {args.workers}")
    if skip:
        print(f"  Skipping: {', '.join(skip)}")
    print(f"{'='*60}\n")

    t_start = time.time()
    source_files = []  # ordered list of files to merge

    # ---- Phase 1: Wikipedia + CC-100 (sequential, direct downloads) ----
    print("=" * 40)
    print("Phase 1: Direct downloads")
    print("=" * 40)

    if "wiki" not in skip:
        print("\n[WIKI] Armenian Wikipedia (~1.5 GB)")
        wiki_file = download_wikipedia()
        source_files.append(wiki_file)
    else:
        print("[SKIP] Wikipedia")

    if "cc100" not in skip:
        print("\n[CC100] CC-100 Armenian (~4.9 GB)")
        cc100_file = download_cc100()
        source_files.append(cc100_file)
    else:
        print("[SKIP] CC-100")

    # ---- Phase 2: HuggingFace sources (parallel downloads) ----
    hf_to_download = {k: v for k, v in hf_sources.items() if k not in skip}

    if hf_to_download:
        print(f"\n{'='*40}")
        print(f"Phase 2: HuggingFace downloads ({len(hf_to_download)} sources in parallel)")
        print(f"{'='*40}")

        # Prepare worker args
        worker_args = []
        hf_file_map = {}
        for name, (dataset_id, lang, text_field) in hf_to_download.items():
            out_file = os.path.join(DATA_DIR, f"{name}_hy.txt")
            marker = os.path.join(DATA_DIR, f".{name}_done")
            worker_args.append((name, dataset_id, lang, text_field, out_file, marker))
            hf_file_map[name] = out_file

        n_workers = min(args.workers, len(worker_args))
        print(f"  Launching {n_workers} parallel workers...\n")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_download_hf_worker, wa): wa[0]
                for wa in worker_args
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    name, out_file, docs, chars, was_cached = future.result()
                    if was_cached:
                        print(f"  [{name}] Already downloaded, skipping.")
                    else:
                        print(f"  [{name}] Complete: {docs:,} docs, "
                              f"{chars / 1024 / 1024:.0f} MB")
                except Exception as e:
                    print(f"  [{name}] ERROR: {e}")
                    print(f"  [{name}] Will be skipped in merge.")

        # Add HF files in consistent order (only if marker exists = completed)
        for name in ["culturax", "oscar", "mc4", "hplt", "glot500"]:
            if name in hf_file_map:
                f = hf_file_map[name]
                marker = os.path.join(DATA_DIR, f".{name}_done")
                if os.path.exists(f) and os.path.exists(marker):
                    source_files.append(f)
                elif os.path.exists(f):
                    print(f"  [{name}] Incomplete download (no marker), skipping.")

    for name in hf_sources:
        if name in skip:
            print(f"[SKIP] {name}")

    # ---- Phase 3: Merge all source files into raw_text.txt ----
    print(f"\n{'='*40}")
    print(f"Phase 3: Merge")
    print(f"{'='*40}")

    if source_files:
        merge_files(source_files, RAW_FILE)
    else:
        print("  No source files to merge!")

    # ---- Phase 4: Cleanup ----
    print(f"\n{'='*40}")
    print(f"Phase 4: Cleanup")
    print(f"{'='*40}")

    # Clear HF cache
    clear_hf_cache()

    # Remove individual source text files (now merged into raw_text.txt)
    freed = 0
    for f in source_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            os.remove(f)
            freed += size
            print(f"  Removed {os.path.basename(f)} ({size / 1024 / 1024:.0f} MB)")
    if freed > 0:
        print(f"  Freed {freed / 1024 / 1024 / 1024:.1f} GB")

    # Remove leftover compressed files
    for f in os.listdir(DATA_DIR):
        if f.endswith((".xz", ".bz2")) and os.path.isfile(os.path.join(DATA_DIR, f)):
            path = os.path.join(DATA_DIR, f)
            print(f"  Removing {f} ({os.path.getsize(path) / 1024 / 1024:.0f} MB)")
            os.remove(path)

    # ---- Summary ----
    elapsed = time.time() - t_start
    final_size = get_file_size_mb(RAW_FILE)
    print(f"\n{'='*60}")
    print(f"  Download Complete!")
    print(f"{'='*60}")
    print(f"  raw_text.txt: {final_size:.0f} MB ({final_size / 1024:.1f} GB)")
    print(f"  Total time:   {fmt_time(elapsed)}")
    print(f"\n  Next step: python data/prepare.py --tokenizer bpe")


if __name__ == "__main__":
    main()
