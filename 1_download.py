"""
Step 1: Download all Armenian text data sources and merge into raw_text.txt.

Optimized for fast parallel downloads on machines with ample RAM (32+ GB).
HuggingFace sources download in parallel to separate files, then merge.

Requirements:
    pip install datasets requests mwxml

Usage:
    python 1_download.py              # download all sources
    python 1_download.py --skip wiki  # skip specific sources
    python 1_download.py --workers 3  # limit parallel HF downloads

Output:
    data/raw_text.txt  — concatenated raw Armenian text (~30 GB)
"""

import os
import sys
import gc
import bz2
import re
import time
import shutil
import argparse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
DUMP_FILE = os.path.join(DATA_DIR, "hywiki-latest-pages-articles.xml.bz2")
WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/hywiki/latest/"
    "hywiki-latest-pages-articles.xml.bz2"
)

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

def _download_wiki_dump():
    """Download the Armenian Wikipedia dump if not already present."""
    if os.path.exists(DUMP_FILE):
        print(f"  Dump already downloaded: {DUMP_FILE}")
        return

    print(f"  Downloading Armenian Wikipedia dump (~500 MB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(WIKI_DUMP_URL, DUMP_FILE, reporthook=progress_hook)
    print()


def _strip_wiki_markup(text):
    """Remove MediaWiki markup, keeping plain Armenian content."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[\[(?:Категория|Category|Պատկdelays|File|Файл|Image):[^\]]*\]\]", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}(.+?)={2,}", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)
    text = re.sub(r"^\s*[|!].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*[*#:;]+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_wiki_articles(dump_path):
    """Yield cleaned article bodies from the bz2 Wikipedia XML dump."""
    current_text = []
    in_text = False
    article_count = 0

    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            if "<text" in line:
                in_text = True
                match = re.search(r"<text[^>]*>(.*)", line)
                if match:
                    current_text.append(match.group(1))
                continue

            if in_text:
                if "</text>" in line:
                    current_text.append(line.split("</text>")[0])
                    full_text = "\n".join(current_text)
                    current_text = []
                    in_text = False

                    if full_text.startswith("#REDIRECT") or \
                       full_text.startswith("#ՎԵՐԱՀՂՈՒՄ") or \
                       len(full_text) < 200:
                        continue

                    cleaned = _strip_wiki_markup(full_text)
                    if len(cleaned) > 100:
                        article_count += 1
                        if article_count % 10000 == 0:
                            print(f"    Extracted {article_count} articles...")
                        yield cleaned

    print(f"  Total articles extracted: {article_count}")


def download_wikipedia():
    """Download Armenian Wikipedia and extract articles."""
    marker = os.path.join(DATA_DIR, ".wiki_done")
    out_file = os.path.join(DATA_DIR, "wiki_hy.txt")
    if os.path.exists(marker):
        print("  Wikipedia: already done, skipping.")
        return out_file

    _download_wiki_dump()

    print("  Extracting articles...")
    chars = 0
    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as out:
        for article_text in _extract_wiki_articles(DUMP_FILE):
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
                      split="train", streaming=True)

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
# Corpus download (default mode)
# ---------------------------------------------------------------------------

def download_corpus(args):
    skip = set(s.lower() for s in args.skip)

    # HuggingFace sources — all downloaded in parallel
    hf_sources = {
        "culturax": ("uonlp/CulturaX", "hy", "text"),
        "oscar": ("oscar-corpus/OSCAR-2301", "hy", "text"),
        "mc4": ("allenai/c4", "hy", "text"),
        "hplt": ("HPLT/HPLT2.0_cleaned", "hye_Armn", "text"),
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
    print(f"\n  Next step: python 2_prepare.py")


# ---------------------------------------------------------------------------
# Q&A download (--qa mode)
# ---------------------------------------------------------------------------

def download_qa(args):
    """Fetch all SFT source JSONs into data/ for later merging + tokenizing."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.fetch_armbench import fetch_armbench_qa
    from data.fetch_aya_armenian import fetch_aya_qa

    skip = set(s.lower() for s in args.skip)

    print(f"{'='*60}")
    print(f"  ArmGPT Q&A Download (SFT sources)")
    print(f"{'='*60}")
    print(f"  Output dir: {DATA_DIR}")
    if skip:
        print(f"  Skipping: {', '.join(skip)}")
    print(f"{'='*60}\n")

    t_start = time.time()

    if "armbench" not in skip:
        print("[ARMBENCH] Native Armenian exam + civics QA")
        fetch_armbench_qa(
            train_output_path=os.path.join(DATA_DIR, "armbench_train.json"),
            eval_output_path=os.path.join(DATA_DIR, "armbench_eval.json"),
        )
    else:
        print("[SKIP] ArmBench")

    if "aya" not in skip:
        print("\n[AYA] Filtered Armenian slice of Aya collection")
        fetch_aya_qa(
            output_path=os.path.join(DATA_DIR, "aya_armenian.json"),
        )
    else:
        print("[SKIP] Aya")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Q&A Download Complete!  ({fmt_time(elapsed)})")
    print(f"{'='*60}")
    print(f"  Optional: also run data/generate_armenian_qa.py to add")
    print(f"            Claude-generated pairs to data/armenian_qa.json")
    print(f"\n  Next step: python 2_prepare.py --qa")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Armenian text data (corpus by default, or --qa for SFT sources)"
    )
    parser.add_argument("--qa", action="store_true",
                        help="Download SFT Q&A sources (ArmBench + Aya) instead of raw corpus")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Sources to skip. Corpus: wiki cc100 culturax oscar mc4 hplt glot500. "
                             "QA: armbench aya")
    parser.add_argument("--workers", type=int, default=5,
                        help="Max parallel HF downloads (corpus mode only; default: 5)")
    args = parser.parse_args()

    if args.qa:
        download_qa(args)
    else:
        download_corpus(args)


if __name__ == "__main__":
    main()
