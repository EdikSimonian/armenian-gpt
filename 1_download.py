"""
Step 1: Download all Armenian text data — pretraining corpus AND fine-tuning Q&A.

This is the sole entry point for any download in the pipeline. It writes to
three top-level subtrees under ./data/:

    data/text/train/      — pretraining corpus sources (corpus mode, default)
    data/text/finetune/   — SFT Q&A JSONs (--qa mode)
    data/hf/              — HuggingFace cache shared by every download

HF_HOME is overridden at script startup so every `datasets.load_dataset` call
(and every huggingface_hub download) lands in data/hf/ rather than leaking
into ~/.cache/huggingface/.

Requirements:
    pip install datasets requests mwxml

Usage:
    python 1_download.py                  # corpus (wiki + cc100 + 5 HF sources)
    python 1_download.py --skip wiki      # skip Wikipedia
    python 1_download.py --workers 3      # limit parallel HF corpus downloads
    python 1_download.py --qa             # SFT Q&A sources (ArmBench + Aya)
    python 1_download.py --qa --skip aya  # only ArmBench

Outputs:
    corpus mode: data/text/train/raw_text.txt  (~30 GB concatenated)
    --qa mode:   data/text/finetune/armbench_train.json
                 data/text/finetune/armbench_eval.json
                 data/text/finetune/aya_armenian.json
"""

import os

# -----------------------------------------------------------------------------
# HF cache redirection — MUST happen before any datasets/huggingface_hub import,
# including imports inside subprocess workers (env vars propagate to children).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
TEXT_DIR = os.path.join(DATA_DIR, "text")
TEXT_TRAIN_DIR = os.path.join(TEXT_DIR, "train")
TEXT_FINETUNE_DIR = os.path.join(TEXT_DIR, "finetune")
HF_CACHE_DIR = os.path.join(DATA_DIR, "hf")

for _d in (DATA_DIR, TEXT_DIR, TEXT_TRAIN_DIR, TEXT_FINETUNE_DIR, HF_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# Force-pin the HF cache to <project>/data/hf/ regardless of any inherited
# HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE from the user's shell. Using
# direct assignment (not setdefault) so downloads never leak into
# ~/.cache/huggingface/ even if the user has those vars exported globally.
# These vars propagate to multiprocessing workers via env inheritance, so
# every subprocess (Phase 2 HF streamer) lands in the same cache.
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

import sys
import gc
import bz2
import re
import time
import json
import random
import shutil
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_FILE = os.path.join(TEXT_TRAIN_DIR, "raw_text.txt")
DUMP_FILE = os.path.join(TEXT_TRAIN_DIR, "hywiki-latest-pages-articles.xml.bz2")
WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/hywiki/latest/"
    "hywiki-latest-pages-articles.xml.bz2"
)

# I/O buffer size — large buffers = fewer syscalls = faster merging
IO_BUFFER = 16 * 1024 * 1024  # 16 MB


def clear_hf_cache():
    """Delete the HF cache under data/hf/ to free disk space."""
    if os.path.exists(HF_CACHE_DIR):
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(HF_CACHE_DIR)
            for f in fnames
        ) / (1024 ** 3)
        print(f"  Clearing HF cache at {HF_CACHE_DIR} ({size:.1f} GB)...")
        shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
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


def _marker_path(name):
    return os.path.join(TEXT_TRAIN_DIR, f".{name}_done")


def _marker_exists(name):
    return os.path.exists(_marker_path(name))


def _write_marker(name):
    with open(_marker_path(name), "w") as f:
        f.write("done")


def _append_to_raw(src_path):
    """Append contents of src_path (+ blank-line separator) to raw_text.txt.

    Opens raw_text.txt in append-binary mode (never truncates) and streams
    src into it using 16 MB buffered reads. fsyncs on close so a crash
    immediately after won't lose the committed bytes.
    """
    with open(RAW_FILE, "ab", buffering=IO_BUFFER) as fout:
        with open(src_path, "rb", buffering=IO_BUFFER) as fin:
            while True:
                chunk = fin.read(IO_BUFFER)
                if not chunk:
                    break
                fout.write(chunk)
        fout.write(b"\n\n")
        fout.flush()
        os.fsync(fout.fileno())


def _commit_source(name, src_path):
    """Append src → raw_text.txt, delete src, write marker.

    The marker is written AFTER the append completes, so a crash during
    append leaves no marker and the source is re-downloaded on restart.
    """
    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    print(f"  [{name}] Appending {size_mb:.0f} MB to raw_text.txt...")
    _append_to_raw(src_path)
    os.remove(src_path)
    _write_marker(name)
    print(f"  [{name}] Committed ({size_mb:.0f} MB), intermediate deleted.")


# =============================================================================
#                           CORPUS DOWNLOAD (default)
# =============================================================================

# -----------------------------------------------------------------------------
# Wikipedia
# -----------------------------------------------------------------------------

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
    """Yield cleaned article bodies from the bz2 Wikipedia XML dump.

    Uses xml.etree.iterparse to stream the dump page-by-page. Only pages in
    the main namespace (<ns>0</ns>) are emitted — skipping MediaWiki:,
    Template:, Help:, Category:, User:, File:, etc. Redirects and very
    short pages are filtered too.

    The previous line-by-line regex version was broken two ways: (1) it
    didn't look at <ns> at all, so ~90% of what it extracted was UI
    translation strings, templates, and help pages, and (2) single-line
    <text>...</text> articles were dropped because the end tag on the same
    line as the open tag was never inspected.
    """
    article_count = 0
    skipped = {"ns": 0, "redirect": 0, "too_short": 0, "empty": 0}

    # Wikipedia dumps declare xmlns="http://www.mediawiki.org/xml/export-0.11/"
    # on the root, so iterparse produces namespaced tags like
    # "{http://www.mediawiki.org/xml/export-0.11/}page". Strip the prefix
    # dynamically off the first element we see so we don't hardcode a version.
    ns_prefix = None

    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        root = None

        for event, elem in context:
            if ns_prefix is None and "}" in elem.tag:
                ns_prefix = elem.tag.split("}", 1)[0] + "}"

            if event == "start" and root is None:
                root = elem
                continue

            if event != "end":
                continue

            tag = elem.tag[len(ns_prefix):] if ns_prefix else elem.tag
            if tag != "page":
                continue

            # Completed a <page> — inspect its ns and revision text.
            ns_elem = elem.find(f"{ns_prefix}ns")
            text_elem = elem.find(f"{ns_prefix}revision/{ns_prefix}text")

            page_ns = None
            if ns_elem is not None and ns_elem.text is not None:
                try:
                    page_ns = int(ns_elem.text)
                except ValueError:
                    pass
            text = text_elem.text if text_elem is not None else None

            # Free memory: detach this page from the root so the DOM doesn't
            # accumulate 325k empty <page> shells.
            elem.clear()
            if root is not None:
                root.remove(elem)

            if page_ns != 0:
                skipped["ns"] += 1
                continue
            if not text:
                skipped["empty"] += 1
                continue
            if text.startswith("#REDIRECT") or text.startswith("#ՎԵՐԱՀՂՈՒՄ"):
                skipped["redirect"] += 1
                continue
            if len(text) < 200:
                skipped["too_short"] += 1
                continue

            cleaned = _strip_wiki_markup(text)
            if len(cleaned) > 100:
                article_count += 1
                if article_count % 10000 == 0:
                    print(f"    Extracted {article_count} articles...")
                yield cleaned
            else:
                skipped["too_short"] += 1

    print(f"  Total articles extracted: {article_count}")
    print(f"  Skipped: ns={skipped['ns']:,}  "
          f"redirect={skipped['redirect']:,}  "
          f"too_short={skipped['too_short']:,}  "
          f"empty={skipped['empty']:,}")


def download_wikipedia():
    """Download Armenian Wikipedia dump and extract articles to wiki_hy.txt.

    Returns the path to the extracted intermediate file. Does NOT write a
    marker — the orchestrator writes the marker only after _commit_source
    appends the file into raw_text.txt.
    """
    out_file = os.path.join(TEXT_TRAIN_DIR, "wiki_hy.txt")
    if os.path.exists(out_file):
        os.remove(out_file)

    _download_wiki_dump()

    print("  Extracting articles...")
    chars = 0
    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as out:
        for article_text in _extract_wiki_articles(DUMP_FILE):
            out.write(article_text)
            out.write("\n\n")
            chars += len(article_text)

    print(f"  Wikipedia: {chars:,} chars ({chars / 1024 / 1024:.0f} MB)")
    return out_file


# -----------------------------------------------------------------------------
# CC-100 (direct download, not HuggingFace)
# -----------------------------------------------------------------------------

def download_cc100():
    """Download and decompress CC-100 Armenian data to cc100_hy.txt.

    Returns the path to the decompressed intermediate file. Does NOT write
    a marker — the orchestrator writes the marker only after _commit_source
    appends the file into raw_text.txt.
    """
    import lzma

    out_file = os.path.join(TEXT_TRAIN_DIR, "cc100_hy.txt")
    cc100_xz = os.path.join(TEXT_TRAIN_DIR, "cc100_hy.txt.xz")

    if os.path.exists(out_file):
        os.remove(out_file)

    cc100_url = "https://data.statmt.org/cc-100/hy.txt.xz"

    if not os.path.exists(cc100_xz):
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

    print("  Decompressing CC-100 (this takes a few minutes)...")
    chars = 0
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

    if os.path.exists(cc100_xz):
        os.remove(cc100_xz)

    return out_file


# -----------------------------------------------------------------------------
# HPLT 3.0 (direct HTTP + zstandard shards — not on HF as a loadable dataset)
# -----------------------------------------------------------------------------

HPLT3_MAP_URL = "https://data.hplt-project.org/three/sorted/hye_Armn.map"


def download_hplt3():
    """Download HPLT 3.0 Armenian shards directly from data.hplt-project.org.

    HPLT 3.0 is distributed as a ``.map`` manifest listing one or more
    zstd-compressed JSONL shard URLs. Each JSONL line is one document with
    a ``text`` field (plus metadata we ignore). Shards are organised by
    quality-score buckets (5 = lowest, 10 = highest quality).

    We stream each shard through the zstandard decompressor line-by-line
    and write the ``text`` field into hplt3_hy.txt. Shard files are NOT
    written to disk — memory-only streaming, bounded by the decompressor
    buffer and one line at a time.
    """
    import json
    import zstandard

    out_file = os.path.join(TEXT_TRAIN_DIR, "hplt3_hy.txt")
    if os.path.exists(out_file):
        os.remove(out_file)

    print("  Fetching HPLT 3.0 Armenian shard manifest...")
    with urllib.request.urlopen(HPLT3_MAP_URL, timeout=30) as resp:
        manifest = resp.read().decode("utf-8").strip().splitlines()
    shard_urls = [line.strip() for line in manifest if line.strip()]
    print(f"  Manifest: {len(shard_urls)} shards")

    chars = 0
    docs = 0
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor()

    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as fout:
        for i, shard_url in enumerate(shard_urls, 1):
            shard_name = shard_url.rsplit("/", 1)[-1]
            print(f"  [{i}/{len(shard_urls)}] {shard_name} ...", flush=True)

            req = urllib.request.Request(shard_url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Stream-decompress the HTTPS body through zstandard.
                # stream_reader wraps the raw response, yielding decompressed
                # bytes; we wrap again with TextIOWrapper for line iteration.
                with dctx.stream_reader(resp) as reader:
                    import io as _io
                    text_stream = _io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = row.get("text", "") or ""
                        text = text.strip()
                        if len(text) < 50:
                            continue
                        fout.write(text)
                        fout.write("\n\n")
                        chars += len(text)
                        docs += 1
                        if docs % 50_000 == 0:
                            elapsed = time.time() - t0
                            rate = chars / elapsed / 1_000_000 if elapsed > 0 else 0
                            print(
                                f"    {docs:,} docs, {chars / 1_000_000:.0f}M chars "
                                f"({rate:.1f} MB/s)",
                                flush=True,
                            )

    elapsed = time.time() - t0
    print(f"  HPLT 3.0: {docs:,} docs, {chars:,} chars "
          f"({chars / 1024 / 1024:.0f} MB) in {fmt_time(elapsed)}")
    return out_file


# -----------------------------------------------------------------------------
# ARLIS Armenian legislation database (direct HTTP, JSONL.xz with HTML body)
# -----------------------------------------------------------------------------

ARLIS_URL = "https://opendataam.sfo3.cdn.digitaloceanspaces.com/arlis/arlis_docs.jsonl.xz"


def _strip_arlis_html(html):
    """Clean ARLIS legal-document body HTML to plain Armenian text.

    ARLIS bodies arrive as HTML with numeric character references like
    ``&#1329;`` for Armenian letters. We unescape entities, drop tags,
    collapse whitespace. Keeps structure at paragraph level.
    """
    import html as _html
    text = _html.unescape(html)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_arlis():
    """Download the ARLIS Armenian legal corpus and extract clean text.

    Produces arlis_hy.txt containing one document per entry, each entry
    being ``title\\n\\nbody_cleaned``.
    """
    import lzma
    import json

    out_file = os.path.join(TEXT_TRAIN_DIR, "arlis_hy.txt")
    arlis_xz = os.path.join(TEXT_TRAIN_DIR, "arlis_docs.jsonl.xz")

    if os.path.exists(out_file):
        os.remove(out_file)

    if not os.path.exists(arlis_xz):
        print("  Downloading ARLIS legal corpus (~508 MB)...")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  {pct:.1f}% ({mb:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(ARLIS_URL, arlis_xz, reporthook=progress)
        print("\n  Download complete!")

    print("  Extracting ARLIS documents (HTML → plain text)...")
    docs = 0
    chars = 0
    t0 = time.time()

    with lzma.open(arlis_xz, "rt", encoding="utf-8") as f_in, \
         open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (row.get("title") or "").strip()
            body_html = row.get("body") or ""
            body = _strip_arlis_html(body_html)
            if len(body) < 100:
                continue
            if title:
                f_out.write(title)
                f_out.write("\n\n")
                chars += len(title) + 2
            f_out.write(body)
            f_out.write("\n\n")
            chars += len(body) + 2
            docs += 1
            if docs % 5000 == 0:
                elapsed = time.time() - t0
                rate = chars / elapsed / 1_000_000 if elapsed > 0 else 0
                print(f"    {docs:,} docs, {chars / 1_000_000:.0f}M chars "
                      f"({rate:.1f} MB/s)", flush=True)

    print(f"  ARLIS: {docs:,} docs, {chars:,} chars "
          f"({chars / 1024 / 1024:.0f} MB)")

    if os.path.exists(arlis_xz):
        os.remove(arlis_xz)

    return out_file


# -----------------------------------------------------------------------------
# OpenSubtitles 2024 (HF, parallel corpus with src/tgt lang columns)
# -----------------------------------------------------------------------------

def download_opensubtitles():
    """Extract Armenian sides of Helsinki-NLP/OpenSubtitles2024.

    The dataset is a parallel bitext: each row has ``src_text``, ``tgt_text``,
    ``src_lang``, ``tgt_lang``. Armenian appears in the ``validation`` and
    ``test`` splits only (no ``train`` split). We iterate both and pull
    ``src_text`` where ``src_lang == "hy"`` and ``tgt_text`` where
    ``tgt_lang == "hy"``, de-duplicating identical lines so we don't count
    the same Armenian utterance twice when it appears with multiple
    target languages.

    Requires HF gate acceptance:
    https://huggingface.co/datasets/Helsinki-NLP/OpenSubtitles2024
    """
    from datasets import load_dataset
    from huggingface_hub import get_token

    out_file = os.path.join(TEXT_TRAIN_DIR, "opensubtitles_hy.txt")
    if os.path.exists(out_file):
        os.remove(out_file)

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    seen = set()  # dedupe identical Armenian sentences across bitext pairs
    chars = 0
    docs = 0
    t0 = time.time()

    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as fout:
        for split in ("validation", "test"):
            print(f"  Streaming OpenSubtitles2024 [{split}] split...", flush=True)
            try:
                ds = load_dataset(
                    "Helsinki-NLP/OpenSubtitles2024", "default",
                    split=split, streaming=True, token=hf_token,
                )
            except Exception as e:
                print(f"  [opensubtitles:{split}] load failed: {e}")
                continue

            for row in ds:
                texts = []
                if row.get("src_lang") == "hy":
                    t = (row.get("src_text") or "").strip()
                    if t:
                        texts.append(t)
                if row.get("tgt_lang") == "hy":
                    t = (row.get("tgt_text") or "").strip()
                    if t:
                        texts.append(t)
                del row
                for t in texts:
                    if len(t) < 5:
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    fout.write(t)
                    fout.write("\n")
                    chars += len(t) + 1
                    docs += 1
                    if docs % 50_000 == 0:
                        elapsed = time.time() - t0
                        rate = chars / elapsed / 1_000_000 if elapsed > 0 else 0
                        print(
                            f"    {docs:,} unique hy lines, {chars / 1_000_000:.1f}M chars "
                            f"({rate:.2f} MB/s)",
                            flush=True,
                        )
            gc.collect()

    elapsed = time.time() - t0
    print(f"  OpenSubtitles: {docs:,} unique lines, {chars:,} chars "
          f"({chars / 1024 / 1024:.1f} MB) in {fmt_time(elapsed)}")
    return out_file


# -----------------------------------------------------------------------------
# HuggingFace streaming download (runs as subprocess for parallelism)
# -----------------------------------------------------------------------------

def _download_hf_worker(args):
    """
    Worker function for parallel HF downloads.
    Runs in a separate process. Streams one dataset and writes to its
    own intermediate text file. The parent process (orchestrator) is
    responsible for appending the file into raw_text.txt and writing
    the marker after the worker returns.

    args format (tuple):
        (name, dataset_id, lang_config, text_field, out_file, hf_token,
         filter_field, filter_value)

    ``filter_field`` / ``filter_value`` are optional (may be ``None``).
    When set, rows where ``example[filter_field] != filter_value`` are
    dropped — used to pull Armenian-only rows from CC-News, which doesn't
    expose language as a config.
    """
    (name, dataset_id, lang_config, text_field, out_file, hf_token,
     filter_field, filter_value) = args

    if os.path.exists(out_file):
        os.remove(out_file)

    import gc as _gc
    from datasets import load_dataset

    print(f"  [{name}] Starting download ({dataset_id})...", flush=True)
    t0 = time.time()

    ds = load_dataset(
        dataset_id, lang_config,
        split="train", streaming=True, token=hf_token,
    )

    chars = 0
    docs = 0
    skipped = 0
    write_buf = []
    buf_size = 0
    FLUSH_THRESHOLD = 5 * 1024 * 1024  # 5 MB

    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as f:
        for example in ds:
            if filter_field is not None and example.get(filter_field) != filter_value:
                skipped += 1
                del example
                continue
            text = example.get(text_field, "")
            del example  # drop reference to Arrow row; helps streaming GC
            if text is None:
                continue
            text = text.strip()
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
                _gc.collect()  # reclaim streaming iterator caches
                elapsed = time.time() - t0
                rate = chars / elapsed / 1_000_000 if elapsed > 0 else 0
                print(
                    f"  [{name}] {docs:,} docs, {chars / 1_000_000:.0f}M chars "
                    f"({rate:.1f} MB/s)",
                    flush=True,
                )

        if write_buf:
            f.write("".join(write_buf))
            write_buf.clear()

    elapsed = time.time() - t0
    print(
        f"  [{name}] Done: {docs:,} docs, {chars / 1024 / 1024:.0f} MB "
        f"in {fmt_time(elapsed)}",
        flush=True,
    )

    return name, out_file, docs, chars


def download_corpus(args):
    """Download pretraining sources and stream-append each into raw_text.txt.

    Per-source lifecycle:
        1. If marker `.{name}_done` exists → skip entirely.
        2. Otherwise: download source → append to raw_text.txt → delete
           intermediate → write marker.

    Peak disk footprint = raw_text.txt + (intermediates of sources currently
    in-flight). The wiki `.bz2` and cc100 `.xz` are deleted as soon as
    their extracted text is committed.

    Source inventory:
        Phase 1 direct downloads:
            wiki     — Armenian Wikipedia (dump → extract)
            cc100    — CC-100 Armenian (xz → decompress)
            hplt3    — HPLT 3.0 Armenian (zstd shards via .map manifest)
            arlis    — ARLIS legal corpus (jsonl.xz, HTML body → text)
        Phase 2 HuggingFace streaming (parallel):
            culturax — uonlp/CulturaX (hy)
            mc4      — allenai/c4 (hy)
            glot500  — cis-lmu/Glot500 (hye_Armn)
            ccnews_2023, ccnews_2024 — stanford-oval/ccnews with lang filter
            opensubtitles — Helsinki-NLP/OpenSubtitles2024 (gated, optional)

    HPLT 2.0 has been replaced by HPLT 3.0 — same pipeline, expanded time
    window (2012-2024), better extractor. OSCAR-2301 not included —
    gated-with-manual-review and access was not granted.
    """
    skip = set(s.lower() for s in args.skip)

    # Each HF source entry: dict with repo, config, text_field,
    # optional filter_field / filter_value for row-level filtering.
    hf_sources = {
        "culturax": {
            "repo": "uonlp/CulturaX", "config": "hy", "text_field": "text",
        },
        "mc4": {
            "repo": "allenai/c4", "config": "hy", "text_field": "text",
        },
        "glot500": {
            "repo": "cis-lmu/Glot500", "config": "hye_Armn", "text_field": "text",
        },
        "ccnews_2023": {
            "repo": "stanford-oval/ccnews", "config": "2023",
            "text_field": "plain_text",
            "filter_field": "language", "filter_value": "hy",
        },
        "ccnews_2024": {
            "repo": "stanford-oval/ccnews", "config": "2024",
            "text_field": "plain_text",
            "filter_field": "language", "filter_value": "hy",
        },
    }

    # Resolve HF token once in the parent; pass explicitly to workers so
    # multiprocessing spawn workers auth reliably (don't rely on env var
    # or ~/.cache lookups inside fresh subprocess interpreters).
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            hf_token = None

    print(f"{'='*60}")
    print(f"  ArmGPT Corpus Download — Stream & Append Mode")
    print(f"{'='*60}")
    print(f"  Output:     {RAW_FILE}")
    print(f"  HF cache:   {HF_CACHE_DIR}")
    print(f"  HF workers: {args.workers}")
    print(f"  HF auth:    {'token present' if hf_token else 'NONE (unauthenticated)'}")
    if skip:
        print(f"  Skipping:   {', '.join(skip)}")
    print(f"{'='*60}\n")

    # Ensure raw_text.txt exists so the first "ab" open doesn't error.
    if not os.path.exists(RAW_FILE):
        open(RAW_FILE, "wb").close()

    t_start = time.time()

    # ---- Phase 1: Wikipedia + CC-100 (sequential direct downloads) ----
    print("=" * 40)
    print("Phase 1: Direct downloads")
    print("=" * 40)

    if "wiki" in skip:
        print("[SKIP] Wikipedia")
    elif _marker_exists("wiki"):
        print("\n[WIKI] Already committed to raw_text.txt, skipping.")
    else:
        print("\n[WIKI] Armenian Wikipedia (~1.5 GB)")
        wiki_file = download_wikipedia()
        _commit_source("wiki", wiki_file)
        # Drop the wiki dump .bz2 right after commit — we never need it again
        # for this run, and cleanup preserves markers so a rerun also skips.
        if os.path.exists(DUMP_FILE):
            dump_mb = os.path.getsize(DUMP_FILE) / (1024 * 1024)
            os.remove(DUMP_FILE)
            print(f"  Removed wiki dump ({dump_mb:.0f} MB)")

    if "cc100" in skip:
        print("[SKIP] CC-100")
    elif _marker_exists("cc100"):
        print("\n[CC100] Already committed to raw_text.txt, skipping.")
    else:
        print("\n[CC100] CC-100 Armenian (~4.9 GB)")
        cc100_file = download_cc100()
        _commit_source("cc100", cc100_file)

    if "hplt3" in skip:
        print("[SKIP] HPLT 3.0")
    elif _marker_exists("hplt3"):
        print("\n[HPLT3] Already committed to raw_text.txt, skipping.")
    else:
        print("\n[HPLT3] HPLT 3.0 Armenian (~8.3 GB compressed, ~25 GB text)")
        hplt3_file = download_hplt3()
        _commit_source("hplt3", hplt3_file)

    if "arlis" in skip:
        print("[SKIP] ARLIS")
    elif _marker_exists("arlis"):
        print("\n[ARLIS] Already committed to raw_text.txt, skipping.")
    else:
        print("\n[ARLIS] Armenian legislation database (~508 MB compressed)")
        arlis_file = download_arlis()
        _commit_source("arlis", arlis_file)

    if "opensubtitles" in skip:
        print("[SKIP] OpenSubtitles")
    elif _marker_exists("opensubtitles"):
        print("\n[OPENSUBTITLES] Already committed to raw_text.txt, skipping.")
    else:
        print("\n[OPENSUBTITLES] Helsinki-NLP/OpenSubtitles2024 (validation + test, hy filter)")
        try:
            os_file = download_opensubtitles()
            if os.path.exists(os_file) and os.path.getsize(os_file) > 0:
                _commit_source("opensubtitles", os_file)
            else:
                print("  OpenSubtitles: empty output, nothing to commit.")
                if os.path.exists(os_file):
                    os.remove(os_file)
        except Exception as e:
            print(f"  OpenSubtitles: ERROR {e}")
            partial = os.path.join(TEXT_TRAIN_DIR, "opensubtitles_hy.txt")
            if os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass

    # ---- Phase 2: HuggingFace sources (parallel download, sequential commit) ----
    hf_to_download = {}
    for name, cfg in hf_sources.items():
        if name in skip:
            print(f"[SKIP] {name}")
        elif _marker_exists(name):
            print(f"[{name.upper()}] Already committed to raw_text.txt, skipping.")
        else:
            hf_to_download[name] = cfg

    if hf_to_download:
        print(f"\n{'='*40}")
        print(f"Phase 2: HuggingFace downloads ({len(hf_to_download)} sources in parallel)")
        print(f"{'='*40}")

        worker_args = []
        for name, cfg in hf_to_download.items():
            out_file = os.path.join(TEXT_TRAIN_DIR, f"{name}_hy.txt")
            worker_args.append((
                name,
                cfg["repo"],
                cfg["config"],
                cfg["text_field"],
                out_file,
                hf_token,
                cfg.get("filter_field"),
                cfg.get("filter_value"),
            ))

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
                    _name, out_file, docs, chars = future.result()
                    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                        print(
                            f"  [{name}] Streamed: {docs:,} docs, "
                            f"{chars / 1024 / 1024:.0f} MB"
                        )
                        _commit_source(name, out_file)
                    else:
                        print(f"  [{name}] Empty output, nothing to commit.")
                except Exception as e:
                    print(f"  [{name}] ERROR: {e}")
                    # Clean up any partial intermediate left by the crashed worker
                    partial = os.path.join(TEXT_TRAIN_DIR, f"{name}_hy.txt")
                    if os.path.exists(partial):
                        try:
                            os.remove(partial)
                        except OSError:
                            pass
                    print(f"  [{name}] Skipped — will be retried on next run.")

    # ---- Phase 3: Final cleanup ----
    print(f"\n{'='*40}")
    print(f"Phase 3: Cleanup")
    print(f"{'='*40}")

    clear_hf_cache()

    # Only sweep stray .bz2 / .xz intermediates. Markers and raw_text.txt
    # are preserved — markers are what lets a subsequent run skip work.
    for f in os.listdir(TEXT_TRAIN_DIR):
        path = os.path.join(TEXT_TRAIN_DIR, f)
        if not os.path.isfile(path):
            continue
        if f.endswith((".bz2", ".xz")):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Removing stray {f} ({size_mb:.0f} MB)")
            os.remove(path)

    elapsed = time.time() - t_start
    final_size = get_file_size_mb(RAW_FILE)
    print(f"\n{'='*60}")
    print(f"  Corpus Download Complete!")
    print(f"{'='*60}")
    print(f"  raw_text.txt: {final_size:.0f} MB ({final_size / 1024:.1f} GB)")
    print(f"  Total time:   {fmt_time(elapsed)}")
    print(f"\n  Next step: python 2_prepare.py")


# =============================================================================
#                         ArmBench Q&A (absorbed, --qa)
# =============================================================================

_ARMBENCH_REPO = "Metric-AI/ArmBench-LLM-data"

ARMBENCH_TRAIN_CONFIGS = [
    "exam_history",
    "exam_literature",
    "exam_math",
    "include-mcqa",
    "public-services-mcqa",
]

ARMBENCH_EVAL_CONFIGS = [
    "simpleqa",
    "squad-in-context-qa",
    "belebele-in-context-mcqa",
]

_ARMENIAN_LETTERS = ["Ա", "Բ", "Գ", "Դ", "Ե", "Զ", "Է", "Ը", "Թ", "Ժ"]


def _first_split(repo_id, config):
    """Load whatever split the config actually provides."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, config)
    split = list(ds.keys())[0]
    return ds[split]


def _normalize_single_label(label, n_choices):
    """Normalize a SINGLE-answer label to a 0-indexed int."""
    if isinstance(label, list):
        if len(label) != 1:
            raise ValueError(f"multi-label, not a single-answer task: {label!r}")
        label = label[0]
    if isinstance(label, str):
        s = label.strip()
        if s.isdigit():
            idx = int(s) - 1
            if not (0 <= idx < n_choices):
                raise ValueError(f"label {s!r} out of range for {n_choices} choices")
            return idx
        if len(s) == 1 and s.upper() in "ABCDEFGH":
            return ord(s.upper()) - ord("A")
    if isinstance(label, int):
        if 0 <= label < n_choices:
            return label
        if 1 <= label <= n_choices:
            return label - 1
    raise ValueError(f"Unrecognized label {label!r}")


def _format_mcq(question, choices, correct_idx, context=None):
    """Format an MCQ as a single instruction string with lettered choices."""
    if len(choices) > len(_ARMENIAN_LETTERS):
        raise ValueError(f"Too many choices ({len(choices)}) for MCQ format")
    body = question.strip()
    if context:
        body = f"Տեքստ:\n{context.strip()}\n\nՀարց: {body}"
    lines = [body, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{_ARMENIAN_LETTERS[i]}) {str(choice).strip()}")
    instruction = "\n".join(lines)
    answer_text = str(choices[correct_idx]).strip()
    output = f"{_ARMENIAN_LETTERS[correct_idx]}) {answer_text}"
    return instruction, output


def _process_exam_config(cfg):
    """exam_history / exam_literature / exam_math — single-answer task types only."""
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    skipped_by_tt = {}
    for row in ds:
        tt = row.get("task_type")
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        task = (row.get("task") or "").strip()
        choices = row.get("choices") or []
        label = row.get("label")
        if not question or label is None:
            continue

        full_question = f"{task} {question}".strip() if task else question

        if tt == 7:
            # Open-ended: label IS the answer string.
            answer = label[0] if isinstance(label, list) and label else label
            answer = str(answer).strip()
            if not answer:
                continue
            instruction = (f"Տեքստ:\n{context}\n\nՀարց: {full_question}"
                           if context else full_question)
            out.append({
                "instruction": instruction,
                "input": "",
                "output": answer,
                "source": f"armbench/{cfg}/open",
            })
            continue

        if tt in (1, 6):
            if not choices:
                continue
            try:
                correct_idx = _normalize_single_label(label, len(choices))
            except Exception:
                skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1
                continue
            try:
                instruction, output = _format_mcq(
                    full_question, choices, correct_idx,
                    context=context or None,
                )
            except ValueError:
                skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1
                continue
            out.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": f"armbench/{cfg}/mcq",
            })
            continue

        skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1

    if skipped_by_tt:
        print(f"  [{cfg}] skipped by task_type: {skipped_by_tt}")
    return out


def _process_include_mcqa(cfg="include-mcqa"):
    """include-mcqa: question + option_a..option_d + answer (1-indexed int)."""
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        choices = [row.get(f"option_{k}") or "" for k in ("a", "b", "c", "d")]
        choices = [c for c in choices if c]
        answer = row.get("answer")
        if not question or len(choices) != 4 or answer is None:
            continue
        try:
            correct_idx = _normalize_single_label(answer, 4)
        except Exception:
            continue
        if not (0 <= correct_idx < 4):
            continue
        instruction, output = _format_mcq(question, choices, correct_idx)
        out.append({"instruction": instruction, "input": "", "output": output,
                    "source": f"armbench/{cfg}"})
    return out


def _process_public_services(cfg="public-services-mcqa"):
    """public-services-mcqa: question + answer (text!) + distractors.

    This config is special: the answer is free-form, not a letter. Emit
    BOTH an open-ended version (highest-quality SFT signal) and an MCQ
    version (for format diversity).
    """
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        distractors = row.get("distractors") or []
        if not question or not answer:
            continue
        out.append({
            "instruction": question,
            "input": "",
            "output": answer,
            "source": f"armbench/{cfg}/open",
        })
        if distractors:
            rng = random.Random(hash(question) & 0xFFFFFFFF)  # deterministic per row
            choices = [answer] + list(distractors)
            idxs = list(range(len(choices)))
            rng.shuffle(idxs)
            shuffled = [choices[i] for i in idxs]
            correct_idx = idxs.index(0)
            instruction, output = _format_mcq(question, shuffled, correct_idx)
            out.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": f"armbench/{cfg}/mcq",
            })
    return out


def _process_simpleqa(cfg="simpleqa"):
    """simpleqa: question + answer, open-ended. Best eval format."""
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        out.append({"instruction": question, "input": "", "output": answer,
                    "source": f"armbench/{cfg}"})
    return out


def _process_squad_in_context(cfg="squad-in-context-qa"):
    """squad-in-context-qa: context + question + answer (extractive)."""
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    for row in ds:
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answer = row.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = (answer or "").strip()
        if not question or not answer:
            continue
        instruction = f"Տեքստ:\n{context}\n\nՀարց: {question}" if context else question
        out.append({"instruction": instruction, "input": "", "output": answer,
                    "source": f"armbench/{cfg}"})
    return out


def _process_belebele(cfg="belebele-in-context-mcqa"):
    """belebele: flores_passage + question + 4 mc_answers + correct_answer_num."""
    ds = _first_split(_ARMBENCH_REPO, cfg)
    out = []
    for row in ds:
        passage = (row.get("flores_passage") or "").strip()
        question = (row.get("question") or "").strip()
        choices = [row.get(f"mc_answer{i}") or "" for i in (1, 2, 3, 4)]
        choices = [c for c in choices if c]
        correct = row.get("correct_answer_num")
        if not question or len(choices) != 4 or correct is None:
            continue
        try:
            correct_idx = _normalize_single_label(correct, 4)
        except Exception:
            continue
        instruction, output = _format_mcq(question, choices, correct_idx,
                                          context=passage)
        out.append({"instruction": instruction, "input": "", "output": output,
                    "source": f"armbench/{cfg}"})
    return out


_ARMBENCH_PROCESSORS = {
    "exam_history": lambda: _process_exam_config("exam_history"),
    "exam_literature": lambda: _process_exam_config("exam_literature"),
    "exam_math": lambda: _process_exam_config("exam_math"),
    "include-mcqa": _process_include_mcqa,
    "public-services-mcqa": _process_public_services,
    "simpleqa": _process_simpleqa,
    "squad-in-context-qa": _process_squad_in_context,
    "belebele-in-context-mcqa": _process_belebele,
}


def fetch_armbench_qa(train_output_path, eval_output_path):
    """Fetch ArmBench native Armenian Q&A and write train/eval JSON files."""
    print("=" * 60)
    print("  ArmBench → Q&A normalizer")
    print("=" * 60)

    train_pairs = []
    for cfg in ARMBENCH_TRAIN_CONFIGS:
        print(f"\n[train] Processing {cfg}...")
        pairs = _ARMBENCH_PROCESSORS[cfg]()
        print(f"  {cfg}: {len(pairs):,} examples")
        train_pairs.extend(pairs)

    eval_pairs = []
    for cfg in ARMBENCH_EVAL_CONFIGS:
        print(f"\n[eval]  Processing {cfg}...")
        pairs = _ARMBENCH_PROCESSORS[cfg]()
        print(f"  {cfg}: {len(pairs):,} examples")
        eval_pairs.extend(pairs)

    print(f"\n{'=' * 60}")
    print(f"  Training:   {len(train_pairs):,} examples")
    print(f"  Eval:       {len(eval_pairs):,} examples")
    print(f"{'=' * 60}")

    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved train → {train_output_path}")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved eval  → {eval_output_path}")

    return len(train_pairs), len(eval_pairs)


# =============================================================================
#                         Aya Q&A (absorbed, --qa)
# =============================================================================

_AYA_REPO = "CohereLabs/aya_collection_language_split"

# Per-source sampling plan. None = take everything.
# Dropped after quality spot-check: Mintaka-inst (nonsense task format),
# NQ-Open (MT mangles short factoid answers), WIKI QA (inverted direction).
_AYA_SOURCE_PLAN = {
    "Arpa-instruct":        None,    # native Armenian, take all (~4K)
    "Dolly-v2 (T)":         None,    # ~14K, all
    "Dolly-v2":             None,    # future-proof if they drop the (T)
    "HotpotQA (T)":         10000,   # aggressive filter here, yields ~21%
    "Adversarial QA (T)":   5000,
    "Flan-CoT-submix (T)":  5000,
    "Flan-unified-QA (T)":  None,    # ~540, take all
}

# Armenian Unicode ranges: main U+0530–U+058F, ligatures U+FB13–U+FB17
_ARTIFACT_PATTERNS = [
    "<unk>", "[unk]", "[UNK]", "<|", "|>", "{{", "}}", "[[", "]]",
]

_WHITESPACE_RE = re.compile(r"\s+")


def _armenian_letter_ratio(s):
    """Fraction of letters in `s` that are Armenian script. 0.0 if no letters."""
    if not s:
        return 0.0
    arm = 0
    alpha = 0
    for c in s:
        if c.isalpha():
            alpha += 1
            if ("\u0530" <= c <= "\u058F") or ("\uFB13" <= c <= "\uFB17"):
                arm += 1
    if alpha == 0:
        return 0.0
    return arm / alpha


def _ws_clean(s):
    if s is None:
        return ""
    return _WHITESPACE_RE.sub(" ", s).strip()


def _ws_normalize_key(s):
    return _WHITESPACE_RE.sub(" ", s or "").strip().lower()


def _aya_to_pair(row, *, min_q_len, max_q_len, min_a_len, max_a_len, min_arm_ratio):
    """Convert one Aya row to an SFT pair or None if it fails quality checks."""
    q = _ws_clean(row.get("inputs"))
    a = _ws_clean(row.get("targets"))
    if not q or not a:
        return None

    if not (min_q_len <= len(q) <= max_q_len):
        return None
    if not (min_a_len <= len(a) <= max_a_len):
        return None

    for pat in _ARTIFACT_PATTERNS:
        if pat in q or pat in a:
            return None

    if _armenian_letter_ratio(a) < min_arm_ratio:
        return None
    if _armenian_letter_ratio(q) < (min_arm_ratio - 0.2):
        return None

    if _ws_normalize_key(q) == _ws_normalize_key(a):
        return None

    # Suspicious stub: very long instruction but trivial answer.
    if len(q) > 300 and len(a) < 15:
        return None

    return {
        "instruction": q,
        "input": "",
        "output": a,
        "source": f"aya/{row.get('dataset_name', '?')}",
    }


def _aya_process_source(ds, name, n_samples, rng, filters):
    """Extract and clean one Aya source."""
    print(f"\nFiltering {name}...")
    sub = ds.filter(lambda x: x["dataset_name"] == name, num_proc=4)
    total = len(sub)
    if total == 0:
        print(f"  {name}: not present in split, skipping")
        return []
    print(f"  {name}: {total:,} rows available")

    if n_samples is not None and n_samples < total:
        idxs = rng.sample(range(total), n_samples)
        candidates = (sub[i] for i in idxs)
        pool_size = n_samples
    else:
        candidates = (sub[i] for i in range(total))
        pool_size = total

    kept = []
    dropped = 0
    for row in candidates:
        pair = _aya_to_pair(row, **filters)
        if pair is None:
            dropped += 1
            continue
        kept.append(pair)
    yield_rate = 100.0 * len(kept) / max(pool_size, 1)
    print(f"  {name}: kept {len(kept):,} / {pool_size:,} "
          f"({yield_rate:.1f}% yield, {dropped:,} dropped)")
    return kept


def fetch_aya_qa(
    output_path,
    seed=42,
    min_q_len=10,
    max_q_len=2000,
    min_a_len=20,
    max_a_len=2000,
    min_arm_ratio=0.75,
    plan=None,
):
    """Fetch and quality-filter the Armenian Aya slice, write to JSON.

    Writes ONLY the Aya-derived pairs (no merging). Cross-source dedup is
    handled later by 2_prepare.py --qa.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    plan = plan or dict(_AYA_SOURCE_PLAN)

    filters = dict(
        min_q_len=min_q_len,
        max_q_len=max_q_len,
        min_a_len=min_a_len,
        max_a_len=max_a_len,
        min_arm_ratio=min_arm_ratio,
    )

    print("=" * 60)
    print("  Aya Armenian filtered fetcher")
    print("=" * 60)
    print(f"  Armenian ratio floor: {min_arm_ratio}")
    print(f"  Q length:             {min_q_len}-{max_q_len}")
    print(f"  A length:             {min_a_len}-{max_a_len}")
    print(f"  Sources in plan:      {len(plan)}")
    print(f"  Output:               {output_path}")

    print(f"\nLoading {_AYA_REPO} (armenian/train)...")
    ds = load_dataset(_AYA_REPO, "armenian", split="train")
    print(f"  Total Armenian rows: {len(ds):,}")

    all_pairs = []
    per_source_counts = {}
    for name, n_samples in plan.items():
        kept = _aya_process_source(ds, name, n_samples, rng, filters)
        all_pairs.extend(kept)
        per_source_counts[name] = len(kept)

    # Intra-source dedup only; cross-source dedup is 2_prepare.py's job.
    seen = set()
    unique = []
    for p in all_pairs:
        key = _ws_normalize_key(p["instruction"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    print(f"\n{'=' * 60}")
    print(f"  Aya filtering complete")
    print(f"{'=' * 60}")
    for name, count in per_source_counts.items():
        if count:
            print(f"  {name}: {count:,}")
    print(f"  Total kept: {len(unique):,} "
          f"(dropped {len(all_pairs) - len(unique):,} intra-source dupes)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {output_path}")

    return len(unique)


# =============================================================================
#                             Q&A download (--qa)
# =============================================================================

def download_qa(args):
    """Fetch SFT Q&A sources (ArmBench + Aya) into data/text/finetune/."""
    skip = set(s.lower() for s in args.skip)

    print(f"{'='*60}")
    print(f"  ArmGPT Q&A Download (SFT sources)")
    print(f"{'='*60}")
    print(f"  Output:    {TEXT_FINETUNE_DIR}")
    print(f"  HF cache:  {HF_CACHE_DIR}")
    if skip:
        print(f"  Skipping:  {', '.join(skip)}")
    print(f"{'='*60}\n")

    t_start = time.time()

    if "armbench" not in skip:
        print("[ARMBENCH] Native Armenian exam + civics QA")
        fetch_armbench_qa(
            train_output_path=os.path.join(TEXT_FINETUNE_DIR, "armbench_train.json"),
            eval_output_path=os.path.join(TEXT_FINETUNE_DIR, "armbench_eval.json"),
        )
    else:
        print("[SKIP] ArmBench")

    if "aya" not in skip:
        print("\n[AYA] Filtered Armenian slice of Aya collection")
        fetch_aya_qa(
            output_path=os.path.join(TEXT_FINETUNE_DIR, "aya_armenian.json"),
        )
    else:
        print("[SKIP] Aya")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Q&A Download Complete!  ({fmt_time(elapsed)})")
    print(f"{'='*60}")
    print(f"  Optional: also run core/generate_armenian_qa.py to add")
    print(f"            Claude-generated pairs under data/text/finetune/")
    print(f"\n  Next step: python 2_prepare.py --qa")


# =============================================================================
#                                    Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download Armenian text data (corpus by default, or --qa for SFT sources)"
    )
    parser.add_argument("--qa", action="store_true",
                        help="Download SFT Q&A sources (ArmBench + Aya) instead of raw corpus")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Sources to skip. Corpus: wiki cc100 hplt3 arlis opensubtitles "
                             "culturax mc4 glot500 ccnews_2023 ccnews_2024. "
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
