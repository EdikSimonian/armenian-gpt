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
    corpus mode: data/text/train/{wiki,wikisource,wiktionary,wikiquote,
                                  cc100,hplt3,arlis,ccnews,
                                  culturax,mc4,glot500,finetranslations}_hy.txt
                 (one plain-text file per source; kept SEPARATE so 2_prepare.py
                  can tokenize them with provenance tags. Each source also has
                  a `.{name}_done` marker file alongside so reruns skip.)
    --qa mode:   data/text/finetune/armbench_train.json
                 data/text/finetune/armbench_eval.json
                 data/text/finetune/aya_armenian.json
                 data/text/finetune/armenian_qa_qwen.json   (if generated)
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

# Force pyarrow to use the system memory allocator instead of jemalloc.
# On macOS jemalloc aggressively retains freed memory (RSS stays elevated
# for minutes/hours after the actual allocations are gone), which inflated
# our CC-News worker RSS to multiple GB on long scans. The system
# allocator returns memory to the OS more eagerly. Must be set BEFORE
# pyarrow is imported.
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

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
    """True if a source is considered already-downloaded.

    Source files are kept separate on disk (one ``{name}_hy.txt`` per source)
    so the marker additionally verifies the file exists. If somebody
    deletes the intermediate manually, the marker is considered stale and
    the source gets re-downloaded on the next run.
    """
    marker = _marker_path(name)
    if not os.path.exists(marker):
        return False
    # Phase 1 and Phase 2 sources both write an intermediate named
    # `{name}_hy.txt`. Defensive: if it's missing, treat the marker as stale.
    expected = os.path.join(TEXT_TRAIN_DIR, f"{name}_hy.txt")
    if not os.path.exists(expected):
        return False
    return True


def _write_marker(name):
    with open(_marker_path(name), "w") as f:
        f.write("done")


def _commit_source(name, src_path):
    """Mark a source as complete. The intermediate file is KEPT on disk.

    Per the "keep files separate" design, we do NOT merge sources into a
    single `raw_text.txt`. Each source lives as its own `{name}_hy.txt`
    intermediate and the downstream tokenizer (2_prepare.py) iterates
    them at read-time, tagging each record with its source. This
    preserves provenance and makes per-source updates cheap (rerun one
    source without re-downloading the others).

    The marker is written AFTER the file is confirmed on disk so a crash
    during download leaves no marker and the source is retried.
    """
    if not os.path.exists(src_path):
        raise RuntimeError(f"[{name}] source file missing: {src_path}")
    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    _write_marker(name)
    print(f"  [{name}] Done ({size_mb:.0f} MB) -> kept as "
          f"{os.path.basename(src_path)}")


# =============================================================================
#                           CORPUS DOWNLOAD (default)
# =============================================================================

# -----------------------------------------------------------------------------
# Wikipedia
# -----------------------------------------------------------------------------

def _download_wikimedia_dump(url, dest_path, label, est_size_mb):
    """Generic Wikimedia XML dump downloader.

    Used by Wikipedia and all sister projects (Wikisource/Wiktionary/
    Wikiquote). The dump file is stored at ``dest_path``; if it already
    exists we skip the download so a rerun after a Phase 4 cleanup that
    kept the dump around (never happens in our pipeline, but harmless)
    doesn't re-fetch.
    """
    if os.path.exists(dest_path):
        print(f"  Dump already downloaded: {dest_path}")
        return

    print(f"  Downloading {label} dump (~{est_size_mb} MB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()


def _download_wiki_dump():
    """Download the Armenian Wikipedia dump."""
    _download_wikimedia_dump(WIKI_DUMP_URL, DUMP_FILE,
                             "Armenian Wikipedia", 500)


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

    Returns the path to the extracted file. Does NOT write a marker —
    the orchestrator calls _commit_source afterwards which writes the
    marker only after verifying the file is on disk.
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
# Wikimedia sister projects (Wikisource, Wiktionary, Wikiquote)
# -----------------------------------------------------------------------------
# Same MediaWiki XML dump format as Wikipedia, same ns=0 filtering, same
# markup-strip pass. The only thing that varies is the project prefix on
# the dumps URL and the local filenames. Each project gets its own
# intermediate + marker so it can be skipped/retried independently.

def _download_wikimedia_project(project, label, est_size_mb, out_filename):
    """Generic Wikimedia sister-project downloader.

    `project` is a dump-URL identifier like "hywikisource" or "hywiktionary".
    `out_filename` is the name of the intermediate text file under
    TEXT_TRAIN_DIR (e.g. "wikisource_hy.txt"). Returns the output path.
    Leaves the `.xml.bz2` dump next to the intermediate so Phase 4 cleanup
    sweeps it away along with the wiki dump and any .xz scraps.
    """
    dump_name = f"{project}-latest-pages-articles.xml.bz2"
    dump_url = f"https://dumps.wikimedia.org/{project}/latest/{dump_name}"
    dump_path = os.path.join(TEXT_TRAIN_DIR, dump_name)
    out_file = os.path.join(TEXT_TRAIN_DIR, out_filename)

    if os.path.exists(out_file):
        os.remove(out_file)

    _download_wikimedia_dump(dump_url, dump_path, label, est_size_mb)

    print("  Extracting articles...")
    chars = 0
    with open(out_file, "w", encoding="utf-8", buffering=IO_BUFFER) as out:
        for article_text in _extract_wiki_articles(dump_path):
            out.write(article_text)
            out.write("\n\n")
            chars += len(article_text)

    print(f"  {label}: {chars:,} chars ({chars / 1024 / 1024:.0f} MB)")

    # Delete the dump as soon as the extract is done — we don't need it
    # again, and Phase 4's .bz2 sweep would catch it anyway but earlier
    # cleanup keeps peak disk lower.
    if os.path.exists(dump_path):
        try:
            os.remove(dump_path)
        except OSError:
            pass

    return out_file


def download_wikisource():
    """Armenian Wikisource: classical literature, chronicles, grabar texts."""
    return _download_wikimedia_project(
        "hywikisource", "Armenian Wikisource", 100, "wikisource_hy.txt",
    )


def download_wiktionary():
    """Armenian Wiktionary: dictionary definitions and usage examples."""
    return _download_wikimedia_project(
        "hywiktionary", "Armenian Wiktionary", 40, "wiktionary_hy.txt",
    )


def download_wikiquote():
    """Armenian Wikiquote: literary quotations and proverbs (tiny, unique domain)."""
    return _download_wikimedia_project(
        "hywikiquote", "Armenian Wikiquote", 3, "wikiquote_hy.txt",
    )


# -----------------------------------------------------------------------------
# CC-100 (direct download, not HuggingFace)
# -----------------------------------------------------------------------------

def download_cc100():
    """Download and decompress CC-100 Armenian data to cc100_hy.txt.

    Returns the path to the decompressed file. Does NOT write a marker —
    the orchestrator writes the marker via _commit_source after the file
    is confirmed on disk.
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

    print("  Extracting ARLIS documents (HTML -> plain text)...")
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
# CC-News (stanford-oval/ccnews): direct parquet reads with pyarrow filtering
# -----------------------------------------------------------------------------
# We bypass the HuggingFace `datasets` streaming iterator here because it has
# a known issue where Arrow RecordBatches are retained between yields (RSS
# growing into multiple GB on long filter-heavy runs) and because per-row
# Python-side filtering is ~100× slower than pyarrow's batched column filter.
#
# This path downloads each parquet shard to a temp dir, scans it with
# pyarrow's column-batch iterator, keeps only rows where language == "hy",
# writes the plain_text field out, and deletes the shard before moving on.
# Peak disk footprint: one parquet shard (~150-200 MB) + the growing
# ccnews_hy.txt output file. Peak RAM per worker: ~50-100 MB (a pyarrow
# batch is small and buffers are released cleanly).

CCNEWS_YEARS = ("2023", "2024")


# Scan-worker source code as a string. Embedded in the main module to
# avoid a separate file / import / pickle dance. Run via
# `python -c CCNEWS_SCAN_WORKER_SRC parquet_path out_path io_buffer`.
# Writes one stats line to stdout: "OK docs chars scanned".
CCNEWS_SCAN_WORKER_SRC = r"""
import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc

parquet_path, out_path, io_buffer = sys.argv[1], sys.argv[2], int(sys.argv[3])
docs = 0
chars = 0
scanned = 0
try:
    pf = pq.ParquetFile(parquet_path, pre_buffer=False, memory_map=False)
    with open(out_path, "ab", buffering=io_buffer) as fout:
        for batch in pf.iter_batches(
            batch_size=10000,
            columns=["plain_text", "language"],
        ):
            scanned += len(batch)
            mask = pc.equal(batch.column("language"), "hy")
            n_hy = pc.sum(mask).as_py() or 0
            if n_hy == 0:
                continue
            filtered = batch.filter(mask)
            for text in filtered.column("plain_text").to_pylist():
                if not text:
                    continue
                t = text.strip()
                if len(t) < 50:
                    continue
                b = t.encode("utf-8")
                fout.write(b)
                fout.write(b"\n\n")
                chars += len(b) + 2
                docs += 1
except Exception as e:
    print(f"ERR {e}", file=sys.stderr)
    print(f"OK {docs} {chars} {scanned}", flush=True)
    sys.exit(1)
print(f"OK {docs} {chars} {scanned}", flush=True)
"""


def download_ccnews(max_parallel_downloads=8):
    """Download CC-News 2023+2024 shards and extract Armenian rows.

    Architecture (memory-safe across 110+ shards):
      - Parent process manages a ThreadPoolExecutor to download parquet
        shards from HuggingFace in parallel (network-bound, GIL releases
        during I/O).
      - As each shard completes downloading, the parent spawns a FRESH
        subprocess (`python -c <inlined scan script>`) to scan it.
        The subprocess imports pyarrow, reads the shard, filters by
        language==hy, appends hy rows to ccnews_hy.txt, and exits.
      - When the subprocess exits, its entire pyarrow/jemalloc memory
        footprint is reclaimed by the OS. Parent process RSS stays
        roughly constant at its baseline (~50-100 MB) regardless of
        how many shards we process.
      - Parquet shard files are deleted from the temp dir as soon as
        their scan subprocess returns, so peak disk is bounded by
        `max_parallel_downloads` shards in flight (~150 MB each).
    """
    import tempfile
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, get_token

    out_file = os.path.join(TEXT_TRAIN_DIR, "ccnews_hy.txt")
    progress_file = os.path.join(TEXT_TRAIN_DIR, ".ccnews_progress")

    # Resume-aware startup:
    #   - If both ccnews_hy.txt AND .ccnews_progress exist, we're resuming
    #     a previous run: load the set of already-processed shard names
    #     and skip them this pass, appending new rows to the same output.
    #   - If only one exists (or neither), that's an inconsistent/fresh
    #     state → start from scratch.
    done_shards = set()
    if os.path.exists(out_file) and os.path.exists(progress_file):
        with open(progress_file) as f:
            done_shards = {line.strip() for line in f if line.strip()}
        print(f"  CC-News resume: {len(done_shards)} shards already processed, "
              f"existing output {os.path.getsize(out_file) / 1024 / 1024:.1f} MB",
              flush=True)
    else:
        if os.path.exists(out_file):
            os.remove(out_file)
        if os.path.exists(progress_file):
            os.remove(progress_file)
        # Create an empty file so scan subprocesses can open it in "ab" mode.
        open(out_file, "wb").close()
        open(progress_file, "w").close()

    token = os.environ.get("HF_TOKEN") or get_token()

    api = HfApi(token=token)
    all_files = api.list_repo_files(
        "stanford-oval/ccnews", repo_type="dataset", token=token
    )
    all_targets = sorted(
        f for f in all_files
        if f.endswith(".parquet")
        and any(f.startswith(f"{y}_") for y in CCNEWS_YEARS)
    )
    targets = [f for f in all_targets if f not in done_shards]
    if done_shards:
        print(f"  CC-News: {len(targets)} shards remaining "
              f"({len(done_shards)} already done, {len(all_targets)} total)",
              flush=True)
    else:
        print(f"  CC-News: {len(targets)} parquet shards to scan "
              f"({'+'.join(CCNEWS_YEARS)})", flush=True)
    if not targets:
        return out_file

    total_docs = 0
    total_chars = 0
    total_scanned = 0
    t_start = time.time()

    def _direct_fetch(fname, tmpdir):
        """Direct HTTPS download via urllib — no HF library involvement.

        hf_hub_download retains ~300-400 MB of parent-process state per
        file (hash tracking, ref metadata, blob symlinks, etc.) even when
        called with local_dir. urllib streams bytes straight from HTTPS
        to disk and leaves no residual state, which is critical when
        processing 110+ shards back-to-back.
        """
        import urllib.request
        import urllib.error
        url = (
            f"https://huggingface.co/datasets/stanford-oval/ccnews/"
            f"resolve/main/{fname}"
        )
        out_path = os.path.join(tmpdir, fname)
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=120) as resp, \
             open(out_path, "wb") as f:
            while True:
                chunk = resp.read(4 * 1024 * 1024)  # 4 MB pipe chunks
                if not chunk:
                    break
                f.write(chunk)
        return out_path

    with tempfile.TemporaryDirectory(dir=TEXT_TRAIN_DIR, prefix="ccnews_pq_") as tmpdir, \
         ThreadPoolExecutor(max_workers=max_parallel_downloads) as pool:

        futures = {pool.submit(_direct_fetch, f, tmpdir): f for f in targets}

        for idx, future in enumerate(as_completed(futures), 1):
            fname = futures[future]
            try:
                path = future.result()
            except Exception as e:
                print(f"  [{idx}/{len(targets)}] {fname}: download failed: {e}",
                      flush=True)
                continue

            # Run the scan in a FRESH subprocess (python -c <inlined src>).
            # When it exits, the OS reclaims all pyarrow memory pool state,
            # which is the only reliable way to prevent multi-GB RSS
            # retention across 110+ shards on macOS.
            file_docs = 0
            file_chars = 0
            file_scanned = 0
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-c", CCNEWS_SCAN_WORKER_SRC,
                        path, out_file, str(IO_BUFFER),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                stdout = proc.stdout.strip()
                last = stdout.rsplit("\n", 1)[-1] if stdout else ""
                if last.startswith("OK "):
                    _, d, c, s = last.split()
                    file_docs = int(d)
                    file_chars = int(c)
                    file_scanned = int(s)
                if proc.returncode != 0:
                    err = proc.stderr.strip() or "unknown"
                    print(
                        f"  [{idx}/{len(targets)}] {fname}: scan subprocess "
                        f"exit={proc.returncode}: {err[:200]}",
                        flush=True,
                    )
            except subprocess.TimeoutExpired:
                print(
                    f"  [{idx}/{len(targets)}] {fname}: scan subprocess "
                    f"timeout (>900s)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"  [{idx}/{len(targets)}] {fname}: subprocess error: {e}",
                    flush=True,
                )

            # Reclaim disk space — delete the shard immediately after scan.
            try:
                os.remove(path)
            except OSError:
                pass

            # Record the shard as done so a future run can resume.
            try:
                with open(progress_file, "a") as pf_:
                    pf_.write(fname + "\n")
                    pf_.flush()
                    os.fsync(pf_.fileno())
            except OSError:
                pass

            total_docs += file_docs
            total_chars += file_chars
            total_scanned += file_scanned
            elapsed = time.time() - t_start
            rate_mb = total_chars / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(
                f"  [{idx}/{len(targets)}] {fname}: +{file_docs} hy docs "
                f"({file_chars / 1024:.0f} KB) | total {total_docs:,} docs, "
                f"{total_chars / 1024 / 1024:.1f} MB, "
                f"scanned {total_scanned / 1e6:.1f} M rows ({rate_mb:.2f} MB/s)",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"  CC-News: {total_docs:,} hy docs, {total_chars:,} chars "
          f"({total_chars / 1024 / 1024:.0f} MB) in {fmt_time(elapsed)}")
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
    own text file ``{name}_hy.txt`` that the parent keeps on disk. The
    parent writes the marker via _commit_source after the worker returns.

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
    """Download pretraining sources. Each source is kept as its own file.

    Per-source lifecycle:
        1. If marker `.{name}_done` exists AND `{name}_hy.txt` is on disk
           → skip entirely.
        2. Otherwise: download source → write `{name}_hy.txt` → write marker.

    No merging into a single `raw_text.txt`. Every source stays as its own
    plain-text file so the tokenizer step (2_prepare.py) can iterate them
    with provenance tags, and so you can regenerate or replace individual
    sources cheaply without rebuilding the whole corpus.

    Source inventory:
        Phase 1 direct downloads (sequential):
            wiki, wikisource, wiktionary, wikiquote — Wikimedia dumps
            cc100  — CC-100 Armenian (xz → decompress)
            hplt3  — HPLT 3.0 Armenian (zstd shards via .map manifest)
            arlis  — ARLIS legal corpus (jsonl.xz, HTML body → text)
            ccnews — Stanford CC-News, direct-parquet hy filter
            opensubtitles — Helsinki-NLP/OpenSubtitles2024 (gated, optional)
        Phase 2 HuggingFace streaming (parallel):
            culturax        — uonlp/CulturaX (hy)
            mc4             — allenai/c4 (hy)
            glot500         — cis-lmu/Glot500 (hye_Armn)
            finetranslations — HuggingFaceFW/finetranslations (hye_Armn)

    HPLT 2.0 has been replaced by HPLT 3.0 — same pipeline, expanded time
    window (2012-2024), better extractor. OSCAR-2301 not included —
    gated-with-manual-review and access was not granted.
    """
    skip = set(s.lower() for s in args.skip)

    # Each HF source entry: dict with repo, config, text_field,
    # optional filter_field / filter_value for row-level filtering.
    # Note: CC-News is NOT here — it uses the dedicated download_ccnews()
    # direct-parquet path to avoid the datasets library's leaky streaming
    # iterator on large filter-heavy scans.
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
        # FineTranslations is a parallel Armenian↔English corpus where the
        # Armenian side (`og_full_text`, og = "original") is the native
        # CommonCrawl text filtered by the FineWeb2 pipeline. We pull only
        # the Armenian column; the English `translated_text` field is
        # discarded. ~943 M tokens of Trafilatura-cleaned Armenian web text.
        "finetranslations": {
            "repo": "HuggingFaceFW/finetranslations",
            "config": "hye_Armn",
            "text_field": "og_full_text",
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
    print(f"  ArmGPT Corpus Download — Per-Source Files Mode")
    print(f"{'='*60}")
    print(f"  Output dir: {TEXT_TRAIN_DIR}")
    print(f"  HF cache:   {HF_CACHE_DIR}")
    print(f"  HF workers: {args.workers}")
    print(f"  HF auth:    {'token present' if hf_token else 'NONE (unauthenticated)'}")
    if skip:
        print(f"  Skipping:   {', '.join(skip)}")
    print(f"{'='*60}\n")

    t_start = time.time()

    # ---- Phase 1: Wikipedia + CC-100 (sequential direct downloads) ----
    print("=" * 40)
    print("Phase 1: Direct downloads")
    print("=" * 40)

    if "wiki" in skip:
        print("[SKIP] Wikipedia")
    elif _marker_exists("wiki"):
        print("\n[WIKI] Already downloaded (marker + file on disk), skipping.")
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

    if "wikisource" in skip:
        print("[SKIP] Wikisource")
    elif _marker_exists("wikisource"):
        print("\n[WIKISOURCE] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[WIKISOURCE] Armenian Wikisource (classical literature, ~100 MB dump)")
        wikisource_file = download_wikisource()
        _commit_source("wikisource", wikisource_file)

    if "wiktionary" in skip:
        print("[SKIP] Wiktionary")
    elif _marker_exists("wiktionary"):
        print("\n[WIKTIONARY] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[WIKTIONARY] Armenian Wiktionary (dictionary, ~40 MB dump)")
        wiktionary_file = download_wiktionary()
        _commit_source("wiktionary", wiktionary_file)

    if "wikiquote" in skip:
        print("[SKIP] Wikiquote")
    elif _marker_exists("wikiquote"):
        print("\n[WIKIQUOTE] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[WIKIQUOTE] Armenian Wikiquote (quotations, ~3 MB dump)")
        wikiquote_file = download_wikiquote()
        _commit_source("wikiquote", wikiquote_file)

    if "cc100" in skip:
        print("[SKIP] CC-100")
    elif _marker_exists("cc100"):
        print("\n[CC100] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[CC100] CC-100 Armenian (~4.9 GB)")
        cc100_file = download_cc100()
        _commit_source("cc100", cc100_file)

    if "hplt3" in skip:
        print("[SKIP] HPLT 3.0")
    elif _marker_exists("hplt3"):
        print("\n[HPLT3] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[HPLT3] HPLT 3.0 Armenian (~8.3 GB compressed, ~25 GB text)")
        hplt3_file = download_hplt3()
        _commit_source("hplt3", hplt3_file)

    if "arlis" in skip:
        print("[SKIP] ARLIS")
    elif _marker_exists("arlis"):
        print("\n[ARLIS] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[ARLIS] Armenian legislation database (~508 MB compressed)")
        arlis_file = download_arlis()
        _commit_source("arlis", arlis_file)

    if "ccnews" in skip:
        print("[SKIP] CC-News")
    elif _marker_exists("ccnews"):
        print("\n[CCNEWS] Already downloaded (marker + file on disk), skipping.")
    else:
        print("\n[CCNEWS] Stanford CC-News 2023+2024 (direct parquet + pyarrow hy filter)")
        try:
            ccn_file = download_ccnews(max_parallel_downloads=8)
            if os.path.exists(ccn_file) and os.path.getsize(ccn_file) > 0:
                _commit_source("ccnews", ccn_file)
                # Clean up the shard-level resume sidecar now that the
                # full file is committed.
                progress_file = os.path.join(TEXT_TRAIN_DIR, ".ccnews_progress")
                if os.path.exists(progress_file):
                    os.remove(progress_file)
            else:
                print("  CC-News: empty output, nothing to commit.")
                if os.path.exists(ccn_file):
                    os.remove(ccn_file)
        except Exception as e:
            print(f"  CC-News: ERROR {e}")
            # Leave ccnews_hy.txt and .ccnews_progress in place — they
            # represent resume state for the next run.

    if "opensubtitles" in skip:
        print("[SKIP] OpenSubtitles")
    elif _marker_exists("opensubtitles"):
        print("\n[OPENSUBTITLES] Already downloaded (marker + file on disk), skipping.")
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
            print(f"[{name.upper()}] Already downloaded (marker + file on disk), skipping.")
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

    # Only sweep stray .bz2 / .xz dumps — the per-source {name}_hy.txt
    # intermediates are KEPT on disk (they're the final output now,
    # not temporary files). Markers are also kept so reruns skip cleanly.
    for f in os.listdir(TEXT_TRAIN_DIR):
        path = os.path.join(TEXT_TRAIN_DIR, f)
        if not os.path.isfile(path):
            continue
        if f.endswith((".bz2", ".xz")):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Removing stray {f} ({size_mb:.0f} MB)")
            os.remove(path)

    elapsed = time.time() - t_start

    # Inventory the per-source files left on disk and report totals.
    source_files = sorted(
        f for f in os.listdir(TEXT_TRAIN_DIR)
        if f.endswith("_hy.txt")
        and os.path.isfile(os.path.join(TEXT_TRAIN_DIR, f))
    )
    total_bytes = sum(
        os.path.getsize(os.path.join(TEXT_TRAIN_DIR, f)) for f in source_files
    )
    total_gb = total_bytes / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"  Corpus Download Complete!")
    print(f"{'='*60}")
    print(f"  Sources on disk: {len(source_files)} files in {TEXT_TRAIN_DIR}")
    for f in source_files:
        size_mb = os.path.getsize(os.path.join(TEXT_TRAIN_DIR, f)) / (1024 * 1024)
        print(f"    {f:32s}  {size_mb:>8.0f} MB")
    print(f"  Total corpus:    {total_bytes:,} bytes ({total_gb:.2f} GB)")
    print(f"  Total time:      {fmt_time(elapsed)}")
    print(f"\n  Next step: python 2_prepare.py  "
          f"(tokenizer reads all {len(source_files)} source files at prep time)")


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
    print("  ArmBench -> Q&A normalizer")
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
    print(f"  Saved train -> {train_output_path}")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved eval  -> {eval_output_path}")

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
    print(f"\n  Saved -> {output_path}")

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
#                       HuggingFace publish / fetch
# =============================================================================
# Two-way door for the prepared corpus + Q&A bundle. Default repo is
# `edisimon/armenian-clean-text`, overridable with --hf-repo.
#
# `--upload` packages the deduped clean_text.txt (from 2_prepare.py) as a
# zstd-compressed file under `corpus/clean_text.txt.zst`, plus the raw
# Q&A JSONs under `finetune/`, plus a generated README.md with full source
# attribution and license notes. Before uploading it deletes any files in
# the repo that aren't part of the new structure so the repo stays clean.
#
# `--download` pulls everything back, decompresses the corpus to
# `data/text/train/clean_text.txt`, places the Q&A files under
# `data/text/finetune/`, and writes a sentinel marker so `1_download.py`
# reruns skip the full re-download path.

DEFAULT_HF_DATASET_REPO = "edisimon/armenian-clean-text"

HF_CORPUS_PATH = "corpus/clean_text.txt.zst"
HF_FINETUNE_DIR = "finetune"
HF_TOKENIZED_DIR = "tokenized"

# Source inventory used by _build_hf_readme. Keeps the README and the
# actual source list in sync — add an entry here whenever a new source
# is wired into download_corpus().
_HF_README_SOURCES = [
    ("Armenian Wikipedia (hywiki)",              "CC BY-SA 4.0",
     "https://dumps.wikimedia.org/hywiki/"),
    ("Armenian Wikisource (hywikisource)",       "CC BY-SA 4.0",
     "https://dumps.wikimedia.org/hywikisource/"),
    ("Armenian Wiktionary (hywiktionary)",       "CC BY-SA 4.0",
     "https://dumps.wikimedia.org/hywiktionary/"),
    ("Armenian Wikiquote (hywikiquote)",         "CC BY-SA 4.0",
     "https://dumps.wikimedia.org/hywikiquote/"),
    ("CC-100 Armenian",                           "Common Crawl Terms of Use",
     "https://data.statmt.org/cc-100/hy.txt.xz"),
    ("HPLT 3.0 Armenian (hye_Armn)",              "CC0 1.0",
     "https://data.hplt-project.org/three/sorted/hye_Armn.map"),
    ("CulturaX Armenian (uonlp/CulturaX)",       "ODC-By 1.0",
     "https://huggingface.co/datasets/uonlp/CulturaX"),
    ("mC4 Armenian (allenai/c4)",                 "ODC-By 1.0",
     "https://huggingface.co/datasets/allenai/c4"),
    ("Glot500 Armenian (cis-lmu/Glot500)",       "Research use (mixed)",
     "https://huggingface.co/datasets/cis-lmu/Glot500"),
    ("ARLIS Armenian legislation database",       "Public domain (HY gov)",
     "https://data.opendata.am/dataset/arlis-db"),
    ("Stanford CC-News (hy filter)",             "Common Crawl Terms of Use",
     "https://huggingface.co/datasets/stanford-oval/ccnews"),
    ("FineTranslations Armenian (hye_Armn)",     "ODC-By 1.0",
     "https://huggingface.co/datasets/HuggingFaceFW/finetranslations"),
]


def _build_hf_readme(corpus_stats=None, qa_files=None):
    """Render the README.md contents for the HF dataset repo.

    Includes YAML front matter, source attribution, license notes, and
    quick-start loading examples.
    """
    attribution = "\n".join(
        f"- **{name}** — {lic}  \n  {url}"
        for (name, lic, url) in _HF_README_SOURCES
    )

    corpus_block = ""
    if corpus_stats:
        corpus_block = (
            "## Corpus statistics\n\n"
            f"- Uncompressed size: **{corpus_stats.get('uncompressed_gb', '?'):.2f} GB**\n"
            f"- Compressed size (zstd L19): **{corpus_stats.get('compressed_gb', '?'):.2f} GB**\n"
            f"- Paragraphs: **{corpus_stats.get('paragraphs', 0):,}**\n"
            f"- Sources: **12** (see list above)\n\n"
            "Deduplicated at the paragraph level across all sources. "
            "Quality-priority dedup order: Wikimedia → ARLIS → news → HPLT 3.0 → "
            "CulturaX → FineTranslations → mC4 → CC-100.\n\n"
        )

    qa_block = ""
    if qa_files:
        qa_block = "## Fine-tuning Q&A files\n\n"
        for fname, n_pairs in qa_files:
            qa_block += f"- `finetune/{fname}` — {n_pairs:,} pairs\n"
        qa_block += "\n"

    return f"""---
license: other
license_name: mixed-source-attribution
language:
- hy
pretty_name: Armenian Clean Corpus
tags:
- armenian
- hy
- hye_Armn
- pretraining
- text-corpus
- sft
size_categories:
- 10B<n<100B
---

# Armenian Clean Corpus (pretraining + SFT bundle)

Combined, deduplicated, cleaned Armenian text assembled for pretraining
and supervised fine-tuning of small language models. Built via the
pipeline at <https://github.com/EdikSimonian/armenian-gpt>:

```
python 1_download.py        # fetch sources
python 2_prepare.py         # clean + dedup + merge
python 1_download.py --upload   # push this bundle
```

## Contents

```
corpus/clean_text.txt.zst    zstd-compressed merged corpus
finetune/*.json              SFT Q&A files (ArmBench, Aya hy slice,
                             optional Qwen/Claude generated pairs)
```

{corpus_block}{qa_block}## Sources used

{attribution}

## License and attribution

This dataset is assembled from multiple publicly available sources for
text-and-data-mining research on low-resource language modeling.
Wikimedia sources (Wikipedia / Wikisource / Wiktionary / Wikiquote) are
CC BY-SA 4.0 — strict application of the ShareAlike clause would require
this derivative work to also be CC BY-SA 4.0. The bundle is published
under a mixed-source-attribution notice invoking the EU TDM exception
(Directive 2019/790 Art. 4) and US fair use for ML research.

Users are responsible for compliance with source license terms in their
own jurisdiction. If you redistribute or fine-tune models on this data,
please also preserve attribution to the upstream sources listed above.

## Quick start

Download the compressed corpus:

```python
from huggingface_hub import hf_hub_download
import zstandard as zstd

path = hf_hub_download(
    repo_id="{DEFAULT_HF_DATASET_REPO}",
    filename="{HF_CORPUS_PATH}",
    repo_type="dataset",
)

# Decompress
with open(path, "rb") as fin, open("clean_text.txt", "wb") as fout:
    zstd.ZstdDecompressor().copy_stream(fin, fout)
```

Or reproduce via the pipeline:

```bash
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install -r requirements.txt
python 1_download.py --download   # uses this HF dataset as source
```
"""


def _compress_zstd(src_path, dst_path, level=12):
    """Compress a file with zstd at the given level. Streams; bounded memory.

    Default level is 12. Rationale:
      * L3  (default):  ~500 MB/s, ~2.8× ratio
      * L12 (sweet spot): ~60-100 MB/s per-core, ~3.4× ratio  ← used here
      * L19 (ultra):    ~10 MB/s per-core, ~3.8× ratio
    L12 gives ~90% of L19's compression at ~6× the throughput, which
    matters a lot when the corpus is 60+ GB. With threads=-1 on a
    10-core Mac this lands a 63 GB corpus in roughly 2-4 minutes.
    """
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=level, threads=-1)
    total_in = os.path.getsize(src_path)
    t0 = time.time()
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        cctx.copy_stream(fin, fout, read_size=16 * 1024 * 1024,
                         write_size=16 * 1024 * 1024)
    total_out = os.path.getsize(dst_path)
    elapsed = time.time() - t0
    ratio = total_in / total_out if total_out else 0
    print(f"  zstd L{level}: {total_in / 1024 ** 3:.2f} GB -> "
          f"{total_out / 1024 ** 3:.2f} GB "
          f"({ratio:.1f}x ratio) in {fmt_time(elapsed)}")
    return total_in, total_out


def _decompress_zstd(src_path, dst_path):
    """Decompress a zstd file to a plain output file (streaming).

    Uses 64 MB I/O buffers for maximum throughput on NVMe/SSD storage.
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    t0 = time.time()
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        dctx.copy_stream(fin, fout, read_size=64 * 1024 * 1024,
                         write_size=64 * 1024 * 1024)
    total_out = os.path.getsize(dst_path)
    elapsed = time.time() - t0
    print(f"  unzstd: {os.path.getsize(src_path) / 1024 ** 3:.2f} GB -> "
          f"{total_out / 1024 ** 3:.2f} GB in {fmt_time(elapsed)}")


def upload_dataset_to_hf(repo_id, corpus_path, finetune_dir, token=None,
                         tokenized=False, data_dir=None):
    """Upload the clean corpus + Q&A bundle to a HF dataset repo.

    Layout written to the repo (existing unrelated files are deleted):
        README.md
        corpus/clean_text.txt.zst
        finetune/*.json
        tokenized/train_bpe.bin.zst   (if --tokenized)
        tokenized/val_bpe.bin.zst     (if --tokenized)
        tokenized/tokenizer_bpe.json  (if --tokenized)
        tokenized/bpe_model.model     (if --tokenized)

    Uses ``HfApi.upload_folder`` with ``delete_patterns=["*"]`` which tells
    HF to remove any existing files not part of this upload. After upload
    we run an LFS orphan sweep via ``permanently_delete_lfs_files`` to
    reclaim storage quota from overwritten blobs.
    """
    import tempfile

    from huggingface_hub import HfApi, get_token

    if not os.path.exists(corpus_path):
        print(f"Error: corpus file not found: {corpus_path}")
        print("Run 'python 2_prepare.py' first to produce clean_text.txt.")
        sys.exit(1)

    if not os.path.isdir(finetune_dir):
        print(f"Error: finetune dir not found: {finetune_dir}")
        sys.exit(1)

    token = token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("Error: no HF_TOKEN found — run 'hf auth login' first.")
        sys.exit(1)

    api = HfApi(token=token)

    # Enumerate Q&A files that will be uploaded (JSONs only, skip sidecars).
    qa_files = []
    for f in sorted(os.listdir(finetune_dir)):
        if not f.endswith(".json"):
            continue
        if f.endswith("_progress.json"):
            continue  # skip generator-side resume state
        qa_files.append(f)

    if not qa_files:
        print(f"Warning: no .json Q&A files under {finetune_dir}")

    # Count pairs in each QA file for README stats.
    qa_summary = []
    for f in qa_files:
        p = os.path.join(finetune_dir, f)
        try:
            with open(p) as fh:
                data = json.load(fh)
            n = len(data) if isinstance(data, list) else 0
        except Exception:
            n = 0
        qa_summary.append((f, n))

    # Stage everything in a tempdir so the upload is a single atomic folder.
    with tempfile.TemporaryDirectory(prefix="armgpt-hf-upload-") as staging:
        stage_corpus = os.path.join(staging, "corpus")
        stage_finetune = os.path.join(staging, "finetune")
        os.makedirs(stage_corpus, exist_ok=True)
        os.makedirs(stage_finetune, exist_ok=True)

        # Compress corpus
        print(f"\n{'='*60}")
        print(f"  Step 1/3: Compressing corpus")
        print(f"{'='*60}")
        out_zst = os.path.join(stage_corpus, "clean_text.txt.zst")
        raw_size, zst_size = _compress_zstd(corpus_path, out_zst, level=12)

        # Copy QA files
        print(f"\n{'='*60}")
        print(f"  Step 2/3: Staging Q&A files ({len(qa_files)})")
        print(f"{'='*60}")
        for f in qa_files:
            shutil.copy2(os.path.join(finetune_dir, f),
                         os.path.join(stage_finetune, f))
            print(f"  {f}")

        # Paragraph count from clean_stats.json (if exists) for the README
        paragraphs = 0
        stats_path = os.path.join(os.path.dirname(corpus_path), "clean_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                paragraphs = stats.get("totals", {}).get("kept_paragraphs", 0)
            except Exception:
                pass

        # Stage tokenized bins (compressed) if requested
        if tokenized and data_dir:
            stage_tokenized = os.path.join(staging, "tokenized")
            os.makedirs(stage_tokenized, exist_ok=True)
            # Large bins: compress with zstd
            for name in ("train_bpe.bin", "val_bpe.bin"):
                src = os.path.join(data_dir, name)
                if not os.path.exists(src):
                    print(f"  Warning: {src} not found, skipping")
                    continue
                dst = os.path.join(stage_tokenized, name + ".zst")
                print(f"\n  Compressing {name}...")
                _compress_zstd(src, dst, level=3)
            # Small files: copy as-is
            for name in ("tokenizer_bpe.json", "bpe_model.model"):
                src = os.path.join(data_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(stage_tokenized, name))
                    print(f"  {name}")

        readme = _build_hf_readme(
            corpus_stats={
                "uncompressed_gb": raw_size / 1024 ** 3,
                "compressed_gb": zst_size / 1024 ** 3,
                "paragraphs": paragraphs,
            },
            qa_files=qa_summary,
        )
        with open(os.path.join(staging, "README.md"), "w") as f:
            f.write(readme)

        # Nuke + recreate the repo. HF's private-repo LFS quota counts the
        # new upload against the limit BEFORE deleting old files, so a
        # simple `delete_patterns=["*"]` sync fails with a 403 once the
        # combined old+new exceeds quota. Delete-then-create is the only
        # reliable way to start from a known-clean slate.
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            print(f"\nDeleting existing repo {repo_id} to free quota...")
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            print(f"  Deleted {repo_id}")
            # Small delay to let HF's quota accounting catch up before recreate
            time.sleep(3)
        except Exception as e:
            # Not found is fine — we'll create fresh
            print(f"  No existing repo to delete: {e}")

        print(f"Creating fresh dataset repo: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        private=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Step 3/3: Uploading to {repo_id}")
        print(f"{'='*60}")
        print(f"  Staged contents: {staging}")
        print()
        t0 = time.time()
        commit_msg = (f"Upload corpus + Q&A bundle "
                      f"(corpus {zst_size / 1024 ** 3:.1f} GB compressed, "
                      f"{len(qa_files)} Q&A files)")
        api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        elapsed = time.time() - t0
        print(f"  Upload complete in {fmt_time(elapsed)}")

    # Reclaim LFS orphan quota after the sync
    print(f"\n{'='*60}")
    print(f"  Post-upload: sweeping LFS orphans")
    print(f"{'='*60}")
    try:
        _lfs_orphan_cleanup(api, repo_id)
    except Exception as e:
        print(f"  LFS orphan cleanup skipped: {e}")

    print(f"\n{'='*60}")
    print(f"  Done -> https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}")


def _lfs_orphan_cleanup(api, repo_id):
    """Permanently delete LFS blobs not referenced in the current tree."""
    lfs_files = list(api.list_lfs_files(repo_id=repo_id, repo_type="dataset"))
    referenced = set(api.list_repo_files(repo_id, repo_type="dataset"))
    orphans = [f for f in lfs_files if f.filename not in referenced]
    if not orphans:
        print("  No orphaned LFS blobs.")
        return
    size_gb = sum(f.size for f in orphans) / 1024 ** 3
    print(f"  Freeing {size_gb:.2f} GB across {len(orphans)} orphan LFS blobs...")
    api.permanently_delete_lfs_files(
        repo_id=repo_id, repo_type="dataset", lfs_files=orphans,
    )
    print(f"  Done.")


def download_dataset_from_hf(repo_id, train_dir, finetune_dir, token=None,
                             tokenized=False, tokenized_only=False,
                             data_dir=None):
    """Fetch the published corpus + Q&A bundle back into data/text/{train,finetune}/.

    This is the counterpart to ``--upload``: reconstructs the local
    layout exactly as ``2_prepare.py`` would leave it, so downstream
    steps (3_tokenize.py, training) run unchanged.

    If ``tokenized=True``, also fetches pre-tokenized BPE bins from
    tokenized/ in the repo and decompresses them into data_dir, so
    you can skip 3_tokenize.py entirely.

    If ``tokenized_only=True``, fetches ONLY the tokenized bins
    (no corpus, no Q&A). Implies tokenized=True.
    """
    if tokenized_only:
        tokenized = True
    from huggingface_hub import HfApi, hf_hub_download, get_token

    token = token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("Error: no HF_TOKEN found — run 'hf auth login' first.")
        sys.exit(1)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)

    clean_out = None
    if not tokenized_only:
        print(f"\n{'='*60}")
        print(f"  Step 1/3: Fetching corpus from {repo_id}")
        print(f"{'='*60}")
        try:
            zst_path = hf_hub_download(
                repo_id=repo_id,
                filename=HF_CORPUS_PATH,
                repo_type="dataset",
                token=token,
                cache_dir=HF_CACHE_DIR,
            )
        except Exception as e:
            print(f"Error fetching {HF_CORPUS_PATH}: {e}")
            print(f"Check that the repo exists and has been populated via --upload.")
            sys.exit(1)

        print(f"  Downloaded: {zst_path}")

        print(f"\n{'='*60}")
        print(f"  Step 2/3: Decompressing -> {train_dir}/clean_text.txt")
        print(f"{'='*60}")
        clean_out = os.path.join(train_dir, "clean_text.txt")
        _decompress_zstd(zst_path, clean_out)

        # Fetch Q&A files
        print(f"\n{'='*60}")
        print(f"  Step 3/3: Fetching Q&A files")
        print(f"{'='*60}")
        api = HfApi(token=token)
        all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
        qa_files = [
            f for f in all_files
            if f.startswith(f"{HF_FINETUNE_DIR}/") and f.endswith(".json")
        ]
        for hf_path in qa_files:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type="dataset",
                token=token,
                cache_dir=HF_CACHE_DIR,
            )
            target = os.path.join(finetune_dir, os.path.basename(hf_path))
            shutil.copy2(local_path, target)
            print(f"  {os.path.basename(hf_path)}")

    # Fetch pre-tokenized BPE bins if requested
    if tokenized and data_dir:
        print(f"\n{'='*60}")
        print(f"  Step 4: Fetching pre-tokenized BPE data")
        print(f"{'='*60}")
        os.makedirs(data_dir, exist_ok=True)

        # Compressed bins
        for name in ("train_bpe.bin.zst", "val_bpe.bin.zst"):
            hf_path = f"{HF_TOKENIZED_DIR}/{name}"
            try:
                cached = hf_hub_download(
                    repo_id=repo_id, filename=hf_path,
                    repo_type="dataset", token=token, cache_dir=HF_CACHE_DIR,
                )
                out_path = os.path.join(data_dir, name.removesuffix(".zst"))
                print(f"  Decompressing {name}...")
                _decompress_zstd(cached, out_path)
            except Exception as e:
                print(f"  Warning: could not fetch {hf_path}: {e}")

        # Small files (tokenizer json, sentencepiece model)
        for name in ("tokenizer_bpe.json", "bpe_model.model"):
            hf_path = f"{HF_TOKENIZED_DIR}/{name}"
            try:
                cached = hf_hub_download(
                    repo_id=repo_id, filename=hf_path,
                    repo_type="dataset", token=token, cache_dir=HF_CACHE_DIR,
                )
                shutil.copy2(cached, os.path.join(data_dir, name))
                print(f"  {name}")
            except Exception as e:
                print(f"  Warning: could not fetch {hf_path}: {e}")

    # Write a sentinel so reruns of 1_download.py know everything was
    # fetched from HF and nothing needs re-downloading from source.
    if not tokenized_only:
        with open(os.path.join(train_dir, ".downloaded_from_hf"), "w") as f:
            f.write(f"{repo_id}\n")

    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"{'='*60}")
    if clean_out:
        print(f"  Corpus:   {clean_out}")
        print(f"  Q&A dir:  {finetune_dir}")
    if tokenized:
        print(f"  Tokenized: {data_dir}")
        print()
        print(f"  Next step: python 4_train.py --preset tiny --tokenizer bpe")
    else:
        print()
        print(f"  Next step: python 3_tokenize.py --tokenizer bpe")


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
                        help="Sources to skip. Corpus: wiki wikisource wiktionary wikiquote "
                             "cc100 hplt3 arlis ccnews opensubtitles culturax mc4 glot500 "
                             "finetranslations. QA: armbench aya")
    parser.add_argument("--workers", type=int, default=5,
                        help="Max parallel HF downloads (corpus mode only; default: 5)")

    # HF publish / fetch
    parser.add_argument("--upload", action="store_true",
                        help="Package clean_text.txt + Q&A files and push to HF "
                             f"(default repo: {DEFAULT_HF_DATASET_REPO})")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the published corpus + Q&A from HF instead of "
                             "running the full source-download pipeline")
    parser.add_argument("--tokenized", action="store_true",
                        help="Include pre-tokenized BPE bins in --upload/--download "
                             "(train_bpe.bin, val_bpe.bin, tokenizer_bpe.json)")
    parser.add_argument("--tokenized-only", action="store_true",
                        help="With --download: fetch ONLY the tokenized BPE bins, "
                             "skip corpus and Q&A")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_DATASET_REPO,
                        help="Override the HF dataset repo for --upload/--download")

    args = parser.parse_args()

    # --upload / --download short-circuit before any other mode
    if args.upload:
        upload_dataset_to_hf(
            repo_id=args.hf_repo,
            corpus_path=os.path.join(TEXT_TRAIN_DIR, "clean_text.txt"),
            finetune_dir=TEXT_FINETUNE_DIR,
            tokenized=args.tokenized,
            data_dir=DATA_DIR,
        )
        return

    if args.download or args.tokenized_only:
        download_dataset_from_hf(
            repo_id=args.hf_repo,
            train_dir=TEXT_TRAIN_DIR,
            finetune_dir=TEXT_FINETUNE_DIR,
            tokenized=args.tokenized,
            tokenized_only=args.tokenized_only,
            data_dir=DATA_DIR,
        )
        return

    if args.qa:
        download_qa(args)
    else:
        download_corpus(args)


if __name__ == "__main__":
    main()
