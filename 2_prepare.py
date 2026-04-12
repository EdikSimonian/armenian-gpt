"""
Step 2: Clean + dedup + merge the corpus sources.

Corpus mode (default):
    Reads the 12 per-source `{name}_hy.txt` files produced by 1_download.py,
    cleans each (NFC normalize, Armenian-script whitelist, collapse
    whitespace), deduplicates at the paragraph level across ALL sources,
    and merges into a single data/text/train/clean_text.txt ready for
    the tokenizer (3_tokenize.py).

    Dedup is exact at the paragraph level using blake2b-64 hashes.
    Sources are processed in quality-priority order — paragraphs from
    earlier (higher-quality / native) sources win dedup ties, so the
    web-crawl sources act as fillers for gaps the curated sources miss.

--qa mode:
    Merges SFT JSON files under data/text/finetune/ produced by
    1_download.py --qa (+ any generated Q&A pairs) into a deduplicated
    qa_merged.json.

Usage:
    python 2_prepare.py              # corpus clean + dedup + merge
    python 2_prepare.py --no-dedup   # corpus clean + merge, no dedup
    python 2_prepare.py --qa         # Q&A merge

Outputs:
    corpus: data/text/train/clean_text.txt
            data/text/train/clean_stats.json
    qa:     data/text/finetune/qa_merged.json
"""

import hashlib
import json
import os
import re
import sys
import time
import unicodedata

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
TEXT_TRAIN_DIR = os.path.join(DATA_DIR, "text", "train")
TEXT_FINETUNE_DIR = os.path.join(DATA_DIR, "text", "finetune")
CLEAN_FILE = os.path.join(TEXT_TRAIN_DIR, "clean_text.txt")
STATS_FILE = os.path.join(TEXT_TRAIN_DIR, "clean_stats.json")

# Source ordering for dedup-priority. Sources listed first "win" dedup ties
# when their paragraphs also appear in lower-priority web-crawl sources.
# Order rationale:
#   1. Wikimedia (native curated, CC BY-SA)
#   2. ARLIS (formal legal Armenian, unique domain)
#   3. News / small curated
#   4. HPLT 3.0 (best-extracted web)
#   5. CulturaX, FineTranslations (cleaned web variants)
#   6. mC4, CC-100 (older/noisier web baselines)
SOURCE_ORDER = [
    "wiki",
    "wikisource",
    "wiktionary",
    "wikiquote",
    "arlis",
    "ccnews",
    "glot500",
    "hplt3",
    "culturax",
    "finetranslations",
    "mc4",
    "cc100",
]

# Minimum paragraph length (in cleaned chars) to keep. Drops navigation
# leftovers, single-word fragments, and broken OCR lines.
MIN_PARAGRAPH_CHARS = 50

# Read/write buffer (bytes)
IO_BUFFER = 16 * 1024 * 1024

# Regex patterns compiled once at module load.
# Armenian block U+0530-U+058F covers the alphabet. U+FB13-U+FB17 are
# ligatures that appear in older printing. We also keep ASCII punctuation,
# digits, and whitespace so the tokenizer sees real sentences.
_RE_NON_ARMENIAN = re.compile(
    r"[^\u0530-\u058F\uFB13-\uFB17 \n\t.,;:!?\-()\"'0-9]"
)
_RE_SPACES = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def clean_chunk(chunk: str) -> str:
    """Normalize + whitelist-filter + collapse whitespace on one text blob."""
    chunk = unicodedata.normalize("NFC", chunk)
    chunk = _RE_NON_ARMENIAN.sub("", chunk)
    chunk = _RE_SPACES.sub(" ", chunk)
    chunk = _RE_NEWLINES.sub("\n\n", chunk)
    return chunk.strip()


def hash_paragraph(text: str) -> bytes:
    """Cheap 64-bit hash of a cleaned paragraph for cross-source dedup.

    blake2b is in the stdlib, faster than md5 for short strings, and
    64-bit digest gives 1.8e19 distinct values — collision probability
    is negligible even for 100M paragraphs (~10^-4 total by birthday
    paradox).
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def iter_paragraphs(fp, min_bytes=50):
    """Stream paragraphs from a text file, delimited by blank lines.

    Reads in chunks and yields complete paragraphs (strings). Incomplete
    paragraphs at chunk boundaries are carried over. Handles multi-GB
    files without materializing them in memory.
    """
    carry = ""
    while True:
        chunk = fp.read(IO_BUFFER)
        if not chunk:
            if carry.strip():
                yield carry
            return
        combined = carry + chunk
        parts = _RE_PARAGRAPH_SPLIT.split(combined)
        # Last element may be a partial paragraph — carry it over
        # unless EOF on next iteration.
        carry = parts.pop()
        for p in parts:
            if len(p) >= min_bytes:
                yield p


def process_source(src_name: str, src_path: str, seen: set, fout,
                   dedup: bool, stats: dict) -> None:
    """Clean + optionally dedup one source file, writing to the merged output.

    Updates `seen` (global hash set) and `stats` in place. Writes kept
    paragraphs to `fout`, separated by blank lines.
    """
    if not os.path.exists(src_path):
        print(f"  [{src_name}] MISSING ({src_path}) - skipping")
        stats[src_name] = {"status": "missing"}
        return

    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    t0 = time.time()
    in_chars = 0
    out_chars = 0
    kept_paras = 0
    drop_short = 0
    drop_dupes = 0

    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin:
        for para in iter_paragraphs(fin):
            in_chars += len(para)
            cleaned = clean_chunk(para)
            if len(cleaned) < MIN_PARAGRAPH_CHARS:
                drop_short += 1
                continue
            if dedup:
                h = hash_paragraph(cleaned)
                if h in seen:
                    drop_dupes += 1
                    continue
                seen.add(h)
            fout.write(cleaned)
            fout.write("\n\n")
            out_chars += len(cleaned) + 2
            kept_paras += 1

    elapsed = time.time() - t0
    compression = (1.0 - out_chars / in_chars) * 100 if in_chars else 0.0

    stats[src_name] = {
        "status": "ok",
        "size_mb_in": size_mb,
        "chars_in": in_chars,
        "chars_out": out_chars,
        "kept_paragraphs": kept_paras,
        "dropped_short": drop_short,
        "dropped_duplicates": drop_dupes,
        "compression_pct": round(compression, 1),
        "elapsed_sec": round(elapsed, 1),
    }

    dup_pct = (drop_dupes / (kept_paras + drop_dupes) * 100
               if (kept_paras + drop_dupes) else 0.0)
    print(
        f"  [{src_name:17s}] "
        f"{size_mb:>6.0f} MB in -> {out_chars / 1024 / 1024:>6.0f} MB out  "
        f"(-{compression:>4.1f}%)  "
        f"kept {kept_paras:>8,}  "
        f"dupes {drop_dupes:>8,} ({dup_pct:>4.1f}%)  "
        f"short {drop_short:>6,}  "
        f"in {elapsed:>5.0f}s"
    )


def prepare_corpus(no_dedup: bool = False) -> None:
    """Clean + dedup + merge all 12 per-source corpus files."""
    # Resolve source paths and verify at least some exist before starting.
    sources = [
        (name, os.path.join(TEXT_TRAIN_DIR, f"{name}_hy.txt"))
        for name in SOURCE_ORDER
    ]
    existing = [(n, p) for n, p in sources if os.path.exists(p)]
    missing = [n for n, p in sources if not os.path.exists(p)]

    if not existing:
        print(f"Error: no source files found in {TEXT_TRAIN_DIR}.")
        print("Run 'python 1_download.py' first.")
        sys.exit(1)

    total_bytes = sum(os.path.getsize(p) for _, p in existing)
    total_gb = total_bytes / (1024 ** 3)

    print(f"\n{'=' * 60}")
    print(f"  Step 2: Clean + dedup + merge corpus")
    print(f"{'=' * 60}")
    print(f"  Sources found:  {len(existing)}/{len(sources)}")
    print(f"  Sources missing: {missing if missing else '(none)'}")
    print(f"  Total input:    {total_gb:.2f} GB")
    print(f"  Dedup:          {'ON (paragraph-level, blake2b-64)' if not no_dedup else 'OFF'}")
    print(f"  Min paragraph:  {MIN_PARAGRAPH_CHARS} chars (after cleaning)")
    print(f"  Output:         {CLEAN_FILE}")
    print(f"{'=' * 60}\n")

    seen: set = set() if not no_dedup else None
    stats: dict = {"sources": {}, "order": SOURCE_ORDER}
    total_t0 = time.time()

    with open(CLEAN_FILE, "w", encoding="utf-8", buffering=IO_BUFFER) as fout:
        for src_name, src_path in sources:
            process_source(
                src_name, src_path,
                seen if seen is not None else set(),
                fout, dedup=not no_dedup,
                stats=stats["sources"],
            )

    elapsed = time.time() - total_t0
    clean_size = os.path.getsize(CLEAN_FILE)
    clean_gb = clean_size / (1024 ** 3)

    # Aggregate totals across sources.
    total_in = sum(s.get("chars_in", 0) for s in stats["sources"].values())
    total_out = sum(s.get("chars_out", 0) for s in stats["sources"].values())
    total_kept = sum(s.get("kept_paragraphs", 0) for s in stats["sources"].values())
    total_dupes = sum(s.get("dropped_duplicates", 0) for s in stats["sources"].values())
    total_short = sum(s.get("dropped_short", 0) for s in stats["sources"].values())
    global_compression = (1.0 - total_out / total_in) * 100 if total_in else 0.0

    stats["totals"] = {
        "chars_in": total_in,
        "chars_out": total_out,
        "kept_paragraphs": total_kept,
        "dropped_duplicates": total_dupes,
        "dropped_short": total_short,
        "compression_pct": round(global_compression, 1),
        "dedup_enabled": not no_dedup,
        "wall_time_sec": round(elapsed, 1),
        "output_bytes": clean_size,
        "output_gb": round(clean_gb, 2),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Step 2 complete")
    print(f"{'=' * 60}")
    print(f"  Input total:     {total_in / 1e9:>8.2f} G chars")
    print(f"  Output total:    {total_out / 1e9:>8.2f} G chars  (-{global_compression:.1f}%)")
    print(f"  Kept paragraphs: {total_kept:,}")
    if not no_dedup:
        dedup_rate = (total_dupes / (total_kept + total_dupes) * 100
                      if (total_kept + total_dupes) else 0.0)
        print(f"  Dedup drops:     {total_dupes:,} ({dedup_rate:.1f}% of all seen)")
    print(f"  Short drops:     {total_short:,}")
    print(f"  Final file:      {clean_gb:.2f} GB ({clean_size:,} bytes)")
    print(f"  Wall time:       {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Stats sidecar:   {STATS_FILE}")
    print()
    print(f"Next step: python 3_tokenize.py --tokenizer bpe")


def prepare_qa() -> None:
    """Merge the SFT source JSONs into data/text/finetune/qa_merged.json."""
    from core.merge_sft_sources import merge_sft_sources

    # Inputs in priority order — earlier sources win dedup ties.
    # armenian_qa.json and armenian_qa_qwen*.json are only present if the
    # user ran the optional generators; they're listed first so their
    # native/curated pairs take priority over the larger translated sets.
    candidates = [
        "armenian_qa.json",           # Claude-generated (optional)
        "armenian_qa_qwen.json",      # Qwen long-form (optional)
        "armenian_qa_qwen_short.json",  # Qwen short (optional)
        "armbench_train.json",        # native exam QA
        "aya_armenian.json",          # filtered Aya (mostly translated)
    ]
    input_paths = [
        os.path.join(TEXT_FINETUNE_DIR, f)
        for f in candidates
        if os.path.exists(os.path.join(TEXT_FINETUNE_DIR, f))
    ]
    output_path = os.path.join(TEXT_FINETUNE_DIR, "qa_merged.json")

    print(f"\n{'=' * 60}")
    print(f"  Step 2: Merge SFT sources (--qa)")
    print(f"{'=' * 60}")
    print(f"  Input files: {len(input_paths)}")
    for p in input_paths:
        print(f"    {os.path.basename(p)}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 60}\n")

    if not input_paths:
        print(f"Error: no Q&A sources found in {TEXT_FINETUNE_DIR}.")
        print("Run 'python 1_download.py --qa' first.")
        sys.exit(1)

    n = merge_sft_sources(input_paths, output_path)
    if n == 0:
        print("merge returned 0 pairs - check source files for parseable JSON")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Step 2 (--qa) complete: {n:,} unique pairs")
    print(f"{'=' * 60}")
    print("Next step: python 3_tokenize.py --qa --tokenizer bpe")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Clean/dedup/merge corpus sources, or merge SFT Q&A JSONs"
    )
    parser.add_argument("--qa", action="store_true",
                        help="Merge SFT Q&A JSON files instead of corpus cleaning")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip paragraph-level dedup (corpus mode only)")
    args = parser.parse_args()

    if args.qa:
        prepare_qa()
    else:
        prepare_corpus(no_dedup=args.no_dedup)


if __name__ == "__main__":
    main()
