"""
Merge multiple SFT JSON files into a single {instruction, input, output}
dataset with global dedup by normalized instruction.

Default inputs:
  - data/armenian_qa_merged.json  (Claude + filtered Aya, from fetch_aya_armenian.py)
  - data/armbench_train.json      (ArmBench native training split, from fetch_armbench.py)

Default output:
  - data/armenian_qa_merged.json  (overwritten in place)

Usage:
    python data/merge_sft_sources.py
    python data/merge_sft_sources.py --inputs a.json b.json c.json --output merged.json
"""

import argparse
import json
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_key(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s or "").strip().lower()


def merge_sft_sources(input_paths, output_path):
    """Merge multiple SFT JSON files with global dedup by normalized instruction.

    Earlier input files take priority (their copies of duplicates win).
    Missing input files are skipped silently. Returns number of unique pairs.
    """
    print("=" * 60)
    print("  SFT source merger")
    print("=" * 60)

    all_pairs = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"  [skip] {path}: not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [load] {path}: {len(data):,} rows")
        all_pairs.extend(data)

    print(f"\n  Combined before dedup: {len(all_pairs):,}")

    seen = set()
    unique = []
    dropped_per_source = {}
    for p in all_pairs:
        key = _normalize_key(p.get("instruction", ""))
        if not key:
            continue
        if key in seen:
            src = p.get("source", "?")
            dropped_per_source[src] = dropped_per_source.get(src, 0) + 1
            continue
        seen.add(key)
        unique.append(p)

    print(f"  After dedup:           {len(unique):,}  (dropped {len(all_pairs) - len(unique):,})")

    by_source = {}
    for p in unique:
        src = p.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    print(f"\n  Final counts by source:")
    for src, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"     {count:>7,}  {src}")

    if dropped_per_source:
        print(f"\n  Duplicates dropped by source:")
        for src, count in sorted(dropped_per_source.items(), key=lambda kv: -kv[1]):
            print(f"     {count:>7,}  {src}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {output_path}")
    return len(unique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+",
        default=[
            os.path.join(DATA_DIR, "armenian_qa.json"),
            os.path.join(DATA_DIR, "armbench_train.json"),
            os.path.join(DATA_DIR, "aya_armenian.json"),
        ],
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "qa_merged.json"),
    )
    args = parser.parse_args()
    merge_sft_sources(args.inputs, args.output)


if __name__ == "__main__":
    main()
