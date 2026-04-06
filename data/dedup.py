"""
Deduplicate raw text data by removing exact and near-duplicate lines.

Web crawl data (CC-100, mC4) contains many duplicate paragraphs — boilerplate,
navigation text, copy-paste content. Removing these improves training quality.

Usage:
    python data/dedup.py              # dedup data/raw_text.txt in place
    python data/dedup.py --min_len 50 # only keep lines with 50+ chars
"""

import os
import sys
import hashlib
import argparse

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def dedup_file(input_path, output_path, min_len=50):
    """Remove duplicate lines and very short lines."""
    seen = set()
    total_lines = 0
    kept_lines = 0
    total_chars = 0
    kept_chars = 0

    print(f"Deduplicating {input_path}...")
    print(f"  Min line length: {min_len}")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        buf = []
        for line in f_in:
            total_lines += 1
            total_chars += len(line)

            # Group by paragraphs (separated by blank lines)
            stripped = line.strip()
            if not stripped:
                # Process buffered paragraph
                if buf:
                    para = " ".join(buf)
                    if len(para) >= min_len:
                        h = hashlib.md5(para.encode("utf-8")).digest()
                        if h not in seen:
                            seen.add(h)
                            f_out.write(para + "\n")
                            kept_lines += 1
                            kept_chars += len(para)
                    buf = []
                continue
            buf.append(stripped)

            if total_lines % 10_000_000 == 0:
                pct = 100 * kept_chars / total_chars if total_chars > 0 else 0
                print(f"  {total_lines / 1_000_000:.0f}M lines, kept {pct:.0f}%")

        # Final paragraph
        if buf:
            para = " ".join(buf)
            if len(para) >= min_len:
                h = hashlib.md5(para.encode("utf-8")).digest()
                if h not in seen:
                    seen.add(h)
                    f_out.write(para + "\n")
                    kept_lines += 1
                    kept_chars += len(para)

    removed_pct = 100 * (1 - kept_chars / total_chars) if total_chars > 0 else 0
    print(f"\n  Input:    {total_chars / 1_000_000_000:.1f} GB ({total_lines:,} lines)")
    print(f"  Output:   {kept_chars / 1_000_000_000:.1f} GB ({kept_lines:,} paragraphs)")
    print(f"  Removed:  {removed_pct:.1f}% duplicate/short content")
    print(f"  Unique hashes: {len(seen):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(DATA_DIR, "raw_text.txt"))
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "raw_text_dedup.txt"))
    parser.add_argument("--min_len", type=int, default=50)
    args = parser.parse_args()

    dedup_file(args.input, args.output, args.min_len)

    # Replace original with deduped version
    orig_size = os.path.getsize(args.input)
    dedup_size = os.path.getsize(args.output)
    print(f"\n  {orig_size/1024/1024/1024:.1f} GB -> {dedup_size/1024/1024/1024:.1f} GB")
    print(f"  Replacing {args.input} with deduped version...")
    os.replace(args.output, args.input)
    print("  Done!")


if __name__ == "__main__":
    main()
