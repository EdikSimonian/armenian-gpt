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
import random
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
TEXT_FINETUNE_DIR = os.path.join(DATA_DIR, "text", "finetune")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_key(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s or "").strip().lower()


def merge_sft_sources(input_paths, output_path, weights=None, seed=1234):
    """Merge multiple SFT JSON files with global dedup by normalized instruction.

    Earlier input files take priority (their copies of duplicates win).
    Missing input files are skipped silently.

    `weights` rebalances the mix per source FILE (keyed by basename), applied
    AFTER dedup:
        {"aya_armenian.json": {"cap": 5000}, "armbench_train.json": {"repeat": 2}}
      - cap N:    randomly subsample that file's surviving pairs down to N
                  (fixed seed -> deterministic). Use to stop one huge, narrow
                  source (Aya rephrase tasks) from dominating the objective.
      - repeat K: duplicate that file's pairs K times. Prefer cap-others over
                  repeat: duplicating a tiny set replays identical gradient
                  steps and invites memorization. repeat defaults to 1.
    Files not listed in `weights` are passed through unchanged. Returns the
    number of pairs written.
    """
    weights = weights or {}
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
        base = os.path.basename(path)
        for p in data:
            p["_origin"] = base  # tag origin file (qwen rows carry no "source")
        print(f"  [load] {path}: {len(data):,} rows")
        all_pairs.extend(data)

    print(f"\n  Combined before dedup: {len(all_pairs):,}")

    seen = set()
    unique = []
    dropped_per_source = {}
    for p in all_pairs:
        # Key on instruction + input: examples sharing an instruction but with
        # different inputs (e.g. a generic "Summarize:" prompt over different
        # passages) are distinct and must not collapse. Current sources have no
        # input field, but generated QA (#4) might.
        instr_key = _normalize_key(p.get("instruction", ""))
        key = instr_key + "\x00" + _normalize_key(p.get("input", ""))
        if not instr_key:
            continue
        if key in seen:
            src = p.get("source", p.get("_origin", "?"))
            dropped_per_source[src] = dropped_per_source.get(src, 0) + 1
            continue
        seen.add(key)
        unique.append(p)

    print(
        f"  After dedup:           {len(unique):,}  "
        f"(dropped {len(all_pairs) - len(unique):,})"
    )

    # --- Rebalance per source file (cap / repeat), preserving input order ----
    groups = {}
    for p in unique:
        groups.setdefault(p["_origin"], []).append(p)

    rng = random.Random(seed)
    rebalance_log = []
    out_pairs = []
    for path in input_paths:
        base = os.path.basename(path)
        items = groups.get(base)
        if not items:
            continue
        n_in = len(items)
        w = weights.get(base, {})
        cap = w.get("cap")
        repeat = int(w.get("repeat", 1))
        if cap is not None and n_in > cap:
            items = rng.sample(items, cap)
        if repeat > 1:
            items = items * repeat
        rebalance_log.append((base, n_in, len(items), cap, repeat))
        out_pairs.extend(items)

    # Strip the temporary origin tag so the written file stays clean.
    for p in out_pairs:
        p.pop("_origin", None)

    print("\n  Final counts by source file (after rebalance):")
    for base, n_in, n_out, cap, repeat in rebalance_log:
        note = ""
        if cap is not None and n_in > cap:
            note = f"  (capped from {n_in:,})"
        elif repeat > 1:
            note = f"  (repeated x{repeat} from {n_in:,})"
        print(f"     {n_out:>7,}  {base}{note}")
    print(f"     {len(out_pairs):>7,}  TOTAL")

    if dropped_per_source:
        print("\n  Duplicates dropped by source:")
        for src, count in sorted(dropped_per_source.items(), key=lambda kv: -kv[1]):
            print(f"     {count:>7,}  {src}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved -> {output_path}")
    return len(out_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            os.path.join(TEXT_FINETUNE_DIR, "armenian_qa.json"),
            os.path.join(TEXT_FINETUNE_DIR, "armbench_train.json"),
            os.path.join(TEXT_FINETUNE_DIR, "aya_armenian.json"),
        ],
    )
    parser.add_argument(
        "--output",
        default=os.path.join(TEXT_FINETUNE_DIR, "qa_merged.json"),
    )
    args = parser.parse_args()
    merge_sft_sources(args.inputs, args.output)


if __name__ == "__main__":
    main()
