"""
Pull Armenian instruction data from CohereLabs/aya_collection_language_split
and filter aggressively for quality before merging into the
{instruction, input, output} format used by prepare_chat.py.

KEY INSIGHT — source distribution of the Armenian slice (3.58M rows, 19 sources):
  Only ONE source is native Armenian: Arpa-instruct (4,017 rows, no (T) suffix).
  Everything else is NLLB-translated English. That means quality filtering is
  essential — the naive version of this script produced visibly garbled output
  like 'Կրակոցներ Հարրի' (MT of the name "Harry" → Armenian word for "shoots").

The quality heuristics applied here:
  1. Source allowlist + per-source sample caps (diversity over volume)
  2. Armenian-letter ratio on the answer (reject Latin/English leakage)
  3. Length bounds (drop stubs and runaway outputs)
  4. Artifact rejection: <unk>, [unk], [UNK], stray HTML/template markers
  5. Question/answer collapse rejection (answer ≠ instruction verbatim)
  6. Global dedup by normalized instruction

Sources kept (priority order):
  - Arpa-instruct      : native Armenian paraphrasing (take ALL ~4K)
  - Dolly-v2 (T)       : human-written instructions, translated (take ALL ~14K)
  - HotpotQA (T)       : multi-hop factual QA (sample)
  - NQ-Open (T)        : Natural Questions, high-quality source (sample)
  - Mintaka-inst (T)   : factual QA (sample)
  - Adversarial QA (T) : curated QA (sample)
  - Flan-CoT-submix (T): chain-of-thought reasoning (sample)
  - Flan-unified-QA (T): small curated QA set (take ALL)
  - WIKI QA (T)        : small wiki QA (take ALL)

Usage:
    python data/fetch_aya_armenian.py
    python data/fetch_aya_armenian.py --output data/armenian_qa_merged.json
    python data/fetch_aya_armenian.py --min_arm_ratio 0.80  # stricter
"""

import argparse
import json
import os
import random
import re
import sys

from datasets import load_dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# Per-source sampling plan. None = take everything.
# Sources not listed here are dropped entirely.
#
# Dropped after quality spot-check:
#   Mintaka-inst (T): task format ("generate trivia in category X") translates
#                     to nonsensical Q/A blobs.
#   NQ-Open (T):      short-answer factoids where MT mangles the answer.
#   WIKI QA (T):      inverted format — given the answer, produce the question.
#                     Wrong direction for chat SFT.
SOURCE_PLAN = {
    "Arpa-instruct":        None,    # native Armenian, take all (~4K)
    "Dolly-v2 (T)":         None,    # ~14K, all
    "Dolly-v2":             None,    # future-proof if they drop the (T)
    "HotpotQA (T)":         10000,   # filter is aggressive here, yields ~21%
    "Adversarial QA (T)":   5000,
    "Flan-CoT-submix (T)":  5000,
    "Flan-unified-QA (T)":  None,    # only ~540, take all
}


# Armenian Unicode ranges: main block U+0530–U+058F, ligatures U+FB13–U+FB17
def armenian_letter_ratio(s: str) -> float:
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


_ARTIFACT_PATTERNS = [
    "<unk>", "[unk]", "[UNK]", "<|", "|>", "{{", "}}",
    "[[", "]]",  # catches Wiki-style remnants
]

_WHITESPACE_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    """Strip and collapse whitespace."""
    if s is None:
        return ""
    return _WHITESPACE_RE.sub(" ", s).strip()


def _normalize_key(s: str) -> str:
    """Hash key for dedup — case-insensitive, whitespace-insensitive."""
    return _WHITESPACE_RE.sub(" ", s or "").strip().lower()


def to_pair(row, *, min_q_len, max_q_len, min_a_len, max_a_len, min_arm_ratio):
    """Convert one Aya row to an SFT pair or None if it fails quality checks.

    Returns dict with keys {instruction, input, output, source} or None.
    """
    q = _clean(row.get("inputs"))
    a = _clean(row.get("targets"))
    if not q or not a:
        return None

    # Length bounds
    if not (min_q_len <= len(q) <= max_q_len):
        return None
    if not (min_a_len <= len(a) <= max_a_len):
        return None

    # Artifact rejection (MT tokenization failures, template leftovers)
    for pat in _ARTIFACT_PATTERNS:
        if pat in q or pat in a:
            return None

    # Language purity — the answer should be predominantly Armenian script.
    if armenian_letter_ratio(a) < min_arm_ratio:
        return None
    # Instruction can have some English prompt scaffolding; be looser but
    # still catch fully-English rows.
    if armenian_letter_ratio(q) < (min_arm_ratio - 0.2):
        return None

    # Collapse rejection: answer identical to instruction is usually a bad
    # paraphrase task result.
    if _normalize_key(q) == _normalize_key(a):
        return None

    # Suspicious stub: very long instruction but trivial answer (<15 chars).
    # This is the signature of the "Harry/Կրակոցներ" failure case.
    if len(q) > 300 and len(a) < 15:
        return None

    return {
        "instruction": q,
        "input": "",
        "output": a,
        "source": f"aya/{row.get('dataset_name', '?')}",
    }


def process_source(ds, name, n_samples, rng, filters):
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
        pair = to_pair(row, **filters)
        if pair is None:
            dropped += 1
            continue
        kept.append(pair)
    yield_rate = 100.0 * len(kept) / max(pool_size, 1)
    print(f"  {name}: kept {len(kept):,} / {pool_size:,} ({yield_rate:.1f}% yield, {dropped:,} dropped)")
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

    Writes ONLY the Aya-derived pairs (no merging with existing files).
    Merging with other sources is handled by 2_prepare.py --qa.

    Returns the number of pairs written.
    """
    rng = random.Random(seed)
    plan = plan or dict(SOURCE_PLAN)

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

    print("\nLoading CohereLabs/aya_collection_language_split (armenian/train)...")
    ds = load_dataset("CohereLabs/aya_collection_language_split",
                      "armenian", split="train")
    print(f"  Total Armenian rows: {len(ds):,}")

    all_pairs = []
    per_source_counts = {}
    for name, n_samples in plan.items():
        kept = process_source(ds, name, n_samples, rng, filters)
        all_pairs.extend(kept)
        per_source_counts[name] = len(kept)

    # Per-source dedup only — cross-source dedup happens in 2_prepare.py
    seen = set()
    unique = []
    for p in all_pairs:
        key = _normalize_key(p["instruction"])
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
    print(f"  Total kept: {len(unique):,} (dropped {len(all_pairs) - len(unique):,} intra-source dupes)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {output_path}")

    return len(unique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=os.path.join(DATA_DIR, "aya_armenian.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_q_len", type=int, default=10)
    parser.add_argument("--max_q_len", type=int, default=2000)
    parser.add_argument("--min_a_len", type=int, default=20)
    parser.add_argument("--max_a_len", type=int, default=2000)
    parser.add_argument("--min_arm_ratio", type=float, default=0.75)
    parser.add_argument("--hotpot_samples",  type=int, default=None)
    parser.add_argument("--flancot_samples", type=int, default=None)
    parser.add_argument("--adv_samples",     type=int, default=None)
    args = parser.parse_args()

    plan = dict(SOURCE_PLAN)
    if args.hotpot_samples is not None:
        plan["HotpotQA (T)"] = args.hotpot_samples
    if args.flancot_samples is not None:
        plan["Flan-CoT-submix (T)"] = args.flancot_samples
    if args.adv_samples is not None:
        plan["Adversarial QA (T)"] = args.adv_samples

    fetch_aya_qa(
        output_path=args.output,
        seed=args.seed,
        min_q_len=args.min_q_len,
        max_q_len=args.max_q_len,
        min_a_len=args.min_a_len,
        max_a_len=args.max_a_len,
        min_arm_ratio=args.min_arm_ratio,
        plan=plan,
    )


if __name__ == "__main__":
    main()
