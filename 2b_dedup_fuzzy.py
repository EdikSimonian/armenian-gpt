"""
Step 2b (optional): MinHash near-duplicate removal across documents.

2_prepare.py removes EXACT duplicate paragraphs. It cannot catch NEAR
duplicates — the same article reworded, boilerplate with one changed line, or
the heavy overlap between the five CommonCrawl-derived sources (cc100, culturax,
mc4, hplt3, finetranslations). Web near-dup rates of 15-30% are typical, and
every surviving near-dup is extra silent epochs on that text → memorization and
wasted capacity. This pass estimates document-to-document Jaccard similarity
with MinHash + LSH and drops the later (lower source-priority) copy of each
near-duplicate cluster.

Runs on CPU, pure numpy + stdlib (no extra dependencies). Operates on the
merged corpus between 2_prepare.py and 3_tokenize.py, splitting on the
<|enddoc|> document separators that 2_prepare preserved:

    python 2_prepare.py
    python 2b_dedup_fuzzy.py          # rewrites data/text/train/clean_text.txt
    python 3_tokenize.py --tokenizer bpe

Tuning (defaults are sensible for an Armenian web corpus):
    --threshold 0.8     keep one of any pair with estimated Jaccard >= this
    --num-perm 128      MinHash permutations (accuracy vs memory/speed)
    --shingle 5         words per shingle (k-gram); lower = catches shorter dups
    --bands 16          LSH bands (with num_perm=128 -> r=8 rows/band)

Memory ~ num_docs * num_perm * 4 bytes for the signature matrix (e.g. 5M docs
* 128 * 4 = ~2.5 GB). Drop --num-perm to 64 to halve it.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import DOC_SEPARATOR  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(_REPO_ROOT, "data", "text", "train", "clean_text.txt")

# A Mersenne prime just under 2^32 — modulus for the (a*h + b) MinHash family,
# so signatures fit in uint32.
_MERSENNE = (1 << 32) - 5
_WORD_SPLIT = None  # set lazily


def iter_documents(path):
    """Yield (text, trailing_had_sep) documents split on DOC_SEPARATOR lines.

    Streams the file so the raw text is never fully held in RAM (only the
    signatures are, later). A document is the text between separators.
    """
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == DOC_SEPARATOR:
                yield "".join(buf)
                buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf)


def shingles(text, k):
    """Set of k-word shingles (as 8-byte hashes). Short docs yield whatever
    words they have as a single shingle so they still get a signature."""
    words = text.split()
    if len(words) < k:
        if not words:
            return np.empty(0, dtype=np.uint64)
        grams = [" ".join(words)]
    else:
        grams = (" ".join(words[i : i + k]) for i in range(len(words) - k + 1))
    hs = [
        int.from_bytes(
            hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little"
        )
        for g in grams
    ]
    # Unique shingles only — multiplicity doesn't affect Jaccard.
    return np.unique(np.array(hs, dtype=np.uint64))


def minhash_signature(shingle_hashes, a, b):
    """MinHash signature: for each of the num_perm (a,b) pairs, the min over
    shingles of (a*h + b) mod prime. Vectorized over shingles."""
    if shingle_hashes.size == 0:
        return np.full(a.shape[0], np.iinfo(np.uint32).max, dtype=np.uint32)
    # (num_perm, num_shingles) then min over axis 1. Compute in uint64 to avoid
    # overflow, reduce mod prime, cast down.
    h = shingle_hashes.astype(np.uint64)[None, :]
    vals = (a[:, None] * h + b[:, None]) % _MERSENNE
    return vals.min(axis=1).astype(np.uint32)


class _DSU:
    """Union-find; documents are merged into near-duplicate clusters."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Attach the higher index under the lower so the survivor (kept doc)
            # is always the lowest index = highest source priority.
            hi, lo = (rx, ry) if rx > ry else (ry, rx)
            self.parent[hi] = lo


def main():
    p = argparse.ArgumentParser(description="MinHash near-duplicate document removal")
    p.add_argument("--input", default=DEFAULT_CORPUS, help="Cleaned corpus file")
    p.add_argument(
        "--output", default=None, help="Default: overwrite input (keeps .bak)"
    )
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--shingle", type=int, default=5, help="Words per shingle")
    p.add_argument("--bands", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: {args.input} not found. Run 2_prepare.py first.")
    if args.num_perm % args.bands != 0:
        sys.exit(
            f"--num-perm ({args.num_perm}) must be divisible by --bands ({args.bands})"
        )
    rows = args.num_perm // args.bands

    print(f"\n{'=' * 60}\n  Step 2b: MinHash near-dedup\n{'=' * 60}")
    print(f"  Input:      {args.input}")
    print(f"  num_perm={args.num_perm}  bands={args.bands} (rows/band={rows})")
    print(f"  threshold={args.threshold}  shingle={args.shingle} words")
    approx = (1.0 / args.bands) ** (1.0 / rows)
    print(
        f"  LSH catches pairs with Jaccard >~ {approx:.2f}; confirmed >= {args.threshold}"
    )
    print(f"{'=' * 60}\n")

    rng = np.random.default_rng(args.seed)
    a = rng.integers(1, _MERSENNE, size=args.num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=args.num_perm, dtype=np.uint64)

    # Pass 1: signatures (held in RAM; the document text is re-read in pass 2).
    print("Pass 1/2: hashing documents...")
    sigs = []
    n_docs = 0
    for doc in iter_documents(args.input):
        sigs.append(minhash_signature(shingles(doc, args.shingle), a, b))
        n_docs += 1
        if n_docs % 200_000 == 0:
            print(f"  {n_docs:,} docs hashed...")
    if n_docs == 0:
        sys.exit("No documents found (is the corpus empty / missing separators?)")
    sig = np.stack(sigs)
    del sigs
    print(f"  {n_docs:,} documents, signature matrix {sig.nbytes / 1e6:.0f} MB")

    # LSH: bucket by each band; same band sub-signature -> candidate pair.
    print("Banding + union-find...")
    dsu = _DSU(n_docs)
    n_pairs_checked = 0
    for band in range(args.bands):
        sub = sig[:, band * rows : (band + 1) * rows]
        buckets = {}
        for i in range(n_docs):
            key = sub[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            asig = sig[anchor]
            for j in members[1:]:
                if dsu.find(anchor) == dsu.find(j):
                    continue
                # Confirm with the full-signature agreement (unbiased Jaccard est).
                est = float(np.mean(asig == sig[j]))
                n_pairs_checked += 1
                if est >= args.threshold:
                    dsu.union(anchor, j)

    # A doc is dropped iff it is not the representative (lowest index) of its
    # cluster.
    drop = np.zeros(n_docs, dtype=bool)
    for i in range(n_docs):
        if dsu.find(i) != i:
            drop[i] = True
    n_drop = int(drop.sum())
    print(
        f"  Candidate pairs checked: {n_pairs_checked:,} | "
        f"near-dup docs dropped: {n_drop:,} ({100 * n_drop / n_docs:.1f}%)"
    )

    # Pass 2: stream-rewrite, keeping non-dropped documents.
    out_path = args.output or args.input
    if out_path == args.input and not args.output:
        bak = args.input + ".bak"
        if not os.path.exists(bak):
            os.replace(args.input, bak)
            src = bak
        else:
            src = bak  # a .bak already exists from a prior run; reuse as source
        print(f"  (original backed up to {bak})")
    else:
        src = args.input

    print("Pass 2/2: writing kept documents...")
    kept = 0
    first = True
    with open(out_path, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as out:
        for i, doc in enumerate(iter_documents(src)):
            if drop[i]:
                continue
            text = doc.strip("\n")
            if not text:
                continue
            if not first:
                out.write(DOC_SEPARATOR + "\n\n")
            out.write(text)
            out.write("\n\n")
            first = False
            kept += 1

    print(f"\n{'=' * 60}\n  Step 2b complete\n{'=' * 60}")
    print(f"  Documents in:  {n_docs:,}")
    print(f"  Dropped:       {n_drop:,}")
    print(f"  Documents out: {kept:,}")
    print(f"  Output:        {out_path}")
    print("\nNext step: python 3_tokenize.py --tokenizer bpe")


if __name__ == "__main__":
    main()
