"""
Build two distinct 10k SmolTalk2 fine-tuning sets for ArmGPT.

Set 1 (chat):  multi-turn conversational, target distribution mixes short/med/long
Set 2 (tasks): single-turn instruction completion, balanced across task types

Filters applied to both:
  - 80 ≤ total chars ≤ 3500  (~875 token cap, leaves room for chat-template wrap)
  - no code blocks (```)
  - no URLs (https://, http://)
  - no message < 20 chars
  - dedup by first 100 chars of first user message
  - chat set additionally: ≥4 messages (≥2 user-assistant pairs)
  - chat set additionally: drop where any message contains "```" or "http"
  - reproducible random seed = 42
  - the two sets are guaranteed disjoint by content hash
"""
import os, glob, re, json, hashlib, random
import pandas as pd
from collections import defaultdict

random.seed(42)
DATA = "/workspace/finetune/smoltalk2/SFT"
OUT = "/workspace/finetune/sets"
os.makedirs(OUT, exist_ok=True)
SHARD = re.compile(r'-\d{5}-of-\d{5}\.parquet$')

MIN_CHARS = 80
MAX_CHARS = 3500
MIN_MSG_CHARS = 20

def load_subset(name):
    paths = sorted(glob.glob(f"{DATA}/{name}-*.parquet"))
    dfs = [pd.read_parquet(p, columns=["messages", "source"]) for p in paths]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def is_clean(msgs):
    """Quality filters applied to all examples."""
    for m in msgs:
        c = m["content"]
        if not isinstance(c, str): return False
        if len(c) < MIN_MSG_CHARS: return False
        if "```" in c: return False           # code blocks translate poorly
        if "http://" in c or "https://" in c: return False
        if c.count("\n") > 30: return False   # very long structured content
    return True

def total_chars(msgs):
    return sum(len(m["content"]) for m in msgs)

def length_bucket(n):
    if n < 500: return "short"
    if n < 1500: return "med"
    return "long"

def to_record(msgs, source, subset):
    msgs_list = [{"role": m["role"], "content": m["content"]} for m in msgs]
    chars = total_chars(msgs_list)
    return {
        "messages": msgs_list,
        "n_turns": len(msgs_list),
        "n_chars": chars,
        "length_bucket": length_bucket(chars),
        "source": source,
        "subset": subset,
    }

def content_hash(msgs):
    """First 200 chars of first user message — used for cross-set dedup."""
    first = msgs[0]["content"][:200] if msgs else ""
    return hashlib.md5(first.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Build candidate pools
# ---------------------------------------------------------------------------

print("Loading subsets and applying filters...\n")

CHAT_SUBSETS = [
    "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_smol_magpie_ultra_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think",
]
TASK_SUBSETS = [
    "tulu_3_sft_personas_instruction_following_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
    "smoltalk_smollm3_explore_instruct_rewriting_no_think",
    "OpenHermes_2.5_no_think",
]

chat_pool = []
task_pool_per_subset = defaultdict(list)
seen_hashes = set()

for subset in CHAT_SUBSETS:
    df = load_subset(subset)
    n_in = len(df)
    n_kept = 0
    for _, row in df.iterrows():
        msgs = list(row["messages"])
        if len(msgs) < 4: continue
        if not is_clean(msgs): continue
        chars = total_chars(msgs)
        if not (MIN_CHARS <= chars <= MAX_CHARS): continue
        h = content_hash(msgs)
        if h in seen_hashes: continue
        seen_hashes.add(h)
        chat_pool.append((to_record(msgs, str(row["source"]), subset), h))
        n_kept += 1
    print(f"  CHAT  {subset[:55]:<55} {n_in:>7,} -> {n_kept:>6,}")

for subset in TASK_SUBSETS:
    df = load_subset(subset)
    n_in = len(df)
    n_kept = 0
    for _, row in df.iterrows():
        msgs = list(row["messages"])
        if len(msgs) < 2: continue
        if not is_clean(msgs): continue
        chars = total_chars(msgs)
        if not (MIN_CHARS <= chars <= MAX_CHARS): continue
        h = content_hash(msgs)
        if h in seen_hashes: continue
        seen_hashes.add(h)
        task_pool_per_subset[subset].append((to_record(msgs, str(row["source"]), subset), h))
        n_kept += 1
    print(f"  TASK  {subset[:55]:<55} {n_in:>7,} -> {n_kept:>6,}")

print(f"\nChat pool: {len(chat_pool):,}")
total_task = sum(len(v) for v in task_pool_per_subset.values())
print(f"Task pool: {total_task:,}")

# ---------------------------------------------------------------------------
# Sample CHAT set: 10k with length mix
# ---------------------------------------------------------------------------

print("\n=== Sampling CHAT set (10,000) ===")
TARGET = 10000

# Bucket by length
chat_by_bucket = defaultdict(list)
for rec, h in chat_pool:
    chat_by_bucket[rec["length_bucket"]].append((rec, h))

print(f"  Available: short={len(chat_by_bucket['short']):,} "
      f"med={len(chat_by_bucket['med']):,} long={len(chat_by_bucket['long']):,}")

# Target: 10% short, 35% med, 55% long (matches available distribution but balanced)
targets = {"short": 1000, "med": 3500, "long": 5500}
chat_sampled = []
for bucket, target in targets.items():
    avail = chat_by_bucket[bucket]
    n_take = min(target, len(avail))
    sampled = random.sample(avail, n_take)
    chat_sampled.extend(sampled)
    print(f"  {bucket}: target={target} took={n_take}")

# If short fell, top up from long
deficit = TARGET - len(chat_sampled)
if deficit > 0:
    remaining_long = [x for x in chat_by_bucket["long"] if x not in chat_sampled]
    extra = random.sample(remaining_long, min(deficit, len(remaining_long)))
    chat_sampled.extend(extra)
    print(f"  topped up with {len(extra)} long examples")

random.shuffle(chat_sampled)
chat_records = [rec for rec, h in chat_sampled]
chat_hashes = {h for rec, h in chat_sampled}
print(f"  Final chat set: {len(chat_records):,}")

# ---------------------------------------------------------------------------
# Sample TASK set: 10k, stratified across subsets, disjoint from chat set
# ---------------------------------------------------------------------------

print("\n=== Sampling TASK set (10,000) ===")
# Per-subset quotas (sum = 10000), favoring higher-quality sources
QUOTAS = {
    "tulu_3_sft_personas_instruction_following_no_think": 2000,  # Tulu 3 quality
    "smoltalk_smollm3_smol_rewrite_no_think": 1500,
    "smoltalk_smollm3_smol_summarize_no_think": 2000,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 1500,
    "OpenHermes_2.5_no_think": 3000,                              # broad coverage
}
assert sum(QUOTAS.values()) == 10000

task_sampled = []
for subset, quota in QUOTAS.items():
    pool = [x for x in task_pool_per_subset[subset] if x[1] not in chat_hashes]
    if len(pool) < quota:
        print(f"  WARNING {subset}: pool {len(pool)} < quota {quota}")
    n_take = min(quota, len(pool))
    # Within each subset, stratify by length bucket too
    by_bucket = defaultdict(list)
    for rec, h in pool:
        by_bucket[rec["length_bucket"]].append((rec, h))
    target_long = int(n_take * 0.35)
    target_med = int(n_take * 0.50)
    target_short = n_take - target_long - target_med
    sub_sampled = []
    for bucket, t in [("short", target_short), ("med", target_med), ("long", target_long)]:
        avail = by_bucket[bucket]
        sub_sampled.extend(random.sample(avail, min(t, len(avail))))
    # Top up if any bucket short
    deficit = n_take - len(sub_sampled)
    if deficit > 0:
        remaining = [x for x in pool if x not in sub_sampled]
        sub_sampled.extend(random.sample(remaining, min(deficit, len(remaining))))
    task_sampled.extend(sub_sampled)
    short_n = sum(1 for r, _ in sub_sampled if r["length_bucket"] == "short")
    med_n = sum(1 for r, _ in sub_sampled if r["length_bucket"] == "med")
    long_n = sum(1 for r, _ in sub_sampled if r["length_bucket"] == "long")
    print(f"  {subset[:50]:<50} {len(sub_sampled):>5}  s={short_n} m={med_n} l={long_n}")

random.shuffle(task_sampled)
task_records = [rec for rec, h in task_sampled]
task_hashes = {h for rec, h in task_sampled}

# Verify disjoint
assert chat_hashes.isdisjoint(task_hashes), "DISJOINT VIOLATION"
print(f"  Final task set: {len(task_records):,}")
print(f"  Disjoint check: OK")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {path}  ({os.path.getsize(path)/1024/1024:.1f} MB)")

print("\n=== Saving ===")
save_jsonl(chat_records, f"{OUT}/set1_chat_en.jsonl")
save_jsonl(task_records, f"{OUT}/set2_tasks_en.jsonl")

# Stats summary
def summarize(records, name):
    chars = [r["n_chars"] for r in records]
    turns = [r["n_turns"] for r in records]
    buckets = defaultdict(int)
    sources = defaultdict(int)
    for r in records:
        buckets[r["length_bucket"]] += 1
        sources[r["subset"]] += 1
    print(f"\n=== {name} ===")
    print(f"  N: {len(records):,}")
    print(f"  Chars: min={min(chars)}, avg={sum(chars)/len(chars):.0f}, max={max(chars)}")
    print(f"  Turns: min={min(turns)}, avg={sum(turns)/len(turns):.1f}, max={max(turns)}")
    print(f"  Length: short={buckets['short']:,} med={buckets['med']:,} long={buckets['long']:,}")
    print(f"  Sources:")
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s[:55]:<55} {n:>6,}")

summarize(chat_records, "Set 1: CHAT")
summarize(task_records, "Set 2: TASKS")
