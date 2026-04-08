"""
Translate a small RANDOM SAMPLE (default 100) of a JSONL set with Claude API
for quality comparison against the bulk NLLB translation.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python translate_claude_sample.py set1_chat_en.jsonl set1_chat_claude_sample.jsonl 100

Notes:
    - Costs ~$0.50–$2 for 100 conversations depending on length and model.
    - Uses claude-haiku-4-5 by default (cheap, fast, high quality for translation).
    - Sampling is REPRODUCIBLE (seed 42), so the same indices are picked every run.
    - Output JSONL: same schema as input + `messages_claude` (translated) and `_idx`
      (so you can pair with NLLB output by index).
"""

import json
import os
import random
import sys
import time

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"
SEED = 42

SYS = (
    "You are a professional English-to-Armenian translator. Translate the user's "
    "text into modern Eastern Armenian (Armenian script, hye_Armn). Output ONLY "
    "the Armenian translation — no explanation, no English, no quotes around it. "
    "Preserve meaning, tone, and any technical terms. If the input is very short, "
    "produce a short Armenian translation."
)


def translate(client: Anthropic, text: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYS,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()


def main():
    if len(sys.argv) not in (3, 4):
        print("usage: translate_claude_sample.py INPUT.jsonl OUTPUT.jsonl [N=100]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    n_sample = int(sys.argv[3]) if len(sys.argv) == 4 else 100

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY env var first.")
        sys.exit(1)
    client = Anthropic()

    # Load all rows, fixed-seed sample N indices
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r["_idx"] = i
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {in_path}")

    rng = random.Random(SEED)
    sample_idxs = sorted(rng.sample(range(len(rows)), min(n_sample, len(rows))))
    print(f"Sampling {len(sample_idxs)} rows with seed={SEED} for Claude translation")

    # Resume support
    done_idxs = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_idxs.add(r["_idx"])
                except Exception:
                    pass
        print(f"[resume] {len(done_idxs)} sample rows already done")

    out_f = open(out_path, "a", encoding="utf-8")
    t_start = time.time()
    n_done = 0

    for idx in sample_idxs:
        if idx in done_idxs:
            continue
        row = rows[idx]
        translated_msgs = []
        for msg in row["messages"]:
            try:
                arm = translate(client, msg["content"])
            except Exception as e:
                print(f"  [error idx={idx}] {e}", flush=True)
                arm = f"[ERROR: {e}]"
            translated_msgs.append({"role": msg["role"], "content": arm})

        out_row = dict(row)
        out_row["messages_en"] = row["messages"]
        out_row["messages_claude"] = translated_msgs
        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1

        if n_done % 10 == 0:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            print(f"  [{n_done}/{len(sample_idxs) - len(done_idxs)}] {rate:.2f} rows/s")

    out_f.close()
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. Output: {out_path}")
    print(f"Pair with NLLB output via the `_idx` field for side-by-side comparison.")


if __name__ == "__main__":
    main()
