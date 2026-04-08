"""
Side-by-side compare NLLB and Claude translations for the same `_idx` rows,
then have Claude grade each NLLB translation against the Claude reference.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python compare_nllb_vs_claude.py \
        set1_chat_arm.jsonl set1_chat_claude_sample.jsonl set1_chat_grades.jsonl

Outputs:
    - JSONL with per-message grades (1-5 scale + reasoning)
    - Aggregate stats: mean grade, % of NLLB messages "good enough" (≥4)
"""

import json
import os
import sys
import statistics

from anthropic import Anthropic

GRADE_MODEL = "claude-haiku-4-5-20251001"
SCALE = 5

GRADING_PROMPT = """You are evaluating an automatic translation of English text to Eastern Armenian.

ENGLISH SOURCE:
{en}

REFERENCE ARMENIAN (Claude, treated as gold):
{ref}

NLLB-200 ARMENIAN (under evaluation):
{nllb}

Grade the NLLB translation on a 1-5 scale:
5 = Native-quality, fully accurate, fluent
4 = Minor errors, fully understandable, suitable for fine-tuning
3 = Some grammar/word choice errors, mostly understandable
2 = Significant errors, partially understandable
1 = Broken, incoherent, or English passthrough

Reply with JSON only: {{"grade": N, "reason": "one-sentence reason"}}"""


def main():
    if len(sys.argv) != 4:
        print("usage: compare_nllb_vs_claude.py NLLB_ARM.jsonl CLAUDE_SAMPLE.jsonl OUT_GRADES.jsonl")
        sys.exit(1)
    nllb_path, claude_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY env var first.")
        sys.exit(1)
    client = Anthropic()

    # Load both files indexed by _idx
    def load_indexed(path):
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[r["_idx"]] = r
        return d

    nllb = load_indexed(nllb_path)
    claude = load_indexed(claude_path)
    common = sorted(set(nllb.keys()) & set(claude.keys()))
    print(f"NLLB rows: {len(nllb)}, Claude rows: {len(claude)}, common: {len(common)}")

    # Resume
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["_idx"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} already graded")

    out_f = open(out_path, "a", encoding="utf-8")
    all_grades = []

    for idx in common:
        if idx in done:
            continue
        nllb_msgs = nllb[idx]["messages"]
        claude_msgs = claude[idx]["messages_claude"]
        en_msgs = claude[idx]["messages_en"]
        # Grade per message (only the assistant turns to keep cost down)
        msg_grades = []
        for i, m in enumerate(en_msgs):
            if m["role"] != "assistant":
                continue
            if i >= len(nllb_msgs) or i >= len(claude_msgs):
                continue
            try:
                resp = client.messages.create(
                    model=GRADE_MODEL,
                    max_tokens=200,
                    messages=[{"role": "user", "content": GRADING_PROMPT.format(
                        en=m["content"][:1500],
                        ref=claude_msgs[i]["content"][:1500],
                        nllb=nllb_msgs[i]["content"][:1500],
                    )}],
                )
                txt = resp.content[0].text.strip()
                # Extract JSON
                if txt.startswith("```"):
                    txt = txt.split("```")[1].lstrip("json\n")
                grade_data = json.loads(txt)
                msg_grades.append(grade_data)
                all_grades.append(grade_data["grade"])
            except Exception as e:
                msg_grades.append({"grade": None, "reason": f"error: {e}"})

        out_row = {"_idx": idx, "msg_grades": msg_grades}
        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_f.flush()

        if len(all_grades) and len(all_grades) % 20 == 0:
            print(f"  graded {len(all_grades)} msgs, mean={statistics.mean(all_grades):.2f}")

    out_f.close()

    if all_grades:
        print(f"\n=== Aggregate ===")
        print(f"  Total graded messages: {len(all_grades)}")
        print(f"  Mean grade: {statistics.mean(all_grades):.2f} / 5")
        print(f"  Median:     {statistics.median(all_grades)}")
        print(f"  % >= 4 (good enough for FT): "
              f"{100 * sum(1 for g in all_grades if g >= 4) / len(all_grades):.1f}%")
        print(f"  % >= 3:                       "
              f"{100 * sum(1 for g in all_grades if g >= 3) / len(all_grades):.1f}%")
        from collections import Counter
        dist = Counter(all_grades)
        for g in sorted(dist, reverse=True):
            print(f"  grade {g}: {dist[g]} ({100*dist[g]/len(all_grades):.0f}%)")


if __name__ == "__main__":
    main()
