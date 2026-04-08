"""
Bulk-translate a SmolTalk2 JSONL set from English to Armenian using NLLB-200-3.3B.

Usage:
    python translate_nllb.py set1_chat_en.jsonl set1_chat_arm.jsonl
    python translate_nllb.py set2_tasks_en.jsonl set2_tasks_arm.jsonl

Notes:
    - Loads facebook/nllb-200-3.3B onto CUDA. ~7 GB VRAM at fp16.
    - DO NOT run while training is using the GPU; wait for training to finish.
    - Batches sentences across messages for throughput. Each `messages[i].content`
      is split into sentences, translated in batches, then re-joined.
    - Skips translation if a `*_arm` file already exists for that input (resumable).
    - Output JSONL has same schema as input plus `messages_en` (originals) and a
      new `messages` field with Armenian content.
    - Estimated wall time on A6000: ~1-2 hours per 10k set.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-3.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "hye_Armn"   # Eastern Armenian (Modern), Armenian script
BATCH_SIZE = 16
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 512
NUM_BEAMS = 4

# Sentence splitter — simple, language-agnostic. Splits on . ! ? : followed by space + capital.
SENT_SPLIT = re.compile(r'(?<=[.!?:])\s+(?=[A-Z\u0531-\u0556])')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for batched translation. Preserves order."""
    text = text.strip()
    if len(text) <= MAX_INPUT_LEN * 2:
        return [text]
    parts = SENT_SPLIT.split(text)
    # Merge tiny fragments
    out, buf = [], ""
    for p in parts:
        if len(buf) + len(p) + 1 < MAX_INPUT_LEN * 2:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: translate_nllb.py INPUT.jsonl OUTPUT.jsonl")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}")
        sys.exit(1)

    # Resume support: if output exists, skip already-translated rows
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add(r.get("_idx"))
                except Exception:
                    pass
        print(f"[resume] {len(done_keys)} rows already in {out_path}")

    print(f"Loading {MODEL_NAME} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    forced_bos = tok.convert_tokens_to_ids(TGT_LANG)
    print(f"  loaded in {time.time()-t0:.1f}s, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    @torch.inference_mode()
    def translate_batch(sentences: list[str]) -> list[str]:
        if not sentences:
            return []
        enc = tok(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to("cuda")
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos,
            max_length=MAX_OUTPUT_LEN,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return tok.batch_decode(out, skip_special_tokens=True)

    # Load all input rows
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r["_idx"] = i
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {in_path}")

    pending = [r for r in rows if r["_idx"] not in done_keys]
    print(f"Translating {len(pending)} remaining rows...")

    out_f = open(out_path, "a", encoding="utf-8")
    t_start = time.time()

    for i, row in enumerate(tqdm(pending, desc="rows", smoothing=0.1)):
        translated_msgs = []
        for msg in row["messages"]:
            content_en = msg["content"]
            sentences = split_sentences(content_en)
            # Batch sentences across this message
            translated_pieces = []
            for j in range(0, len(sentences), BATCH_SIZE):
                chunk = sentences[j:j + BATCH_SIZE]
                translated_pieces.extend(translate_batch(chunk))
            translated_msgs.append({"role": msg["role"], "content": " ".join(translated_pieces)})

        out_row = dict(row)
        out_row["messages_en"] = row["messages"]
        out_row["messages"] = translated_msgs
        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_f.flush()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_s = (len(pending) - i - 1) / rate
            print(f"  [{i+1}/{len(pending)}] {rate:.2f} rows/s, eta {eta_s/60:.1f} min")

    out_f.close()
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. Output: {out_path}")


if __name__ == "__main__":
    main()
