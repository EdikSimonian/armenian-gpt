"""
Stage 2: Prepare Conversational Data for Fine-tuning

Formats Q&A data with special chat tokens for fine-tuning, with prompt
loss-masking (only assistant-response tokens are trained on).

Two input shapes are accepted per example:
    - Alpaca single-turn: {instruction, input?, output, category?}
    - Multi-turn:         {messages: [{role: user|assistant, content}, ...]}
Both are normalized to a list of (role, text) turns and encoded identically.

Usage:
    python data/prepare_chat.py --source data/armenian_qa.json

After running, you'll have, under data_chat/:
    train_{char|bpe}.bin       - training tokens (90%)
    val_{char|bpe}.bin         - validation tokens (10%)
    train_mask_{char|bpe}.bin  - loss mask, uint8 (1=response, 0=prompt)
    val_mask_{char|bpe}.bin    - loss mask for the val split
    train_idx_{char|bpe}.bin   - uint64 example-start offsets (for example-
    val_idx_{char|bpe}.bin       aligned packing in 6_finetune.py)
    tokenizer_{char|bpe}.json  - vocabulary (chat tokens are native in v2)

Only response tokens (mask 1) are trained on; 6_finetune.py sets prompt-token
targets to -100 so cross_entropy ignores them. The idx bins let 6_finetune
start every training window at an example boundary, so the model never trains
on an answer-continuation whose prompt has scrolled out of the window.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

DATA_DIR = os.path.join(_REPO_ROOT, "data")
CHAT_DIR = os.path.join(_REPO_ROOT, "data_chat")

# Special tokens for chat formatting
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

# Age-appropriateness blocklist (students 12-18). Matched on WORD BOUNDARIES so
# "sex" no longer trips on "Middlesex" and "drug" not on "drugstore" — the old
# substring match silently dropped legitimate examples.
BLOCKLIST = [
    "kill",
    "murder",
    "suicide",
    "drug",
    "cocaine",
    "heroin",
    "sex",
    "porn",
    "nude",
    "violent",
    "torture",
    "weapon",
    "bomb",
    "terrorist",
    "steal",
]
_BLOCK_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKLIST) + r")\b", re.IGNORECASE
)

# Categories whose whole point is to demonstrate a safe refusal. These MUST
# bypass the blocklist — they contain exactly the words it screens for ("bomb",
# "weapon"), and dropping them is why the model never learned to decline (see
# results/00_SUMMARY.md). An example opts in via category/source containing one
# of these substrings, or an explicit {"keep": true}.
_REFUSAL_MARKERS = ("refus", "safety", "safe", "decline", "մերժ", "անվտանգ")

MIN_OUTPUT_CHARS = 3  # was a hard 10, which dropped valid short answers (Երևան)


def _is_refusal_example(example):
    if example.get("keep") is True:
        return True
    tag = (
        str(example.get("category", "")) + " " + str(example.get("source", ""))
    ).lower()
    return any(m in tag for m in _REFUSAL_MARKERS)


def _example_text(example):
    """All user+assistant text of an example, for the blocklist scan."""
    if "messages" in example:
        return " ".join(m.get("content", "") for m in example["messages"])
    return " ".join(example.get(k, "") for k in ("instruction", "input", "output"))


def is_appropriate(example, min_output_chars=MIN_OUTPUT_CHARS):
    """Filter content not suitable for students 12-18.

    Refusal/safety examples bypass the blocklist (they intentionally contain the
    screened words to teach safe declines). Everything else is matched on word
    boundaries. Very short outputs are dropped EXCEPT for refusals.
    """
    refusal = _is_refusal_example(example)
    if not refusal and _BLOCK_RE.search(_example_text(example)):
        return False
    # Length gate on the final assistant turn.
    out = _final_assistant_text(example)
    if not refusal and len(out) < min_output_chars:
        return False
    return True


def _final_assistant_text(example):
    if "messages" in example:
        for m in reversed(example["messages"]):
            if m.get("role") == "assistant":
                return (m.get("content") or "").strip()
        return ""
    return (example.get("output") or "").strip()


def _strip_special(s):
    """Strip LITERAL chat control-token strings from a field so they can't be
    encoded as real control tokens and corrupt the prompt/response boundary."""
    for t in (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN):
        s = s.replace(t, " ")
    return s


def to_turns(example):
    """Normalize either input shape to a list of (role, text) turns, with chat
    control tokens stripped from every text field. Returns [] if the example has
    no usable assistant turn."""
    turns = []
    if "messages" in example:
        for m in example["messages"]:
            role = m.get("role")
            text = _strip_special((m.get("content") or "").strip())
            if role in ("user", "assistant") and text:
                turns.append((role, text))
    else:
        instruction = _strip_special((example.get("instruction") or "").strip())
        inp = _strip_special((example.get("input") or "").strip())
        output = _strip_special((example.get("output") or "").strip())
        user_msg = f"{instruction}\n{inp}" if inp else instruction
        if user_msg:
            turns.append(("user", user_msg))
        if output:
            turns.append(("assistant", output))
    # Must contain at least one assistant turn to supply a training signal.
    if not any(role == "assistant" for role, _ in turns):
        return []
    return turns


def encode_turns(tokenizer, turns):
    """Encode a turn list into (ids, mask). Assistant CONTENT and its trailing
    <|end|> are masked 1 (trained, so the model learns to answer AND to stop);
    the <|assistant|> tag, all user turns, and their markup are masked 0.

    Each piece is encoded separately. Because the special tokens are hard
    segment boundaries in the encoder, this yields exactly the same ids as
    encoding the whole string — so the mask stays token-aligned.
    """
    ids, mask = [], []

    def add(text, m):
        e = tokenizer.encode(text)
        ids.extend(e)
        mask.extend([m] * len(e))

    for role, text in turns:
        if role == "user":
            add(f"{USER_TOKEN}{text}{END_TOKEN}", 0)
        else:  # assistant
            add(ASSISTANT_TOKEN, 0)  # the tag is part of the prompt
            add(f"{text}{END_TOKEN}", 1)  # content + stop token are trained on
    return ids, mask


def load_local_json(path: str) -> list:
    """Load pre-generated Q&A pairs from a local JSON file."""
    print(f"Loading Q&A from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"  Loaded {len(examples):,} examples")
    return examples


def prepare_chat_data(source_path, tokenizer_type, min_output_chars=MIN_OUTPUT_CHARS):
    """Build data_chat/ bins from a local SFT JSON using the Stage 1 tokenizer.

    source_path: path to a JSON list of Alpaca and/or multi-turn examples.
    tokenizer_type: "char" or "bpe" — must match what was used in 3_tokenize.py.
    """
    from core import (
        bin_paths,
        load_tokenizer as _load_tokenizer,
        tokenizer_path,
    )

    os.makedirs(CHAT_DIR, exist_ok=True)

    all_examples = load_local_json(source_path)

    print("Filtering inappropriate content (refusal/safety rows exempt)...")
    filtered = [ex for ex in all_examples if is_appropriate(ex, min_output_chars)]
    n_refusal = sum(1 for ex in filtered if _is_refusal_example(ex))
    print(
        f"  After filtering: {len(filtered):,} examples "
        f"({len(all_examples) - len(filtered)} removed; {n_refusal} refusal/safety kept)"
    )

    stage1_tok_path = tokenizer_path(DATA_DIR, tokenizer_type)
    if not os.path.exists(stage1_tok_path):
        print(f"\nError: {stage1_tok_path} not found!")
        print("Run Stage 1 data preparation first:")
        print("  python 1_download.py")
        print("  python 2_prepare.py")
        print(f"  python 3_tokenize.py --tokenizer {tokenizer_type}")
        sys.exit(1)

    tokenizer = _load_tokenizer(DATA_DIR, tokenizer_type)
    old_vocab_size = tokenizer.vocab_size
    print(f"\nStage 1 vocabulary: {old_vocab_size} tokens (type: {tokenizer_type})")

    # Register the chat tokens. In v2 they are already native pieces, so this
    # reuses their ids; in v1 they are grafted beyond the base vocab.
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN])
    print(
        f"Extended vocabulary: {tokenizer.vocab_size} tokens "
        f"(+{tokenizer.vocab_size - old_vocab_size} grafted)"
    )

    print("Normalizing to turns and encoding (prompt-masked)...")
    encoded = []  # list of (ids, mask) per usable example
    n_multi = 0
    for ex in filtered:
        turns = to_turns(ex)
        if not turns:
            continue
        if sum(1 for r, _ in turns if r == "assistant") > 1:
            n_multi += 1
        # Char tokenizer: register any unseen characters before encoding.
        if tokenizer_type == "char":
            text = "".join(t for _, t in turns)
            new = sorted({c for c in text if c not in tokenizer.stoi and len(c) == 1})
            for c in new:
                tokenizer.stoi[c] = len(tokenizer.itos)
                tokenizer.itos.append(c)
        ids, mask = encode_turns(tokenizer, turns)
        if any(mask):  # at least one trained token
            encoded.append((ids, mask))
    print(f"  Usable examples: {len(encoded):,} ({n_multi:,} multi-turn)")

    # Example-level shuffle then 90/10 split (no example is bisected; seed fixed
    # so the val split is representative regardless of source merge order).
    order = np.random.default_rng(1234).permutation(len(encoded))
    n_train = int(len(order) * 0.9)
    train_idx_examples, val_idx_examples = order[:n_train], order[n_train:]

    def _build_split(example_indices):
        ids_list, mask_list, offsets = [], [], []
        for j in example_indices:
            offsets.append(len(ids_list))  # example START offset
            ex_ids, ex_mask = encoded[j]
            ids_list.extend(ex_ids)
            mask_list.extend(ex_mask)
        return (
            np.array(ids_list, dtype=np.uint16),
            np.array(mask_list, dtype=np.uint8),
            np.array(offsets, dtype=np.uint64),
        )

    train_ids, train_mask, train_off = _build_split(train_idx_examples)
    val_ids, val_mask, val_off = _build_split(val_idx_examples)
    total = len(train_ids) + len(val_ids)
    n_resp = int(train_mask.sum()) + int(val_mask.sum())
    print(f"  Total tokens: {total:,}")
    print(
        f"  Response (trained) tokens: {n_resp:,} ({100 * n_resp / max(total, 1):.1f}%)"
    )

    train_path, val_path = bin_paths(CHAT_DIR, tokenizer_type)
    train_mask_path = os.path.join(CHAT_DIR, f"train_mask_{tokenizer_type}.bin")
    val_mask_path = os.path.join(CHAT_DIR, f"val_mask_{tokenizer_type}.bin")
    train_idx_path = os.path.join(CHAT_DIR, f"train_idx_{tokenizer_type}.bin")
    val_idx_path = os.path.join(CHAT_DIR, f"val_idx_{tokenizer_type}.bin")
    tok_path = tokenizer_path(CHAT_DIR, tokenizer_type)

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    train_mask.tofile(train_mask_path)
    val_mask.tofile(val_mask_path)
    train_off.tofile(train_idx_path)
    val_off.tofile(val_idx_path)
    tokenizer.save(tok_path)

    print(f"\n{'=' * 50}")
    print("  Chat Data Preparation Complete!")
    print(f"{'=' * 50}")
    print(f"  Examples:     {len(encoded):,}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")
    print(f"  Train tokens: {len(train_ids):,} ({train_path})")
    print(f"  Val tokens:   {len(val_ids):,} ({val_path})")
    print(
        f"  Train resp:   {int(train_mask.sum()):,} trained tokens "
        f"({100 * int(train_mask.sum()) / max(len(train_ids), 1):.1f}%)"
    )
    print(f"  Masks:        {train_mask_path}, {val_mask_path}")
    print(f"  Example idx:  {train_idx_path}, {val_idx_path}")
    print(f"  Tokenizer:    {tok_path}")

    return len(encoded), len(train_ids), len(val_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to pre-generated JSON file (e.g. data/qa_merged.json).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        choices=["char", "bpe"],
        help="Stage 1 tokenizer type. If omitted, auto-detects from data/.",
    )
    args = parser.parse_args()

    from core import detect_tokenizer_type

    if args.tokenizer:
        tok_type = args.tokenizer
    else:
        try:
            tok_type = detect_tokenizer_type(DATA_DIR)
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            sys.exit(1)

    source_path = args.source
    if not os.path.isabs(source_path):
        source_path = os.path.join(os.path.dirname(DATA_DIR), source_path)

    prepare_chat_data(source_path, tok_type)


if __name__ == "__main__":
    main()
