"""
Stage 2: Prepare Conversational Data for Fine-tuning

Formats Q&A data with special chat tokens for fine-tuning.

Usage:
    # Use hand-crafted Armenian Q&A (recommended):
    python data/prepare_chat.py --source data/armenian_qa.json

    # Use Alpaca-Armenian from HuggingFace (lower quality):
    python data/prepare_chat.py

After running, you'll have:
    data_chat/train_{char|bpe}.bin       - training tokens (90%)
    data_chat/val_{char|bpe}.bin         - validation tokens (10%)
    data_chat/train_mask_{char|bpe}.bin  - loss mask, uint8 (1=response, 0=prompt)
    data_chat/val_mask_{char|bpe}.bin    - loss mask for the val split
    data_chat/tokenizer_{char|bpe}.json  - extended vocabulary with chat tokens

Only response tokens (mask 1) are trained on; 6_finetune.py sets prompt-token
targets to -100 so cross_entropy ignores them.

The tokenizer type is auto-detected from data/ unless --tokenizer is passed.
"""

import os
import sys
import json
import argparse
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

DATA_DIR = os.path.join(_REPO_ROOT, "data")
CHAT_DIR = os.path.join(_REPO_ROOT, "data_chat")

# Special tokens for chat formatting
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

# Words to filter out for age-appropriate content (ages 12-18)
# This is a basic blocklist — extend as needed
BLOCKLIST = [
    # English terms that may appear in translated data
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
    "hack someone",
    "steal",
]


def is_appropriate(example):
    """Filter out content not suitable for students ages 12-18."""
    text = (
        example.get("instruction", "")
        + " "
        + example.get("input", "")
        + " "
        + example.get("output", "")
    ).lower()
    for word in BLOCKLIST:
        if word in text:
            return False
    # Skip very short or empty responses
    if len(example.get("output", "")) < 10:
        return False
    return True


def format_chat(example):
    """
    Format an Alpaca example as a chat conversation.

    Input:  {instruction: "...", input: "...", output: "..."}
    Output: "<|user|>instruction input<|end|><|assistant|>output<|end|>"
    """
    instruction = example["instruction"].strip()
    inp = example.get("input", "").strip()
    output = example["output"].strip()

    # Combine instruction and input
    if inp:
        user_msg = f"{instruction}\n{inp}"
    else:
        user_msg = instruction

    return f"{USER_TOKEN}{user_msg}{END_TOKEN}{ASSISTANT_TOKEN}{output}{END_TOKEN}"


def load_local_json(path: str) -> list:
    """Load pre-generated Q&A pairs from a local JSON file."""
    print(f"Loading Q&A from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"  Loaded {len(examples):,} examples")
    return examples


def prepare_chat_data(source_path, tokenizer_type):
    """Build data_chat/ bins from a local SFT JSON using Stage 1 tokenizer.

    source_path: path to a {instruction, input?, output} JSON file.
    tokenizer_type: "char" or "bpe" — must match what was used in 3_tokenize.py.
    """
    from core import (
        bin_paths,
        load_tokenizer as _load_tokenizer,
        tokenizer_path,
    )

    os.makedirs(CHAT_DIR, exist_ok=True)

    all_examples = load_local_json(source_path)

    print("Filtering inappropriate content...")
    filtered = [ex for ex in all_examples if is_appropriate(ex)]
    print(
        f"  After filtering: {len(filtered):,} examples "
        f"({len(all_examples) - len(filtered)} removed)"
    )

    # Load Stage 1 tokenizer and extend it with the chat special tokens BEFORE
    # encoding. The special tokens act as hard segment boundaries in the BPE
    # encoder, so encoding the prompt and response separately yields exactly
    # the same ids as encoding the whole example — which lets us build a
    # token-aligned loss mask with no boundary ambiguity.
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

    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN])
    print(
        f"Extended vocabulary: {tokenizer.vocab_size} tokens "
        f"(+{tokenizer.vocab_size - old_vocab_size} special)"
    )

    # Sanitize: strip any LITERAL chat control-token strings appearing inside
    # the dataset fields, so they can't be encoded as real <|user|>/<|assistant|>/
    # <|end|> control tokens and corrupt the prompt/response boundary + mask.
    def _strip_special(s):
        for t in (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN):
            s = s.replace(t, " ")
        return s

    # Each example -> <|user|>{msg}<|end|><|assistant|>{output}<|end|>, split
    # into a PROMPT part (up to and including <|assistant|>) and a RESPONSE
    # part ({output}<|end|>). Only response tokens are trained on (mask 1);
    # prompt tokens get mask 0 so cross_entropy(ignore_index=-100) skips them.
    # This concentrates the SFT signal on *answering* instead of also learning
    # to echo the questions.
    print("Formatting as prompt-masked chat conversations...")
    prompts, responses = [], []
    n_sanitized = 0
    for ex in filtered:
        raw_instr = ex["instruction"].strip()
        raw_inp = ex.get("input", "").strip()
        raw_out = ex["output"].strip()
        instruction = _strip_special(raw_instr)
        inp = _strip_special(raw_inp)
        output = _strip_special(raw_out)
        if instruction != raw_instr or inp != raw_inp or output != raw_out:
            n_sanitized += 1
        user_msg = f"{instruction}\n{inp}" if inp else instruction
        prompts.append(f"{USER_TOKEN}{user_msg}{END_TOKEN}{ASSISTANT_TOKEN}")
        responses.append(f"{output}{END_TOKEN}")
    if n_sanitized:
        print(f"  Sanitized literal control tokens out of {n_sanitized} example(s)")

    # Char tokenizer: register any unseen characters before encoding.
    if tokenizer_type == "char":
        full_text = "".join(p + r for p, r in zip(prompts, responses))
        new_chars = sorted(
            {ch for ch in full_text if ch not in tokenizer.stoi and len(ch) == 1}
        )
        if new_chars:
            for ch in new_chars:
                tokenizer.stoi[ch] = len(tokenizer.itos)
                tokenizer.itos.append(ch)
            print(f"  Added {len(new_chars)} new characters from chat data")
            print(f"  Final vocabulary: {tokenizer.vocab_size} tokens")

    # Split at the EXAMPLE level (not the flat token stream) so no single
    # example is bisected across train/val — a flat cut would put an answer's
    # prefix in train and its suffix in val, muddying val loss. Shuffle with a
    # fixed seed first so the val split is representative regardless of the
    # order sources were merged in.
    order = np.random.default_rng(1234).permutation(len(prompts))
    n_train = int(len(order) * 0.9)
    train_idx, val_idx = order[:n_train], order[n_train:]

    def _encode_split(indices):
        ids_list, mask_list = [], []
        for j in indices:
            p_ids = tokenizer.encode(prompts[j])
            r_ids = tokenizer.encode(responses[j])
            ids_list.extend(p_ids)
            ids_list.extend(r_ids)
            mask_list.extend([0] * len(p_ids))  # prompt: loss ignored
            mask_list.extend([1] * len(r_ids))  # response: trained on
        return (
            np.array(ids_list, dtype=np.uint16),
            np.array(mask_list, dtype=np.uint8),
        )

    print("Encoding text + building loss mask (example-level 90/10 split)...")
    train_ids, train_mask = _encode_split(train_idx)
    val_ids, val_mask = _encode_split(val_idx)
    total = len(train_ids) + len(val_ids)
    n_resp = int(train_mask.sum()) + int(val_mask.sum())
    print(f"  Total tokens: {total:,}")
    print(
        f"  Response (trained) tokens: {n_resp:,} ({100 * n_resp / max(total, 1):.1f}%)"
    )

    train_path, val_path = bin_paths(CHAT_DIR, tokenizer_type)
    train_mask_path = os.path.join(CHAT_DIR, f"train_mask_{tokenizer_type}.bin")
    val_mask_path = os.path.join(CHAT_DIR, f"val_mask_{tokenizer_type}.bin")
    tok_path = tokenizer_path(CHAT_DIR, tokenizer_type)

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    train_mask.tofile(train_mask_path)
    val_mask.tofile(val_mask_path)
    tokenizer.save(tok_path)

    print(f"\n{'=' * 50}")
    print("  Chat Data Preparation Complete!")
    print(f"{'=' * 50}")
    print(f"  Examples:     {len(filtered):,}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")
    print(f"  Train tokens: {len(train_ids):,} ({train_path})")
    print(f"  Val tokens:   {len(val_ids):,} ({val_path})")
    print(
        f"  Train resp:   {int(train_mask.sum()):,} trained tokens "
        f"({100 * int(train_mask.sum()) / max(len(train_ids), 1):.1f}%)"
    )
    print(f"  Masks:        {train_mask_path}, {val_mask_path}")
    print(f"  Train size:   {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val size:     {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")
    print(f"  Tokenizer:    {tok_path}")

    return len(filtered), len(train_ids), len(val_ids)


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
