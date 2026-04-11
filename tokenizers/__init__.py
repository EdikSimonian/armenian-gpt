"""Tokenizer package helpers.

Both the char and BPE pipelines write their artifacts side-by-side using a
tokenizer-type suffix:

    data/train_char.bin      data/train_bpe.bin
    data/val_char.bin        data/val_bpe.bin
    data/tokenizer_char.json data/tokenizer_bpe.json

The helpers here centralize path resolution and loading so callers never need
to hand-build these filenames.
"""

import os


def tokenizer_path(data_dir, tokenizer_type):
    return os.path.join(data_dir, f"tokenizer_{tokenizer_type}.json")


def bin_paths(data_dir, tokenizer_type):
    return (
        os.path.join(data_dir, f"train_{tokenizer_type}.bin"),
        os.path.join(data_dir, f"val_{tokenizer_type}.bin"),
    )


def detect_tokenizer_type(data_dir):
    """Return the tokenizer type (char or bpe) present in data_dir.

    Raises FileNotFoundError if neither is present, or ValueError if both are
    present (caller must disambiguate via --tokenizer).
    """
    has_char = os.path.exists(tokenizer_path(data_dir, "char"))
    has_bpe = os.path.exists(tokenizer_path(data_dir, "bpe"))
    if has_char and has_bpe:
        raise ValueError(
            f"Both tokenizer_char.json and tokenizer_bpe.json exist in {data_dir}. "
            "Pass --tokenizer char|bpe to choose."
        )
    if has_char:
        return "char"
    if has_bpe:
        return "bpe"
    raise FileNotFoundError(
        f"No tokenizer_char.json or tokenizer_bpe.json in {data_dir}. "
        "Run 3_tokenize.py first."
    )


def load_tokenizer(data_dir, tokenizer_type):
    """Instantiate and load the right tokenizer class for the given type."""
    path = tokenizer_path(data_dir, tokenizer_type)
    if tokenizer_type == "char":
        from tokenizers.char_tokenizer import CharTokenizer
        return CharTokenizer.load(path)
    if tokenizer_type == "bpe":
        from tokenizers.bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(path)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
