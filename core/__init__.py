"""Shared runtime modules imported by the numbered pipeline steps.

Contents:
    core.model            — GPT architecture (RMSNorm, RoPE, SwiGLU)
    core.config           — training presets and argparse-backed get_config()
    core.char_tokenizer   — character-level tokenizer
    core.bpe_tokenizer    — SentencePiece BPE tokenizer

And the path/resolver helpers re-exported here at package level:
    tokenizer_path(data_dir, type)        → data_dir/tokenizer_{type}.json
    bin_paths(data_dir, type)             → (train_{type}.bin, val_{type}.bin)
    detect_tokenizer_type(data_dir)       → "char" or "bpe"
    load_tokenizer(data_dir, type)        → instantiated tokenizer

Both the char and BPE pipelines write their artifacts side-by-side using a
tokenizer-type suffix:

    data/train_char.bin      data/train_bpe.bin
    data/val_char.bin        data/val_bpe.bin
    data/tokenizer_char.json data/tokenizer_bpe.json

The helpers centralize path resolution so callers never hand-build filenames.
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
        from .char_tokenizer import CharTokenizer
        return CharTokenizer.load(path)
    if tokenizer_type == "bpe":
        from .bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(path)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
