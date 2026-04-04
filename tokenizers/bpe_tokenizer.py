"""
Level 2 (Advanced): BPE Tokenizer using SentencePiece

Instead of one token per character, BPE groups common character sequences
into "subwords". For example, the common Armenian word "Հայաստան" might
become just 1-2 tokens instead of 8 characters.

This gives better results but requires an extra library:
    pip install sentencepiece

How it works:
    1. Train: learn common character groups from Armenian text
    2. Encode: split text into subword tokens
    3. Decode: join subword tokens back into text
"""

import os
import json


class BPETokenizer:
    """Subword tokenizer using SentencePiece BPE."""

    def __init__(self):
        self.sp = None  # SentencePiece processor
        self._vocab_size = 0

    @property
    def vocab_size(self):
        if self.sp is not None:
            return self.sp.get_piece_size()
        return self._vocab_size

    def train(self, text_file, model_prefix="data/bpe_model", vocab_size=8000):
        """
        Train a BPE model on Armenian text.

        Args:
            text_file: path to a .txt file with training text
            model_prefix: where to save the model (creates .model and .vocab files)
            vocab_size: number of subword tokens to learn (8000 is good for Armenian)
        """
        try:
            import sentencepiece as spm
        except ImportError:
            print("Error: sentencepiece not installed!")
            print("Install it with: pip install sentencepiece")
            raise

        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9999,  # cover almost all Armenian characters
            normalization_rule_name="nfkc",
            pad_id=3,
        )
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        print(f"BPE tokenizer trained! Vocab size: {self.vocab_size}")

    def encode(self, text):
        """Convert text to a list of integer token IDs."""
        return self.sp.encode(text)

    def decode(self, ids):
        """Convert a list of integer token IDs back to text."""
        return self.sp.decode(ids)

    def save(self, path):
        """Save tokenizer metadata (the .model file is saved during training)."""
        data = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "model_file": self.sp.serialized_model_proto().hex()
            if self.sp else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load BPE tokenizer from saved metadata."""
        try:
            import sentencepiece as spm
        except ImportError:
            print("Error: sentencepiece not installed!")
            print("Install it with: pip install sentencepiece")
            raise

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls()
        tok._vocab_size = data["vocab_size"]

        if data.get("model_file"):
            tok.sp = spm.SentencePieceProcessor()
            tok.sp.load_from_serialized_proto(bytes.fromhex(data["model_file"]))

        return tok
