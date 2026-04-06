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
        self._special_token_to_id = {}  # e.g. {"<|user|>": 8000}
        self._id_to_special_token = {}  # e.g. {8000: "<|user|>"}

    @property
    def vocab_size(self):
        base = self.sp.get_piece_size() if self.sp is not None else self._vocab_size
        return base + len(self._special_token_to_id)

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
            input_sentence_size=1_000_000,  # sample 1M sentences for large files
            shuffle_input_sentence=True,
        )
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        print(f"BPE tokenizer trained! Vocab size: {self.vocab_size}")

    def add_special_tokens(self, tokens):
        """
        Register multi-character special tokens.
        SentencePiece doesn't natively add tokens after training, so we
        map them to IDs beyond the existing vocab.
        """
        for token in tokens:
            if token not in self._special_token_to_id:
                idx = self.vocab_size
                self._special_token_to_id[token] = idx
                self._id_to_special_token[idx] = token
        return self

    def encode(self, text):
        """Convert text to a list of integer token IDs."""
        if not self._special_token_to_id:
            return self.sp.encode(text)

        # Split text around special tokens, encode each segment, insert special IDs
        import re
        pattern = re.compile("(" + "|".join(re.escape(t) for t in self._special_token_to_id) + ")")
        parts = pattern.split(text)
        ids = []
        for part in parts:
            if part in self._special_token_to_id:
                ids.append(self._special_token_to_id[part])
            elif part:
                ids.extend(self.sp.encode(part))
        return ids

    def decode(self, ids):
        """Convert a list of integer token IDs back to text."""
        # Filter out unk tokens (id 0) to avoid ⁇ in output
        unk_id = self.sp.unk_id() if self.sp else 0
        ids = [i for i in ids if i != unk_id]
        # Decode in segments, replacing special token IDs with their strings
        result = []
        sp_ids = []
        for i in ids:
            if i in self._id_to_special_token:
                if sp_ids:
                    result.append(self.sp.decode(sp_ids))
                    sp_ids = []
                result.append(self._id_to_special_token[i])
            else:
                sp_ids.append(i)
        if sp_ids:
            result.append(self.sp.decode(sp_ids))
        return "".join(result)

    def save(self, path):
        """Save tokenizer metadata (the .model file is saved during training)."""
        data = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "model_file": self.sp.serialized_model_proto().hex()
            if self.sp else None,
            "special_tokens": self._special_token_to_id,
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
        if data.get("model_file"):
            tok.sp = spm.SentencePieceProcessor()
            tok.sp.load_from_serialized_proto(bytes.fromhex(data["model_file"]))
        else:
            tok._vocab_size = data["vocab_size"]

        if data.get("special_tokens"):
            tok._special_token_to_id = {k: int(v) for k, v in data["special_tokens"].items()}
            tok._id_to_special_token = {v: k for k, v in tok._special_token_to_id.items()}

        return tok
