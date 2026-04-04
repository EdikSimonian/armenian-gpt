"""
Level 1: Character-Level Tokenizer

The simplest possible tokenizer — each character is one token.
For Armenian, this gives us ~80-100 tokens (Armenian letters + punctuation + space).

How it works:
    "Բարdelays" -> [12, 33, 45, ...]   (encode: text to numbers)
    [12, 33, 45, ...] -> "Բարdelays"   (decode: numbers back to text)
"""

import json


class CharTokenizer:
    """Maps each unique character to an integer and back."""

    def __init__(self):
        self.stoi = {}  # string (char) to integer
        self.itos = []  # integer to string (char)

    @property
    def vocab_size(self):
        return len(self.itos)

    def build_vocab(self, text):
        """Scan text and create the character vocabulary."""
        # Get all unique characters, sorted for reproducibility
        chars = sorted(set(text))
        self.itos = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        return self

    def encode(self, text):
        """Convert text to a list of integer token IDs."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        """Convert a list of integer token IDs back to text."""
        return "".join(self.itos[i] for i in ids if i < len(self.itos))

    def save(self, path):
        """Save the vocabulary to a JSON file."""
        data = {
            "type": "char",
            "itos": self.itos,
            "stoi": self.stoi,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Load a vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.itos = data["itos"]
        tok.stoi = data["stoi"]
        return tok
