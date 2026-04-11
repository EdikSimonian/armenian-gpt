"""
Level 1: Character-Level Tokenizer

The simplest possible tokenizer — each character is one token.
For Armenian, this gives us ~80-100 tokens (Armenian letters + punctuation + space).

How it works:
    "Բdelays" -> [12, 33, 45, ...]   (encode: text to numbers)
    [12, 33, 45, ...] -> "Բdelays"   (decode: numbers back to text)

Stage 2 adds special tokens like <|user|> and <|assistant|> that map to
single token IDs even though they're multiple characters.
"""

import json


class CharTokenizer:
    """Maps each unique character to an integer and back."""

    def __init__(self):
        self.stoi = {}  # string (char or special token) to integer
        self.itos = []  # integer to string (char or special token)
        self.special_tokens = []  # list of multi-char special tokens

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

    def add_special_tokens(self, tokens):
        """
        Add multi-character special tokens (e.g. '<|user|>', '<|assistant|>').
        Each special token gets a single new integer ID.
        """
        for token in tokens:
            if token not in self.stoi:
                idx = len(self.itos)
                self.stoi[token] = idx
                self.itos.append(token)
                self.special_tokens.append(token)
        # Sort special tokens longest-first so encoding matches greedily
        self.special_tokens.sort(key=len, reverse=True)
        return self

    def encode(self, text):
        """Convert text to a list of integer token IDs."""
        if not self.special_tokens:
            # Fast path: no special tokens, pure character-level
            return [self.stoi[ch] for ch in text if ch in self.stoi]

        # With special tokens: scan for them before falling back to char-by-char
        ids = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for token in self.special_tokens:
                if text[i:i+len(token)] == token:
                    ids.append(self.stoi[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self.stoi:
                    ids.append(self.stoi[ch])
                i += 1
        return ids

    def decode(self, ids):
        """Convert a list of integer token IDs back to text."""
        return "".join(self.itos[i] for i in ids if i < len(self.itos))

    def save(self, path):
        """Save the vocabulary to a JSON file."""
        data = {
            "type": "char",
            "itos": self.itos,
            "stoi": self.stoi,
            "special_tokens": self.special_tokens,
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
        tok.special_tokens = data.get("special_tokens", [])
        return tok
