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

v2 training config (see TRAINING_REVIEW.md):
    - normalization_rule_name="identity": unicode normalization happens ONCE,
      as NFC in 2_prepare.py. The old "nfkc" rule decomposed և (U+0587) into
      եւ, so every model generation wrote non-reform orthography.
    - user_defined_symbols includes "\\n" and the chat control tokens: newline
      becomes a real vocab piece (the old config encoded \\n\\n as UNK — the
      model's only paragraph separator was the unknown-token, and decode()
      stripped it). Chat specials being native removes the 16000→16003 vocab
      graft at SFT time and its trail of vocab-mismatch bugs.
    - byte_fallback=True: characters outside the vocab decompose into byte
      pieces instead of collapsing to UNK, so rare symbols degrade gracefully.

Tokenizer JSONs saved by the old (v1) config still load and behave exactly as
before — grafted special tokens keep their beyond-vocab ids.
"""

import json
import re
import unicodedata

# Special pieces baked into newly trained (v2) models. Order matters only for
# readability; SentencePiece assigns their ids after the control symbols.
V2_USER_DEFINED_SYMBOLS = ["\n", "<|user|>", "<|assistant|>", "<|end|>"]


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
        return base + self._n_grafted(base)

    def _n_grafted(self, base):
        """Count special tokens that live BEYOND the SentencePiece vocab.

        v1 tokenizers grafted specials at ids >= piece_size (extending the
        vocab); v2 models have them as native pieces (ids < piece_size), which
        must not be double-counted.
        """
        return sum(1 for i in self._special_token_to_id.values() if i >= base)

    @property
    def eos_id(self):
        """SentencePiece </s> id (2 by convention) — used as the document
        separator in the pretraining stream. None for metadata-only loads."""
        if self.sp is None:
            return None
        eos = self.sp.eos_id()
        return eos if eos >= 0 else None

    @property
    def newline_id(self):
        """Id of the "\\n" piece in v2 models, else None (v1 models)."""
        if self.sp is None:
            return None
        nl = self.sp.piece_to_id("\n")
        return nl if nl != self.sp.unk_id() else None

    def train(self, text_file, model_prefix="data/bpe_model", vocab_size=32000):
        """
        Train a BPE model on Armenian text.

        Args:
            text_file: path to a .txt file with training text (NFC-normalized
                by 2_prepare.py — this trainer applies NO unicode normalization)
            model_prefix: where to save the model (creates .model and .vocab files)
            vocab_size: number of subword tokens to learn
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
            normalization_rule_name="identity",  # NFC is applied upstream, once
            user_defined_symbols=V2_USER_DEFINED_SYMBOLS,
            byte_fallback=True,  # no UNK: unknown chars become byte pieces
            pad_id=3,
            input_sentence_size=1_000_000,  # sample 1M sentences for large files
            shuffle_input_sentence=True,
            num_threads=16,
        )
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        # Chat specials are native pieces in v2 — register them so encode()'s
        # boundary-exact special handling and decode()'s literal rendering work.
        self.add_special_tokens([t for t in V2_USER_DEFINED_SYMBOLS if t != "\n"])
        print(f"BPE tokenizer trained! Vocab size: {self.vocab_size}")

    def add_special_tokens(self, tokens):
        """
        Register multi-character special tokens.

        If a token is already a native piece of the SentencePiece model (v2
        training bakes the chat tokens in), its native id is used. Otherwise
        (v1 models) the token is grafted onto an id beyond the existing vocab.
        """
        for token in tokens:
            if token in self._special_token_to_id:
                continue
            native_id = None
            if self.sp is not None:
                pid = self.sp.piece_to_id(token)
                if pid != self.sp.unk_id():
                    native_id = pid
            idx = native_id if native_id is not None else self.vocab_size
            self._special_token_to_id[token] = idx
            self._id_to_special_token[idx] = token
        return self

    def encode(self, text):
        """Convert text to a list of integer token IDs.

        Applies NFC first: the v2 SentencePiece model performs no unicode
        normalization of its own, and the training corpus is NFC (2_prepare).
        For v1 (nfkc) models, NFC-then-NFKC equals NFKC, so this is a no-op
        difference.
        """
        text = unicodedata.normalize("NFC", text)
        if not self._special_token_to_id:
            return self.sp.encode(text)

        # Split text around special tokens, encode each segment, insert special
        # IDs. Even for v2 models (where sp.encode would find the native pieces
        # itself) the explicit split guarantees prompt/response boundaries fall
        # exactly on special-token edges — prepare_chat's loss mask relies on it.
        pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in self._special_token_to_id) + ")"
        )
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
        # Filter out unk tokens (id 0) to avoid ⁇ in output. v2 models with
        # byte_fallback never produce UNK; this is v1-checkpoint safety.
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
            "model_file": self.sp.serialized_model_proto().hex() if self.sp else None,
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
            tok._special_token_to_id = {
                k: int(v) for k, v in data["special_tokens"].items()
            }
            tok._id_to_special_token = {
                v: k for k, v in tok._special_token_to_id.items()
            }

        return tok
