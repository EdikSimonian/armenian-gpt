"""
Step 3: Tokenize the prepared data.

Default mode (corpus): Reads data/clean_text.txt (produced by 2_prepare.py),
trains either a character-level or BPE (SentencePiece) tokenizer, and writes
the encoded corpus as a 90/10 train/val split.

--qa mode: Reads data/qa_merged.json (produced by 2_prepare.py --qa), loads
the Stage 1 tokenizer from data/tokenizer_{type}.json, extends it with chat
special tokens (<|user|>, <|assistant|>, <|end|>), and writes the chat-
formatted conversations as train/val bins under data_chat/.

Outputs (corpus mode, --tokenizer char):
    data/train_char.bin, data/val_char.bin, data/tokenizer_char.json

Outputs (corpus mode, --tokenizer bpe):
    data/train_bpe.bin, data/val_bpe.bin, data/tokenizer_bpe.json
    (also data/bpe_model.{model,vocab} from SentencePiece training)

Outputs (--qa mode):
    data_chat/train_{char,bpe}.bin
    data_chat/val_{char,bpe}.bin
    data_chat/tokenizer_{char,bpe}.json

Char and BPE outputs use disjoint filenames so they coexist on disk.

Usage:
    python 3_tokenize.py --tokenizer bpe              # corpus mode
    python 3_tokenize.py --tokenizer char             # corpus mode
    python 3_tokenize.py --qa --tokenizer bpe         # chat mode
"""

import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLEAN_FILE = os.path.join(DATA_DIR, "clean_text.txt")


def _find_segment_boundaries(path, num_segments):
    """Find byte offsets that split a file into segments at newline boundaries."""
    file_size = os.path.getsize(path)
    boundaries = [0]
    with open(path, "rb") as f:
        for i in range(1, num_segments):
            target = (file_size * i) // num_segments
            f.seek(target)
            chunk = f.read(8192)
            nl = chunk.find(b"\n")
            if nl != -1:
                boundaries.append(target + nl + 1)
            else:
                boundaries.append(target)
    boundaries.append(file_size)
    return boundaries


def build_char_vocab(clean_path, chunk_bytes=50_000_000):
    """Scan the clean file to build character vocabulary without loading it all."""
    from tokenizers.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    chars = set()

    with open(clean_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            chars.update(chunk)

    tokenizer.itos = sorted(chars)
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.itos)}
    return tokenizer


def encode_char_chunked(clean_path, tokenizer, output_path, chunk_bytes=50_000_000):
    """Encode text file to token IDs in chunks, appending to output binary file."""
    max_cp = max(ord(ch) for ch in tokenizer.stoi) + 1
    lookup = np.full(max_cp, -1, dtype=np.int32)
    for ch, idx in tokenizer.stoi.items():
        lookup[ord(ch)] = idx

    total_tokens = 0
    with open(clean_path, "r", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:
        while True:
            chunk = fin.read(chunk_bytes)
            if not chunk:
                break
            codepoints = np.array([ord(ch) for ch in chunk], dtype=np.int32)
            valid = codepoints[codepoints < max_cp]
            token_ids = lookup[valid]
            token_ids = token_ids[token_ids >= 0].astype(np.uint16)
            token_ids.tofile(fout)
            total_tokens += len(token_ids)

    return total_tokens


def _encode_bpe_segment(args):
    """Encode one segment of clean text with BPE (for multiprocessing)."""
    clean_path, start_byte, end_byte, segment_id, model_proto_hex = args
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(bytes.fromhex(model_proto_hex))

    out_path = os.path.join(DATA_DIR, f"encode_seg_{segment_id}.bin")
    chunk_size = 10_000_000
    total_tokens = 0

    with open(clean_path, "rb") as fin, open(out_path, "wb") as fout:
        fin.seek(start_byte)
        buffer = ""
        while fin.tell() < end_byte:
            remaining = end_byte - fin.tell()
            raw_bytes = fin.read(min(chunk_size, remaining))
            if not raw_bytes:
                break
            if fin.tell() < end_byte:
                extra = fin.read(8192)
                if extra:
                    nl = extra.find(b"\n")
                    if nl != -1:
                        raw_bytes += extra[:nl + 1]
                        fin.seek(-(len(extra) - nl - 1), 1)
                    else:
                        raw_bytes += extra
            buffer += raw_bytes.decode("utf-8", errors="ignore")
            last_para = buffer.rfind("\n\n")
            if last_para != -1:
                split_at = last_para + 2
            else:
                last_nl = buffer.rfind("\n")
                if last_nl != -1:
                    split_at = last_nl + 1
                elif len(buffer) > 3 * chunk_size:
                    split_at = len(buffer)
                else:
                    continue
            to_encode = buffer[:split_at]
            buffer = buffer[split_at:]
            ids = sp.encode(to_encode)
            np.array(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)

        if buffer:
            ids = sp.encode(buffer)
            np.array(ids, dtype=np.uint16).tofile(fout)
            total_tokens += len(ids)

    seg_mb = (end_byte - start_byte) / (1024 * 1024)
    print(f"  Segment {segment_id}: {seg_mb:.0f} MB -> {total_tokens:,} tokens")
    return out_path, total_tokens


def encode_bpe_chunked(clean_path, tokenizer, output_path):
    """Encode text file with BPE, using parallel workers."""
    total_bytes = os.path.getsize(clean_path)
    num_workers = min(cpu_count(), 16)

    print(f"  Encoding {total_bytes / 1024 / 1024:.0f} MB with {num_workers} parallel workers...")
    boundaries = _find_segment_boundaries(clean_path, num_workers)
    model_proto_hex = tokenizer.sp.serialized_model_proto().hex()
    args = [
        (clean_path, boundaries[i], boundaries[i + 1], i, model_proto_hex)
        for i in range(len(boundaries) - 1)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_encode_bpe_segment, args)

    total_tokens = 0
    with open(output_path, "wb") as fout:
        for seg_path, seg_tokens in results:
            total_tokens += seg_tokens
            with open(seg_path, "rb") as fin:
                while True:
                    chunk = fin.read(100 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
            os.remove(seg_path)

    return total_tokens


def split_bin_file(all_tokens_path, train_path, val_path, val_ratio=0.1):
    """Split a single .bin file into train and val without loading into RAM."""
    total_bytes = os.path.getsize(all_tokens_path)
    total_tokens = total_bytes // 2  # uint16 = 2 bytes per token
    split_token = int(total_tokens * (1 - val_ratio))
    split_byte = split_token * 2

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Train: {split_token:,} tokens, Val: {total_tokens - split_token:,} tokens")

    chunk_size = 100 * 1024 * 1024
    with open(all_tokens_path, "rb") as fin:
        with open(train_path, "wb") as fout:
            remaining = split_byte
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = fin.read(to_read)
                if not data:
                    break
                fout.write(data)
                remaining -= len(data)

        with open(val_path, "wb") as fout:
            while True:
                data = fin.read(chunk_size)
                if not data:
                    break
                fout.write(data)

    print(f"  Train: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")
    print(f"  Val:   {os.path.getsize(val_path) / 1024 / 1024:.1f} MB")


def tokenize_corpus(tokenizer_type):
    if not os.path.exists(CLEAN_FILE):
        print(f"Error: {CLEAN_FILE} not found!")
        print("Run 'python 2_prepare.py' first to clean the data.")
        sys.exit(1)

    clean_size = os.path.getsize(CLEAN_FILE)
    print(f"\n{'='*50}")
    print(f"  Step 3: Train tokenizer and encode")
    print(f"{'='*50}")
    print(f"  Input:     {CLEAN_FILE} ({clean_size / 1024 / 1024:.0f} MB)")
    print(f"  Tokenizer: {tokenizer_type}")
    print(f"{'='*50}\n")

    print("Step 3a: Building tokenizer...")
    all_tokens_path = os.path.join(DATA_DIR, "all_tokens.bin")

    if tokenizer_type == "char":
        tokenizer = build_char_vocab(CLEAN_FILE)
        print(f"  Vocabulary: {tokenizer.vocab_size} characters")

        print("\nStep 3b: Encoding text...")
        total_tokens = encode_char_chunked(CLEAN_FILE, tokenizer, all_tokens_path)
    else:
        from tokenizers.bpe_tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.train(CLEAN_FILE, model_prefix=os.path.join(DATA_DIR, "bpe_model"))
        print(f"  Vocabulary: {tokenizer.vocab_size} tokens")

        print("\nStep 3b: Encoding text...")
        total_tokens = encode_bpe_chunked(CLEAN_FILE, tokenizer, all_tokens_path)

    print(f"  Total tokens: {total_tokens:,}")

    print("\nStep 3c: Splitting train/val (90/10)...")
    from tokenizers import bin_paths, tokenizer_path
    train_path, val_path = bin_paths(DATA_DIR, tokenizer_type)
    split_bin_file(all_tokens_path, train_path, val_path)

    os.remove(all_tokens_path)

    tok_path = tokenizer_path(DATA_DIR, tokenizer_type)
    tokenizer.save(tok_path)

    print(f"\n{'='*50}")
    print(f"  Step 3 complete")
    print(f"{'='*50}")
    print(f"  Tokenizer:    {tokenizer_type} ({tokenizer.vocab_size} tokens)")
    print(f"  Train tokens: {os.path.getsize(train_path) // 2:,} ({train_path})")
    print(f"  Val tokens:   {os.path.getsize(val_path) // 2:,} ({val_path})")
    print(f"  Tokenizer:    {tok_path}")
    print(f"\nNext step: python 4_train.py --preset tiny --tokenizer {tokenizer_type}")


def tokenize_qa(tokenizer_type):
    """Tokenize the merged SFT JSON into data_chat/*.bin for fine-tuning."""
    from data.prepare_chat import prepare_chat_data

    source_path = os.path.join(DATA_DIR, "qa_merged.json")
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found!")
        print("Run 'python 2_prepare.py --qa' first to merge SFT sources.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Step 3: Tokenize chat data (--qa)")
    print(f"{'='*50}")
    print(f"  Input:     {source_path}")
    print(f"  Tokenizer: {tokenizer_type} (extended with chat special tokens)")
    print(f"{'='*50}\n")

    prepare_chat_data(source_path, tokenizer_type)

    print(f"\nNext step: python 5_finetune.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        choices=["char", "bpe"],
                        help="Tokenizer type: 'char' (Level 1) or 'bpe' (Level 2)")
    parser.add_argument("--qa", action="store_true",
                        help="Tokenize Q&A data (data/qa_merged.json) into data_chat/ "
                             "instead of the raw corpus")
    args = parser.parse_args()

    if args.qa:
        tokenize_qa(args.tokenizer)
    else:
        tokenize_corpus(args.tokenizer)


if __name__ == "__main__":
    main()
