# ArmGPT v2 — changes on the `next` branch

All improvements from `TRAINING_REVIEW.md` (2026-06-11), implemented and
smoke-tested end-to-end on the real numbered pipeline. This file is the runbook:
what changed, what it fixes, what needs a retrain to take effect, and how to run
the v2 pipeline.

## TL;DR — what to do

1. **Chat quality now (no base retrain):** rebuild the SFT data and re-finetune
   the existing 120k base. The refusal fix, multi-turn, quality-ranked Aya, and
   example-aligned packing all land here. Biggest user-visible win.
2. **Best quality (one ~11 GPU-day run):** rebuild the corpus + tokenizer
   (fixes below are baked at tokenize time) and pretrain `giant_v2`, then SFT.
3. **Decide the ceiling:** run the `eval_armbench.py` MCQ harness across base
   checkpoints to see whether from-scratch has plateaued before committing to a
   from-scratch v2 vs. continued-pretraining of a multilingual base.

The architecture, sampler, optimizer, and inference changes are **backward
compatible** — existing v1 checkpoints still load and run unchanged (qk_norm is
auto-detected as off; the v1 grafted-tokenizer path still works).

---

## What needs a retrain vs. what is live immediately

| Change | Takes effect |
|---|---|
| Tokenizer: identity-norm (`և` fix), newline piece, byte_fallback, native chat tokens, 32k vocab | **retokenize + retrain** (`3_tokenize`) |
| Corpus cleaning: keep «»/dashes/Latin, space-replace, canonical dedup | **re-clean + retrain** (`2_prepare`) |
| Document EOS / stop signal | **retokenize + retrain** |
| MinHash near-dedup | **re-clean + retrain** (optional `2b`) |
| Scaled residual init, QK-norm option, AdamW betas, wd grouping, WSD | **retrain** (init/optimizer are train-time) |
| Sampling: top_p / min_p | **live** on any checkpoint |
| SFT: refusal bypass, multi-turn, quality cap, example-aligned packing, mean-init | **re-SFT** (no base retrain) |
| Inference: multi-turn chat, MCQ eval harness | **live** |

---

## v2 pipeline (full)

```bash
# 1. Download (now emits <|enddoc|> document separators)
python 1_download.py

# 2. Clean + exact-dedup (keeps «»/dashes/Latin; canonical dedup; passes separators)
python 2_prepare.py

# 3. (optional) MinHash near-duplicate removal across documents
python 2b_dedup_fuzzy.py            # rewrites clean_text.txt, backs up .bak

# 4. Tokenize: 32k vocab, identity-norm, byte_fallback, EOS between docs,
#    doc-boundary train/val split, prints chars/token
python 3_tokenize.py --tokenizer bpe --vocab_size 32000

# 5. Pretrain the v2 base (WSD schedule, scaled init, betas, wd grouping)
python 4_train.py --preset giant_v2 --tokenizer bpe
#    add --qk_norm if you raise LR/depth and see late-run loss spikes

# --- Chat ---
python 1_download.py --qa                              # SFT sources
python 2_prepare.py --qa                               # merge + quality-rank Aya (best 15k)
python 3_tokenize.py --qa --tokenizer bpe              # writes *_idx bins too
python 6_finetune.py --preset finetune_giant --resume_from checkpoints/final.pt

# --- Talk / generate / evaluate ---
python 8_chat.py --checkpoint checkpoints_chat/chat_best.pt --min_p 0.05
python 5_generate.py --checkpoint checkpoints/final.pt --top_p 0.9
python eval_armbench.py --checkpoint-dir checkpoints/ \
    --data data/text/finetune/armbench_eval.json --log eval/armbench_log.json
```

---

## Detailed changes

### Data pipeline

- **`core/bpe_tokenizer.py`** — `normalization_rule_name="identity"` (NFC is now
  applied once, upstream, in `encode()` and `2_prepare`; the old `nfkc`
  decomposed `և`→`եւ` in every generation). `user_defined_symbols=["\n", <chat
  tokens>]` so newline is a real piece (was UNK — the model could not emit a
  paragraph break) and the chat tokens are native (no more 16000→16003 graft).
  `byte_fallback=True` eliminates UNK. Default vocab 16000→32000. New
  `eos_id`/`newline_id`; `add_special_tokens()` reuses native ids and still
  grafts for v1 models.
- **`2_prepare.py`** — cleaning whitelist now keeps `A-Za-z «» – — …` and
  **replaces** disallowed chars with a space instead of deleting (no more
  `մի—երկու`→`միերկու` or dangling `()`). Dedup hash is canonicalized
  (whitespace + casefold; digits/punct kept distinct). Document separators pass
  through; `_MergedWriter` collapses leading/trailing/duplicate-doc separators.
  Aya capped to the **best 15k by quality** (was random 5k).
- **`1_download.py`** — every document writer emits a `<|enddoc|>` sentinel via
  one `_write_doc` helper (wiki, wikimedia, hplt3, arlis, ccnews, streaming HF).
- **`3_tokenize.py`** — trains the tokenizer on a separator-free copy, then
  converts each `<|enddoc|>` to **EOS** at encode time (stop signal + no
  cross-document attention bleed). Train/val split **snaps to a document
  boundary**. Prints **chars/token**. New `--vocab_size`.
- **`2b_dedup_fuzzy.py`** (new) — MinHash + LSH near-duplicate document removal,
  pure numpy/stdlib. Keeps the highest-priority (earliest) copy of each cluster.

### Model & training recipe

- **`core/model.py`** — scaled residual init (`c_proj`/`mlp.w2` std =
  `0.02/sqrt(2·n_layer)`); optional per-head **QK-norm** (`qk_norm`, off by
  default, KV-cache-equivalent); `generate()` gains **top_p** and **min_p**.
- **`core/training.py`** (new, shared by pretrain + SFT) — `configure_optimizer`
  (weight decay on dim≥2 only, **AdamW betas (0.9, 0.95)**, fused on CUDA);
  `get_lr` (**cosine or WSD**); `load_optimizer_state` (tolerates a pre-v2
  single-group optimizer state on resume).
- **`core/config.py`** — `giant_v2` preset (WSD, qk_norm hook); knobs
  `lr_schedule/decay_frac/qk_norm/compile_mode` + CLI flags; **min_lr clamped**
  to the peak LR; added missing `--warmup_iters/--weight_decay/--grad_clip/
  --eval_iters` flags.
- **`4_train.py`** — uses the shared optimizer + schedule; passes `qk_norm`;
  `torch.compile(mode=...)`; **grad-norm logged** each interval; `estimate_loss`
  runs under autocast (2-3× faster, matches train regime); pinned-memory
  `non_blocking` host→device copies; fixed mixed-script sample seed
  `Հayastan`→`Հայաստան`.

### SFT & inference

- **`core/prepare_chat.py`** — refusal/safety rows **bypass** the (now
  word-boundary) blocklist; accepts **multi-turn** `{messages:[...]}`; writes
  **`*_idx` example-offset bins**; min output length 10→3 (refusals exempt).
- **`core/merge_sft_sources.py`** — quality-ranked cap (`rank: "quality"`):
  keep the best N of a noisy source by Armenian-ratio/length/no-artifacts.
- **`6_finetune.py`** — **example-aligned** `get_batch` (windows start at
  example boundaries; short tails right-padded), so the model never trains on a
  prompt-less answer continuation; **mean-init** of grafted token rows; shared
  optimizer + schedule.
- **`8_chat.py`** — **multi-turn** conversation memory with context-budget
  eviction (`reset` clears, `--single_turn` opts out); response decoded from the
  newly generated tokens only; `--top_p/--min_p`; qk_norm auto-detected.
- **`5_generate.py`** — `--top_p/--min_p`; qk_norm auto-detected.
- **`eval_armbench.py`** (new) — multiple-choice accuracy via length-normalized
  log-prob (`acc` + `acc_norm`), works on **base** checkpoints. Parses both
  structured options and the real ArmBench inline-lettered belebele format
  (all 50 belebele items parse; the 100 open-ended items correctly skip).

---

## Not done here (bigger calls, documented for later)

- **Stratified-by-source held-out eval split** — the current split is a
  document-aligned tail; a per-source stratified holdout needs source tags
  carried past the `2_prepare` merge (or a separate held-out build step).
- **One-batch-ahead prefetch thread** in `4_train` — pinned `non_blocking`
  copies capture most of the win; a full double-buffer thread is the remaining
  few percent.
- **Continued pretraining of a multilingual base (Qwen2.5-1.5B)** — the
  strategic step-change for math/reasoning; keep the whole v2 data + SFT + eval
  pipeline, swap only the model/trainer.

## Verification

`/tmp/smoke.sh` ran the real scripts on a synthetic corpus:
`2_prepare → 2b_dedup_fuzzy → 3_tokenize (multiprocessing) → 4_train (WSD +
qk_norm) → 5_generate (top_p/min_p) → eval_armbench (real belebele) →
prepare_chat → 6_finetune (example-aligned) → 8_chat (multi-turn)` — all green.
Each component also has a focused unit test (tokenizer round-trips, EOS
placement, optimizer grouping, LR schedules, sampler edge cases, refusal/
multi-turn encoding, quality cap, MinHash clustering, MCQ gold parsing).
