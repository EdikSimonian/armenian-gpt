# ArmGPT Training & Performance Review — 2026-06-11

Deep review of the full pipeline (data → tokenizer → pretrain → SFT → inference) focused on
**what would improve model quality**, not bugs (see CODE_AUDIT.md for those). Context: giant
base (930.65M params) stopped deliberately at step 120,000 of 125k (~31.5B tokens seen,
~3.8 epochs over 8.20B unique tokens; LR had effectively annealed to min_lr by then, so the
early stop cost ~nothing). Chat v3 SFT'd from the 120k base on 17,025 examples / 1.49M tokens.

---

## Headline findings (new, empirically confirmed this session)

### 1. Paragraph breaks are UNK tokens — the corpus has no usable structure ⚠⚠
Verified by training a toy SentencePiece model with the exact flags from
`core/bpe_tokenizer.py:35-63`: **`\n\n` encodes to `<unk>` (id 0)**. SentencePiece with
`allow_whitespace_only_pieces=0` (default), no `user_defined_symbols`, and `byte_fallback=0`
has no piece containing `\n`, so every paragraph boundary in the 8.2B-token stream became UNK.

Consequences:
- The model's *only* learned paragraph/document separator is the unknown-token.
- `BPETokenizer.decode()` strips id 0 (to avoid "⁇"), so **the model can never render a
  paragraph break** — all output is a single run-on block. Lists, dialogue, and structured
  answers are visibly impaired.
- There is also **no EOS between documents** (`3_tokenize.py` confirms: no eos/bos insertion
  anywhere). Unrelated documents are butt-joined inside every 2048-token window
  (cross-document attention contamination), and the model never learns a stop signal —
  which is why generation needs `repetition_penalty` + hard token caps to terminate.

**Fix (v2 tokenizer + corpus rebuild):**
- `user_defined_symbols=["\n"]` (newline becomes a real token; `\n\n` = two tokens).
- Insert SP's `</s>` (id 2) between documents when building the bins in `3_tokenize.py`.
- `byte_fallback=True` so UNK ceases to exist (rare chars become byte pieces).
- Bake `<|user|>`/`<|assistant|>`/`<|end|>` into the SP training via `user_defined_symbols`
  so the 16000→16003 vocab-graft (and the recurring vocab-mismatch bugs it spawned across
  eval_bpb/7_deploy/Space) disappears.

### 2. NFKC decomposes և → եւ in every generation ⚠⚠
`core/bpe_tokenizer.py:58` trains SP with `normalization_rule_name="nfkc"`, while
`2_prepare.py` cleans with NFC. NFKC decomposes **U+0587 (և — among the most frequent
words in Armenian, "and")** into `եւ`. Confirmed in the toy test AND in production output:
`results/*.txt` contain **28 instances of "եւ" and 0 model-written "և"**. The model is
self-consistent (input prompts get NFKC'd too) but writes non-reform orthography on
virtually every sentence — a visible quality defect for modern Eastern Armenian.

**Fix:** `normalization_rule_name="identity"` + keep NFC in cleaning (one normalization,
applied once, upstream). Requires retokenize + retrain to fully fix; a decode-time
`եւ→և` rewrite is a stopgap but corrupts genuinely classical text.

### 3. The cleaning whitelist deletes, it doesn't replace ⚠
`2_prepare.py:83` — `_RE_NON_ARMENIAN.sub("", chunk)` removes every char outside
Armenian + ASCII `. , ; : ! ? - ( ) " ' 0-9`:
- **«» (U+00AB/BB) — the standard Armenian quotation marks — are deleted corpus-wide**,
  as are em/en-dashes and ellipsis. Deletion (not space-replacement) glues adjacent words:
  `մի—երկու` → `միերկու`.
- **All Latin script is deleted**: `Միացյալ Նահանգներ (USA)` → `Միացյալ Նահանգներ ()`.
  Mixed-script web text (very common) is left with dangling parens/fragments, and the model
  can never read or write a Latin name, acronym, or loanword.

**Fix:** substitute with space (never empty); whitelist «», –, —, …; decide deliberately on
Latin (recommend: keep A-Za-z — with byte_fallback even rare scripts degrade gracefully).
Re-cleaning is CPU-only and cheap relative to a GPU run.

---

## Priority 1 — Chat quality now, no base retrain needed

The June 1 eval (results/00_SUMMARY.md) profile — fluent + on-topic, but hallucinated
details, failed arithmetic, no clean refusals — is the classic signature of **data-poor
SFT, not base-model limits**. 1.49M tokens / ~17k single-turn examples / ~200 optimizer
steps is roughly 10-50× below what comparable small chat models use.

1. **Scale SFT data ~10× (to 10-20M tokens / 100k+ examples).** The generators already
   exist (Claude, Qwen/DeepInfra). Target the measured weaknesses: step-by-step reasoning
   and arithmetic (worst category), procedural how-tos, grounded factual QA.
2. **Stop capping Aya, start filtering it.** `2_prepare.py` caps aya at 5000 of ~40k to fix
   the mix ratio — that throws away 87% of the largest source. Replace the blunt cap with
   quality scoring (length-ratio, Armenian-ratio you already compute, dedup near-identical
   instructions) and keep the best 15-20k.
3. **Refusal data is being filtered out by the safety blocklist** (English-substring
   blocklist in `core/prepare_chat.py` drops the very examples that teach refusals —
   already noted in results/00_SUMMARY.md). Exempt refusal-category rows.
4. **Multi-turn**: `8_chat.py:179` sends only the current turn (stateless) and the SFT data
   is single-turn Alpaca-style. Add multi-turn conversations to the data AND accumulate
   history in the prompt at chat time (mask all-but-last responses, or all prior turns, at
   train time). Until both change, "chat" is really single-shot QA.
5. **SFT packing trains on truncated context.** `6_finetune.py:266` samples random windows
   over the concatenated stream, so a window that starts mid-response trains the model to
   produce answer-continuations with **no visible prompt** (~5-10% of supervised tokens at
   current sizes). Pack example-aligned instead: start windows at example boundaries and
   pad/concatenate greedily to block_size.
6. **New special-token rows init randomly** (std 0.02 in the tied wte/lm_head). With only
   ~200 SFT steps, initialize the 3 grafted rows to the mean embedding for faster adaptation.
7. **Sampling**: `generate()` has only top_k. Add **top_p (or better, min_p)** — typically
   strictly better than top_k=40 for small models; and use temp ~0.4-0.5 for factual prompts
   (results/00 already recommends this). Cheap, immediate.
8. Targeted-fact patch sets (`_gen_v2.py` ×12 upweights for letter-count/Ararat) fix demo
   prompts but won't generalize — keep them small so they don't crowd the mix.

Also: **commit the working tree** — the KV-cache rewrite of `core/model.py` is uncommitted
and is the version the shipped chat checkpoints depend on.

## Priority 2 — v2 pretrain corpus + tokenizer (the big lever per GPU-dollar)

Besides headline items 1-3 above:

9. **Near-duplicate dedup across CC-derived sources.** Five sources (cc100, culturax, mc4,
   hplt3, finetranslations) are all CommonCrawl derivatives; the blake2b exact-paragraph
   dedup (`2_prepare.py:97`) only catches identical strings. Web near-dup rates of 15-30%
   are typical, which silently multiplies the effective epoch count on duplicated content
   (you're nominally at 3.8 epochs — on duplicated text it's substantially higher →
   memorization, wasted capacity). MinHash-LSH at document level (`text-dedup` or
   `datasketch`) is the standard fix and runs on CPU.
10. **Add cheap quality filters** (Gopher-style): symbol-to-word ratio, duplicate-line
    fraction, mean-word-length bounds, minimum stopword presence. mC4/cc100-hy are known
    noisy for low-resource languages. Per-source files are already kept — score each source
    with bpb on a clean reference (wiki+arlis) and downweight the worst.
11. **Quality-annealed data ordering (midtraining).** You already order sources by priority;
    exploit it: train mostly on the full mix, then make the last ~10-15% of steps (the LR
    anneal tail) wiki/arlis/news-heavy. Consistent wins in recent recipes (MiniCPM, OLMo-2)
    for ~zero cost.
12. **Vocab 16k → 32k (measure first).** Compute chars/token of the current tokenizer on
    held-out text; if it's under ~2.5, a 32k vocab buys ~10-20% better compression → more
    text per 2048-token window, faster inference per char, more tokens/param headroom. At
    32k×1536 the (tied) embedding is 49M params — still <6% of the model. 16k is unusually
    small for a 1B-class model even monolingually.
13. **Val/test are not representative.** The 90/10 split is a *sequential tail* of a corpus
    that is written in source-priority order, so val/test ≈ the lowest-priority web filler
    (mc4/cc100 end). Val loss and test-bpb therefore track web-junk perplexity, not
    wiki-quality Armenian. For v2: hold out a stratified per-source slice (and keep
    splitting at document boundaries, not byte offsets — the current split bisects a doc).
14. `eval/heldout.txt` is 27KB (~7k tokens) — bpb on it is noisy. Collect ≥1-5MB of
    guaranteed-post-scrape text (new hy-wiki articles, recent news) as the canonical
    cross-run comparison set.

## Priority 3 — Training-recipe fixes (free, adopt for any future run)

15. **Weight-decay grouping** — `4_train.py:262` decays *everything*, including RMSNorm
    gains. Standard: decay only dim≥2 params (nanoGPT `configure_optimizers` pattern).
16. **AdamW betas (0.9, 0.95)** — currently PyTorch default (0.9, 0.999). Every GPT-family
    recipe (GPT-2/3, LLaMA, OLMo) uses β2=0.95 at this scale; 0.999 adapts too slowly to
    gradient-variance shifts and correlates with late-run spikes.
17. **Scaled residual init** — `core/model.py:_init_weights` gives every linear std 0.02.
    GPT-2/nanoGPT scale the residual-output projections (`attn.c_proj`, `mlp.w2`) by
    1/√(2·n_layer) → std ≈ 0.0025 at 32 layers. Helps early optimization in deep stacks.
18. Optional stabilizers if you push LR or scale: **QK-norm**; grad-norm logging in the
    step log (catches incipient divergence — currently only loss is logged).
19. **Consider WSD (warmup-stable-decay) instead of cosine** for v2: this run's horizon
    changed mid-flight (122k→125k, stopped 120k) — WSD lets you stop/extend at any point
    and "anneal on demand", and enables clean continued-pretraining later.
20. Throughput (matters at 11-day runs): batch loading is synchronous, unpinned
    (`get_batch_seq` → `.to(device)` between every micro-backward). Pinned memory +
    `non_blocking=True` + one-batch-ahead prefetch thread typically buys a few percent ×16
    micro-steps. `estimate_loss` in 4_train runs un-autocast (fp32) — 2-3× slower eval than
    needed; wrap it like 6_finetune does. A/B `torch.compile(mode="max-autotune")`.
21. Effective batch 262k tokens is on the small side for ~1B (GPT-3 1.3B used ~1M). If you
    revisit, raising accum to 32 (≈524k) with LR ~2.8-3e-4 is the conventional pairing —
    second-order gain, not urgent.

## Priority 4 — Strategy: the from-scratch ceiling

At 930M params / 8.2B unique tokens (33.8 tokens/param, 3.8 epochs) you are already at the
sensible data ceiling — the config comment saying "don't go bigger from scratch" is right.
The eval's persistent weaknesses (math, science, multi-step procedures) are exactly what
small from-scratch models can't learn from 8B tokens of one language.

The step-change option is **continued pretraining of a multilingual open base**
(Qwen2.5-1.5B / Llama-3.2-1B / Qwen2.5-3B with grad-checkpointing — all fit the 96GB card)
on the Armenian corpus, then your SFT. World knowledge and reasoning transfer across
languages; you'd keep the whole existing pipeline (data, SFT, eval) and only swap the
model/trainer. Expect a different capability class than any from-scratch 1B. Trade-off:
not "ours from scratch", tokenizer is theirs (Qwen's covers Armenian poorly per-token —
budget ~2× tokens per char vs a custom 32k, still usually worth it).

Eval infra to make any of these comparisons honest:
- Wire **ArmBench** (already downloaded: simpleqa, squad-in-context, belebele-MCQ) into a
  log-prob MCQ scorer → capability metric beyond bpb, works on base models too.
- Expand the 10-prompt diagnostic to ~50 incl. multi-turn; keep `test_bpe.bin` frozen;
  grow the fresh-text bpb set (item 14).

---

## What's already good (keep)

- Architecture choices are modern and correct: RMSNorm, SwiGLU (8/3, 64-aligned), RoPE,
  SDPA/Flash, weight tying (standard at ≤1B), dropout 0 for pretrain, bf16 autocast,
  fused AdamW, tf32, torch.compile.
- min_lr = 10% of peak (fixed mid-run), warmup 2k, grad-clip 1.0 — all sane.
- The block-aligned no-replacement epoch sampler with per-epoch phase shift
  (`4_train.py:104`) is genuinely better than nanoGPT's sample-with-replacement.
- Loss-masked SFT with hard-error on missing masks, val-tracked `chat_best.pt`,
  example-level shuffled chat split — right design.
- bpb harness for cross-tokenizer comparison; frozen test split discipline.
- KV-cached generation with correct RoPE offsets (uncommitted — commit it).

## Suggested order of operations

1. (days, CPU+API) SFT data scale-up + refusal/blocklist fix + multi-turn + packing fix →
   re-run SFT (~hours on gpu-pc) → re-eval. Biggest user-visible win, no base retrain.
2. (days, CPU) v2 corpus: cleaning fixes + MinHash dedup + tokenizer rebuild (byte_fallback,
   \n, identity-norm, EOS, specials, 32k) + stratified holdout. Measure chars/token before/after.
3. (~11 GPU-days) v2 giant retrain with recipe fixes (items 15-19) + annealed data ordering.
4. (parallel, exploratory) CPT a Qwen2.5-1.5B on the same corpus for one short run and
   compare ArmBench/bpb/diagnostic against giant-v2 before committing to either lineage.
