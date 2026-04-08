# ArmGPT fine-tuning pipeline

End-to-end pipeline for translating SmolTalk2 chat data to Armenian and
fine-tuning ArmGPT.

## Files

| File | Purpose |
|---|---|
| `build_two_sets.py` | Sample 2 × 10k disjoint sets from SmolTalk2 — DONE |
| `translate_nllb.py` | Bulk EN→ARM via local NLLB-200-3.3B (~1-2h per 10k) |
| `translate_claude_sample.py` | Translate ~100 samples via Claude API for quality reference |
| `compare_nllb_vs_claude.py` | Grade NLLB output against Claude reference, get pass rate |
| _(future)_ `finetune_lora.py` | LoRA fine-tune ArmGPT on the Armenian set |

## Order of operations (run AFTER training finishes at step 36000)

```bash
cd /workspace/finetune

# 1. Bulk translate Set 1 (chat) - ~1-2 hours on idle A6000
python scripts/translate_nllb.py sets/set1_chat_en.jsonl sets/set1_chat_arm.jsonl

# 2. Same for Set 2 (tasks)
python scripts/translate_nllb.py sets/set2_tasks_en.jsonl sets/set2_tasks_arm.jsonl

# 3. Quality validation: 100-sample Claude reference for Set 1
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/translate_claude_sample.py sets/set1_chat_en.jsonl sets/set1_chat_claude_sample.jsonl 100

# 4. Grade NLLB vs Claude for the same indices
python scripts/compare_nllb_vs_claude.py \
    sets/set1_chat_arm.jsonl \
    sets/set1_chat_claude_sample.jsonl \
    sets/set1_chat_grades.jsonl
```

## Expected outputs

After step 4, you'll see something like:
```
Total graded messages: ~250
Mean grade: 3.8 / 5
% >= 4 (good enough for FT): 65%
```

**Decision rule:** if NLLB hits ≥4 on ≥60% of samples, the bulk translation is
fine-tuning quality and you proceed with FT. Otherwise consider:
- Using NLLB-200-1.3B distilled (faster but worse) → not the issue
- Switching to Claude API for the full bulk translation (~$30-60 for both sets)
- Using a different multilingual model

## Notes

- Translation requires the GPU. **Don't run while training is active.**
- All scripts are resumable (re-run after interruption to continue).
- Output JSONL has the original English under `messages_en` for traceability.
- Sets are at `/workspace/finetune/sets/`, scripts at `/workspace/finetune/scripts/`.
