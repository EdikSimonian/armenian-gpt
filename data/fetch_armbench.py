"""
Pull native-Armenian QA/MCQ from Metric-AI/ArmBench-LLM-data and normalize
into the {instruction, input, output} format used by prepare_chat.py.

ArmBench is small (~2.9k rows across 24 task configs) but its exam and
civics configs contain GENUINELY ARMENIAN-NATIVE content — real national
exam questions, real public-services policy questions — which is uniquely
rare. Most other "Armenian" datasets are machine-translated English.

Training configs (native Armenian exam/civics content):
  - exam_history          : national history exam MCQ
  - exam_literature       : literature / grammar exam MCQ
  - exam_math             : math exam MCQ (answer label is A/B/C/D)
  - include-mcqa          : driving-license exam MCQ
  - public-services-mcqa  : civics/legal questions with real answer text
                            (not just a choice letter) — highest-quality slice

Eval-holdout configs (stay out of training so they can measure progress):
  - simpleqa              : open-ended encyclopedic Q&A, best eval signal
  - squad-in-context-qa   : context + question + extractive answer
  - belebele-in-context-mcqa : human-translated reading comp

Usage:
    python data/fetch_armbench.py
    python data/fetch_armbench.py --output data/armbench_train.json \
                                  --eval_output data/armbench_eval.json
"""

import argparse
import json
import os
import sys

from datasets import load_dataset


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CONFIGS = [
    "exam_history",
    "exam_literature",
    "exam_math",
    "include-mcqa",
    "public-services-mcqa",
]

EVAL_CONFIGS = [
    "simpleqa",
    "squad-in-context-qa",
    "belebele-in-context-mcqa",
]


def _first_split(repo_id, config):
    """Load whatever split the config actually provides (varies across configs)."""
    ds = load_dataset(repo_id, config)
    split = list(ds.keys())[0]
    return ds[split]


def _normalize_single_label(label, n_choices):
    """Normalize a SINGLE-answer label to a 0-indexed int.

    ArmBench label shapes we care about:
      - ['2']   → 1-indexed string int  → returns 1
      - ['B']   → letter (exam_math)    → returns 1
      - 2       → raw 1-indexed int     → returns 1
    Raises ValueError for multi-answer / T-F / matching / ordering labels.
    """
    if isinstance(label, list):
        if len(label) != 1:
            raise ValueError(f"multi-label, not a single-answer task: {label!r}")
        label = label[0]
    if isinstance(label, str):
        s = label.strip()
        if s.isdigit():
            idx = int(s) - 1  # 1-indexed
            if not (0 <= idx < n_choices):
                raise ValueError(f"label {s!r} out of range for {n_choices} choices")
            return idx
        if len(s) == 1 and s.upper() in "ABCDEFGH":
            return ord(s.upper()) - ord("A")
    if isinstance(label, int):
        if 0 <= label < n_choices:
            return label
        if 1 <= label <= n_choices:
            return label - 1
    raise ValueError(f"Unrecognized label {label!r}")


_ARMENIAN_LETTERS = ["Ա", "Բ", "Գ", "Դ", "Ե", "Զ", "Է", "Ը", "Թ", "Ժ"]


def _format_mcq(question, choices, correct_idx, context=None):
    """Format an MCQ as a single instruction string with lettered choices."""
    if len(choices) > len(_ARMENIAN_LETTERS):
        raise ValueError(f"Too many choices ({len(choices)}) for MCQ format")
    body = question.strip()
    if context:
        body = f"Տեքստ:\n{context.strip()}\n\nՀարց: {body}"
    lines = [body, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{_ARMENIAN_LETTERS[i]}) {str(choice).strip()}")
    instruction = "\n".join(lines)
    answer_text = str(choices[correct_idx]).strip()
    output = f"{_ARMENIAN_LETTERS[correct_idx]}) {answer_text}"
    return instruction, output


def process_exam_config(cfg):
    """exam_history / exam_literature / exam_math.

    Handles only the three task_types that map cleanly to a single-answer
    Q&A pair:
      - task_type=1: standard MCQ, label = ['N'] (1-indexed)
      - task_type=6: letter-labeled MCQ, label = ['A'..'D']
      - task_type=7: open-ended (no choices), label = [answer_string]

    Skips task_types 2 (multi-select), 3 (per-statement T/F), 4 (matching),
    5 (chronological ordering) — those don't collapse into a single answer.
    """
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    skipped_by_tt = {}
    for row in ds:
        tt = row.get("task_type")
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        task = (row.get("task") or "").strip()
        choices = row.get("choices") or []
        label = row.get("label")
        if not question or label is None:
            continue

        # exam_math sometimes carries a task prefix like "Կատարել առաջադրանքները."
        full_question = f"{task} {question}".strip() if task else question

        if tt == 7:
            # Open-ended: no choices, the label IS the answer.
            answer = label[0] if isinstance(label, list) and label else label
            answer = str(answer).strip()
            if not answer:
                continue
            instruction = (f"Տեքստ:\n{context}\n\nՀարց: {full_question}"
                           if context else full_question)
            out.append({
                "instruction": instruction,
                "input": "",
                "output": answer,
                "source": f"armbench/{cfg}/open",
            })
            continue

        if tt in (1, 6):
            if not choices:
                continue
            try:
                correct_idx = _normalize_single_label(label, len(choices))
            except Exception:
                skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1
                continue
            try:
                instruction, output = _format_mcq(
                    full_question, choices, correct_idx,
                    context=context or None,
                )
            except ValueError:
                skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1
                continue
            out.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": f"armbench/{cfg}/mcq",
            })
            continue

        # task_types 2/3/4/5 — format doesn't map to single-answer SFT
        skipped_by_tt[tt] = skipped_by_tt.get(tt, 0) + 1

    if skipped_by_tt:
        print(f"  [{cfg}] skipped by task_type: {skipped_by_tt}")
    return out


def process_include_mcqa(cfg="include-mcqa"):
    """include-mcqa: question + option_a..option_d + answer (1-indexed int)."""
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        choices = [row.get(f"option_{k}") or "" for k in ("a", "b", "c", "d")]
        choices = [c for c in choices if c]
        answer = row.get("answer")
        if not question or len(choices) != 4 or answer is None:
            continue
        try:
            correct_idx = _normalize_single_label(answer, 4)
        except Exception:
            continue
        if not (0 <= correct_idx < 4):
            continue
        instruction, output = _format_mcq(question, choices, correct_idx)
        out.append({"instruction": instruction, "input": "", "output": output,
                    "source": f"armbench/{cfg}"})
    return out


def process_public_services(cfg="public-services-mcqa"):
    """public-services-mcqa: question + answer (text!) + distractors.

    This config is special: it gives a free-form answer, not just a letter.
    Emit BOTH an open-ended version (highest-quality SFT signal) AND an MCQ
    version (for format diversity).
    """
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        distractors = row.get("distractors") or []
        if not question or not answer:
            continue
        # Open-ended version — cleanest for a chat model
        out.append({
            "instruction": question,
            "input": "",
            "output": answer,
            "source": f"armbench/{cfg}/open",
        })
        # MCQ version (answer is always index 0 in our emit; shuffle below)
        if distractors:
            import random
            rng = random.Random(hash(question) & 0xFFFFFFFF)  # deterministic per row
            choices = [answer] + list(distractors)
            idxs = list(range(len(choices)))
            rng.shuffle(idxs)
            shuffled = [choices[i] for i in idxs]
            correct_idx = idxs.index(0)
            instruction, output = _format_mcq(question, shuffled, correct_idx)
            out.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": f"armbench/{cfg}/mcq",
            })
    return out


def process_simpleqa(cfg="simpleqa"):
    """simpleqa: question + answer, open-ended. Best eval format."""
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        out.append({"instruction": question, "input": "", "output": answer,
                    "source": f"armbench/{cfg}"})
    return out


def process_squad_in_context(cfg="squad-in-context-qa"):
    """squad-in-context-qa: context + question + answer (extractive)."""
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    for row in ds:
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answer = row.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = (answer or "").strip()
        if not question or not answer:
            continue
        instruction = f"Տեքստ:\n{context}\n\nՀարց: {question}" if context else question
        out.append({"instruction": instruction, "input": "", "output": answer,
                    "source": f"armbench/{cfg}"})
    return out


def process_belebele(cfg="belebele-in-context-mcqa"):
    """belebele: flores_passage + question + 4 mc_answers + correct_answer_num."""
    ds = _first_split("Metric-AI/ArmBench-LLM-data", cfg)
    out = []
    for row in ds:
        passage = (row.get("flores_passage") or "").strip()
        question = (row.get("question") or "").strip()
        choices = [row.get(f"mc_answer{i}") or "" for i in (1, 2, 3, 4)]
        choices = [c for c in choices if c]
        correct = row.get("correct_answer_num")
        if not question or len(choices) != 4 or correct is None:
            continue
        try:
            correct_idx = _normalize_single_label(correct, 4)
        except Exception:
            continue
        instruction, output = _format_mcq(question, choices, correct_idx,
                                          context=passage)
        out.append({"instruction": instruction, "input": "", "output": output,
                    "source": f"armbench/{cfg}"})
    return out


PROCESSORS = {
    "exam_history": lambda: process_exam_config("exam_history"),
    "exam_literature": lambda: process_exam_config("exam_literature"),
    "exam_math": lambda: process_exam_config("exam_math"),
    "include-mcqa": process_include_mcqa,
    "public-services-mcqa": process_public_services,
    "simpleqa": process_simpleqa,
    "squad-in-context-qa": process_squad_in_context,
    "belebele-in-context-mcqa": process_belebele,
}


def fetch_armbench_qa(train_output_path, eval_output_path):
    """Fetch ArmBench native Armenian Q&A and write train/eval JSON files.

    Returns (n_train, n_eval) counts.
    """
    print("=" * 60)
    print("  ArmBench → Q&A normalizer")
    print("=" * 60)

    train_pairs = []
    for cfg in TRAIN_CONFIGS:
        print(f"\n[train] Processing {cfg}...")
        pairs = PROCESSORS[cfg]()
        print(f"  {cfg}: {len(pairs):,} examples")
        train_pairs.extend(pairs)

    eval_pairs = []
    for cfg in EVAL_CONFIGS:
        print(f"\n[eval]  Processing {cfg}...")
        pairs = PROCESSORS[cfg]()
        print(f"  {cfg}: {len(pairs):,} examples")
        eval_pairs.extend(pairs)

    print(f"\n{'=' * 60}")
    print(f"  Training:   {len(train_pairs):,} examples")
    print(f"  Eval:       {len(eval_pairs):,} examples")
    print(f"{'=' * 60}")

    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved train → {train_output_path}")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved eval  → {eval_output_path}")

    return len(train_pairs), len(eval_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=os.path.join(DATA_DIR, "armbench_train.json"))
    parser.add_argument("--eval_output", type=str,
                        default=os.path.join(DATA_DIR, "armbench_eval.json"))
    args = parser.parse_args()
    fetch_armbench_qa(args.output, args.eval_output)


if __name__ == "__main__":
    main()
