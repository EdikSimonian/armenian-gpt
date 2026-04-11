"""
Armenian Q&A Dataset Generator — DeepInfra + Qwen3.5-27B variant.

Same prompt/format as the Claude-based `generate_armenian_qa.py` so the two
outputs are drop-in compatible with `prepare_chat.py`. The API client is
OpenAI-compatible pointing at DeepInfra, and the model is Qwen/Qwen3.5-27B
(explicitly documented to include Armenian in its training mix per Qwen3
technical report).

Usage:
    export DEEPINFRA_API_KEY=...
    python core/generate_armenian_qa_deepinfra.py

    # Optional: override target count for a test run (default 1000)
    python core/generate_armenian_qa_deepinfra.py --count 1000
    python core/generate_armenian_qa_deepinfra.py --count 50    # quick sanity check

Output:
    data/text/finetune/armenian_qa_qwen.json  — {instruction, input, output} pairs

Length distribution is shaped to exercise a 2048-token training context:
    ~30% short  (2-3 sentences, factual recall)
    ~40% medium (5-7 sentences, explanatory)
    ~30% long   (10-15 sentences, detailed reasoning / examples)

Cost estimate (DeepInfra Qwen3-Next-80B-A3B-Instruct):
    1000 pairs  → ~67 calls  → ~50k in + 550k out tokens → ~$0.15 total
    5000 pairs  → ~334 calls → ~250k in + 2.75M out → ~$0.75 total

Resume: safe to interrupt and rerun — resumes from last completed batch.
"""

import argparse
import json
import os
import sys
import time

# ── Topics (same 25 as the Claude variant for parity) ──────────────────────
TOPICS = [
    ("Հայոց պատմություն",           "Armenian history — ancient kingdoms, Urartu, Tigranes the Great, Genocide, independence"),
    ("Հայ մշակույթ",                "Armenian culture — traditions, weddings, music, dances, clothing, customs"),
    ("Հայ գրականություն",           "Armenian literature — poets, writers, Charents, Tumanyan, Siamanto, Narekatsi"),
    ("Հայ լեզու",                   "Armenian language — grammar, alphabet, Mesrop Mashtots, dialects, vocabulary"),
    ("Հայ ճարտարապետություն",       "Armenian architecture — churches, monasteries, Garni, Geghard, Etchmiadzin"),
    ("Աշխարհի պատմություն",         "World history — ancient civilizations, empires, revolutions, world wars, modern era"),
    ("Աշխարհագրություն",            "Geography — continents, countries, capitals, rivers, mountains, oceans"),
    ("Կենսաբանություն",             "Biology — cells, plants, animals, ecosystems, human body, genetics basics"),
    ("Ֆիզիկա",                     "Physics — motion, forces, energy, electricity, light, sound for beginners"),
    ("Քիմիա",                      "Chemistry — elements, atoms, reactions, periodic table, everyday chemistry"),
    ("Մաթեմատիկա",                  "Mathematics — numbers, geometry, algebra, fractions, logic, problem solving"),
    ("Տեխնոլոգիա",                  "Technology — computers, internet, programming basics, AI, smartphones, inventions"),
    ("Տիեզերք",                    "Space & astronomy — planets, stars, galaxies, astronauts, space exploration"),
    ("Բնություն ու շրջակա միջավայր", "Nature & environment — climate, ecosystems, forests, conservation, pollution"),
    ("Կենդանիներ",                  "Animals — mammals, birds, reptiles, insects, endangered species, animal behavior"),
    ("Առողջություն",                "Health & wellness — nutrition, exercise, sleep, hygiene, mental health basics"),
    ("Սպորտ",                      "Sports — football, chess, athletics, Olympic Games, Armenian athletes, teamwork"),
    ("Արվեստ ու երաժշտություն",     "Arts & music — painting, sculpture, classical music, Armenian folk songs, instruments"),
    ("Հայտնի գիտնականներ",         "Famous scientists & inventors — Newton, Einstein, Curie, Edison, Armenian scientists"),
    ("Ուտեստ ու խոհանոց",          "Food & cooking — Armenian cuisine, recipes, healthy eating, world foods"),
    ("Տոներ ու ավանդույթներ",       "Holidays — New Year, Easter, Vardavar, Trndez, Armenian celebrations, world holidays"),
    ("Տրամաբանություն",             "Logic & critical thinking — riddles, puzzles, patterns, reasoning, problem solving"),
    ("Բնապատմություն",              "Natural history — dinosaurs, evolution, fossils, ice age, prehistoric life"),
    ("Ճամփորդություն",              "Travel & countries — world cultures, famous cities, travel tips, tourism"),
    ("Ֆիլոսոֆիա ու արժեքներ",      "Philosophy & values — kindness, honesty, justice, friendship, ethics for teens"),
]

BATCH_SIZE = 15              # Q&A pairs per API call (dropped from 20 because
                             # longer average answers means each batch produces
                             # more output tokens; 15 keeps us under max_tokens)
# Qwen3-Next-80B-A3B-Instruct is an explicit NON-thinking instruct variant
# (no need for chat_template_kwargs.enable_thinking toggling). It's an MoE
# with 80 B total / 3 B active params, so per-request serving cost is low
# and DeepInfra's queue for it is much shorter than Qwen3.5-27B which is
# heavily oversubscribed. Tested with Armenian prompts — fluent output.
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"

# Target block_size of downstream training. Training config xxlarge uses
# block_size=2048. We shape the answer-length distribution so generated
# pairs exercise the full context window without risking truncation.
TRAINING_BLOCK_SIZE = 2048

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(_REPO_ROOT, "data", "text", "finetune", "armenian_qa_qwen.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
PROGRESS_FILE = OUTPUT_FILE.replace(".json", "_progress.json")


def make_prompt(topic_name: str, topic_desc: str, batch_num: int, total_batches: int) -> str:
    # Format: JSONL — ONE JSON object per line, no outer array. If one line
    # has unescaped quotes or other glitches we still recover the other
    # 14 pairs instead of losing the whole batch (which is what happens
    # when a single pair breaks a JSON array parse).
    return f"""Դու հայ կրթական փորձագետ ես։ Ստեղծիր {BATCH_SIZE} հարց-պատասխան զույգ «{topic_name}» թեմայով։

Թեմայի նկարագրություն (անգլերեն): {topic_desc}

Պահանջներ.
- Հարցերն ու պատասխանները պետք է լինեն արևելահայերենով (Arevelahayeren)
- Նախատեսված է լայն հանդիսատեսի՝ ուսանողներ և հետաքրքրված մեծահասակներ
- Հարցերը պետք է բազմազան լինեն ձևով ու խորությամբ (batch {batch_num}/{total_batches}).
  * Մի քանիսը՝ կարճ փաստական («Ի՞նչ է...», «Ո՞վ էր...», «Ե՞րբ...»)
  * Մի քանիսը՝ բացատրական («Ինչո՞ւ...», «Ինչպե՞ս...», «Որո՞նք են տարբերությունները...»)
  * Մի քանիսը՝ վերլուծական կամ մանրամասն («Նկարագրիր...», «Համեմատիր...», «Բացատրիր քայլ առ քայլ...»)
- Պատասխանների երկարությունը ՊԵՏՔ Է տարբեր լինի.
  * 30%-ը՝ ԿԱՐՃ (2-3 նախադասություն)
  * 40%-ը՝ ՄԻՋԻՆ (5-7 նախադասություն, բացատրական)
  * 30%-ը՝ ԵՐԿԱՐ (10-15 նախադասություն՝ օրինակներով, մանրամասն)

ՁԵՎԱՉԱՓ (շատ կարևոր).
- Վերադարձրու ճիշտ {BATCH_SIZE} JSON օբյեկտ, ԱՄԵՆ ՄԵԿԸ ԱՌԱՆՁԻՆ ՏՈՂՈՒՄ (JSONL, not JSON array)
- Ամեն տող պետք է լինի. {{"question": "...", "answer": "..."}}
- Եթե պատասխանի ներսում օգտագործում ես չակերտ, ՓԱԽՑՐՈՒ այն որպես \\"
- Մի՛ օգտագործիր ``` markdown fence, մի՛ ավելացնիր բացատրություններ, միայն JSONL
- Ամեն տող պետք է լինի ինքնուրույն վավեր JSON օբյեկտ

Օրինակ ճշգրիտ ձևաչափի (your output should look exactly like this, with {BATCH_SIZE} lines):
{{"question": "Ո՞վ է Մեսրոպ Մաշտոցը:", "answer": "Մեսրոպ Մաշտոցը 5-րդ դարի հայ մանկավարժ և գիտնական էր, որ 405 թվականին ստեղծեց հայկական այբուբենը:"}}
{{"question": "Ի՞նչ էր Ուրարտուն:", "answer": "Ուրարտուն մ.թ.ա. 9-6-րդ դարերում Հայկական լեռնաշխարհում գոյություն ունեցած հզոր պետություն էր, որը համարվում է հայկական պետականության նախահիմքերից մեկը:"}}"""


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_batches": [], "pairs": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


_QA_REGEX = None  # compiled lazily on first use


def _append_pair(result, q, a):
    """Normalize + validate a (question, answer) tuple and append if usable."""
    q = (q or "").strip().strip('"').strip()
    a = (a or "").strip().strip('"').strip()
    if q and a and len(a) > 20:
        result.append({"instruction": q, "input": "", "output": a})


def _parse_pairs(raw: str) -> list:
    """Robust parser for Qwen's Q&A output.

    Handles four tiers of formatting in order of preference:
      1. JSONL (one `{"question": "...", "answer": "..."}` per line) —
         the format the prompt explicitly asks for. Line-by-line parsing
         means a single broken line drops one pair instead of the whole batch.
      2. JSON array — legacy / fallback when the model disobeys the prompt
         and wraps everything in `[...]`.
      3. Regex sweep for `"question": "..." , "answer": "..."` blocks —
         catches outputs where the JSON grammar is broken (e.g. unescaped
         quotes inside answers) but the key/value pattern is still visible.
      4. Newline-delimited scan for lines starting with `{` and parseable
         in isolation.

    Returns a list of {instruction, input, output} dicts. Returns an empty
    list if nothing could be extracted (caller should log a parse error).
    """
    import re

    raw = raw.strip()

    # Strip ``` markdown fences if the model included them despite instructions.
    if raw.startswith("```"):
        # Remove the opening fence + optional language tag
        raw = re.sub(r"^```(?:json|jsonl)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

    result = []

    # ── Tier 1: JSONL (preferred format from the prompt) ──────────────────
    # Try each non-empty line as a standalone JSON object. Drop broken lines.
    jsonl_parsed = 0
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        q = obj.get("question") or obj.get("instruction") or ""
        a = obj.get("answer") or obj.get("output") or ""
        prev = len(result)
        _append_pair(result, q, a)
        if len(result) > prev:
            jsonl_parsed += 1

    if jsonl_parsed >= 3:
        # Decent JSONL hit rate — trust this path and return.
        return result

    # ── Tier 2: JSON array (legacy) ───────────────────────────────────────
    # Cut from first [ to last ].
    if "[" in raw and "]" in raw:
        lb, rb = raw.find("["), raw.rfind("]")
        if lb != -1 and rb > lb:
            try:
                arr = json.loads(raw[lb:rb + 1])
                if isinstance(arr, list):
                    tier2 = []
                    for p in arr:
                        if isinstance(p, dict):
                            q = p.get("question") or p.get("instruction") or ""
                            a = p.get("answer") or p.get("output") or ""
                            _append_pair(tier2, q, a)
                    if len(tier2) > len(result):
                        return tier2
            except json.JSONDecodeError:
                pass

    # ── Tier 3: regex sweep for `"question": "X", "answer": "Y"` blocks ──
    # Handles outputs with unescaped quotes or other structural damage.
    # Non-greedy match, requires the two keys in order.
    global _QA_REGEX
    if _QA_REGEX is None:
        _QA_REGEX = re.compile(
            r'"(?:question|instruction)"\s*:\s*"(.+?)"\s*,\s*'
            r'"(?:answer|output)"\s*:\s*"(.+?)"(?=\s*[},\n])',
            re.DOTALL,
        )
    tier3 = []
    for q, a in _QA_REGEX.findall(raw):
        _append_pair(tier3, q, a)
    if len(tier3) > len(result):
        result = tier3

    # ── Tier 4: still nothing — try each line individually with relaxed parse ─
    if not result:
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            # Find the matching closing brace of a well-formed object
            depth = 0
            end = -1
            in_str = False
            esc = False
            for i, ch in enumerate(line):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                candidate = line[:end]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        q = obj.get("question") or obj.get("instruction") or ""
                        a = obj.get("answer") or obj.get("output") or ""
                        _append_pair(result, q, a)
                except json.JSONDecodeError:
                    continue

    return result


def generate_batch(client, topic_name: str, topic_desc: str,
                   batch_num: int, total_batches: int) -> list:
    """Call Qwen3.5-27B via DeepInfra, return list of {instruction, input, output}."""
    prompt = make_prompt(topic_name, topic_desc, batch_num, total_batches)

    # Qwen3-Next-80B-A3B-Instruct is a non-thinking variant — no need to
    # toggle enable_thinking via chat_template_kwargs.
    #
    # max_tokens is sized for the longest plausible batch output. A batch
    # of 15 pairs with the mixed-length distribution (30/40/30 short/med/
    # long) produces ~6-9k output tokens typically; 16k gives headroom
    # for outlier long-answer batches without truncation.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=16000,
    )
    raw = response.choices[0].message.content or ""
    pairs = _parse_pairs(raw)
    if not pairs:
        # Parser couldn't recover anything — treat as a parse error so the
        # retry loop in the caller gets a consistent exception signal.
        raise json.JSONDecodeError(
            "parser extracted 0 pairs from non-empty response", raw[:200], 0
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Armenian Q&A pairs via DeepInfra Qwen3.5-27B"
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Target total Q&A pairs to generate (default: 1000)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        print("Error: DEEPINFRA_API_KEY not set!")
        print("  export DEEPINFRA_API_KEY=<your key>")
        print("  Get one at https://deepinfra.com/dash/api_keys")
        sys.exit(1)

    try:
        import openai
    except ImportError:
        print("Error: openai package not installed.")
        print("  pip install openai")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    # Distribute the target across topics as evenly as possible. Each
    # topic gets roughly count/len(TOPICS) pairs, rounded up to the
    # next multiple of BATCH_SIZE so every API call returns a full batch.
    target = args.count
    pairs_per_topic = max(BATCH_SIZE, ((target + len(TOPICS) - 1) // len(TOPICS)))
    # Round up to whole batches
    pairs_per_topic = ((pairs_per_topic + BATCH_SIZE - 1) // BATCH_SIZE) * BATCH_SIZE
    batches_per_topic = pairs_per_topic // BATCH_SIZE
    total_batches = len(TOPICS) * batches_per_topic
    effective_target = total_batches * BATCH_SIZE

    # Rough per-batch cost at DeepInfra pricing for Qwen3-Next-80B-A3B:
    # ~500 in + ~8k out tokens per batch, ~$0.0023 per batch.
    est_cost_per_batch = 0.0023

    print(f"\n{'='*55}")
    print(f"  Armenian Q&A Generator (DeepInfra {MODEL.split('/')[-1]})")
    print(f"{'='*55}")
    print(f"  Topics:        {len(TOPICS)}")
    print(f"  Per topic:     {pairs_per_topic} ({batches_per_topic} batches × {BATCH_SIZE})")
    print(f"  Target pairs:  {effective_target:,}  (requested: {target:,})")
    print(f"  API calls:     {total_batches}")
    print(f"  Length mix:    30% short / 40% medium / 30% long-form")
    print(f"  Training ctx:  {TRAINING_BLOCK_SIZE} tokens (xxlarge preset)")
    print(f"  Model:         {MODEL}")
    print(f"  Endpoint:      {BASE_URL}")
    print(f"  Est. cost:     ~${total_batches * est_cost_per_batch:.2f}  "
          f"(~${est_cost_per_batch:.4f} per batch)")
    print(f"  Output:        {OUTPUT_FILE}")
    print(f"{'='*55}\n")

    progress = load_progress()
    completed = set(progress["completed_batches"])
    all_pairs = progress["pairs"]
    print(f"Resuming: {len(completed)}/{total_batches} batches done, "
          f"{len(all_pairs):,} pairs so far\n")

    # Build the work queue of outstanding batches, skipping any already
    # marked complete in the progress file. Submit them to a thread pool
    # so multiple DeepInfra API calls fly in parallel.
    work = []
    for topic_name, topic_desc in TOPICS:
        for batch_idx in range(batches_per_topic):
            batch_key = f"{topic_name}::{batch_idx}"
            if batch_key not in completed:
                work.append((topic_name, topic_desc, batch_idx, batch_key))

    print(f"Dispatching {len(work)} remaining batches to a thread pool...")
    print()

    # Shared state — guarded by a lock. The main thread just drains
    # as_completed; the per-batch function does a full retry loop inside.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    state_lock = threading.Lock()
    # Mutable counters
    fatal_errors = 0
    finished = 0
    t_start = time.time()

    def run_one_batch(topic_name, topic_desc, batch_idx, batch_key):
        """Retry-forever loop for a single batch. Runs inside a worker thread.

        Returns ("ok", pairs) on success, ("fatal", error_str) on non-transient
        failure or exhausted retries. Transient errors (429 / 5xx / timeouts)
        trigger exponential backoff inside this function.
        """
        strikes = 0
        max_transient_strikes = 8  # ~15+30+60+120+240+300+300+300 ≈ 23 min per batch ceiling
        while True:
            try:
                pairs = generate_batch(
                    client, topic_name, topic_desc,
                    batch_idx + 1, batches_per_topic,
                )
                return ("ok", pairs, strikes)
            except json.JSONDecodeError as e:
                return ("parse_error", str(e), strikes)
            except Exception as e:
                msg = str(e).lower()
                transient = any(
                    tok in msg for tok in
                    ("rate", "429", "overloaded", "busy", "500", "502", "503", "504",
                     "inference error", "timeout")
                )
                if transient and strikes < max_transient_strikes:
                    strikes += 1
                    backoff = min(15 * (2 ** (strikes - 1)), 300)
                    time.sleep(backoff)
                    continue
                return ("fatal", str(e), strikes)

    max_parallel = 5  # DeepInfra tolerates 5-8 concurrent reqs per account
                      # for mid-size models without triggering burst 429s.

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_one_batch, t, d, i, k): (t, d, i, k)
            for (t, d, i, k) in work
        }
        for future in as_completed(futures):
            topic_name, topic_desc, batch_idx, batch_key = futures[future]
            status, payload, strikes = future.result()
            finished += 1

            if status == "ok":
                with state_lock:
                    all_pairs.extend(payload)
                    completed.add(batch_key)
                    progress["completed_batches"] = list(completed)
                    progress["pairs"] = all_pairs
                    save_progress(progress)
                label = f"✓ {len(payload)} pairs"
                extra = f" ({strikes} retries)" if strikes else ""
            elif status == "parse_error":
                label = f"✗ JSON parse error: {payload[:80]}"
                extra = ""
            else:  # fatal
                with state_lock:
                    fatal_errors += 1
                label = f"✗ fatal: {payload[:80]}"
                extra = ""

            elapsed = time.time() - t_start
            rate = finished / elapsed * 60 if elapsed > 0 else 0
            print(
                f"[{finished:3d}/{len(work)}] {topic_name[:28]:<28} "
                f"b{batch_idx+1}/{batches_per_topic}  {label}{extra}  "
                f"(total: {len(all_pairs):,} pairs, {rate:.1f}/min)"
            )

            if fatal_errors >= 20:
                print(f"\n{fatal_errors} fatal errors. Cancelling remaining work.")
                for f in futures:
                    f.cancel()
                break

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Generation Complete!")
    print(f"{'='*55}")
    print(f"  Total pairs:  {len(all_pairs):,}")
    print(f"  Wall time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved to:     {OUTPUT_FILE}")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print(f"\nSample Q&A (first 3):")
    for i, pair in enumerate(all_pairs[:3]):
        q = pair["instruction"]
        a = pair["output"]
        print(f"\n  [{i+1}] Q: {q[:100]}{'...' if len(q) > 100 else ''}")
        print(f"       A: {a[:150]}{'...' if len(a) > 150 else ''}")

    print(f"\nNext step: review quality, then feed to prepare_chat.py or merge "
          f"with {os.path.basename(OUTPUT_FILE.replace('_qwen', ''))}")


if __name__ == "__main__":
    main()
