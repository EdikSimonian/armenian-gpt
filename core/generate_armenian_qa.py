"""
Armenian Q&A Dataset Generator
Uses Claude API to generate 5,000 hand-crafted Armenian Q&A pairs.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python data/generate_armenian_qa.py

Output:
    data/armenian_qa.json  — 5,000 {instruction, output} pairs ready for prepare_chat.py

Cost estimate: ~$0.80 with claude-haiku-4-5 (250 calls × 20 pairs each)
Resume: Safe to interrupt and re-run — resumes from last completed batch.
"""

import os
import sys
import json
import time

# ── Topics ──────────────────────────────────────────────────────────────────
# 25 topics × 200 Q&A each = 5,000 total
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

PAIRS_PER_TOPIC = 200      # 25 × 200 = 5,000 total
BATCH_SIZE = 20            # Q&A pairs per API call
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(_REPO_ROOT, "data", "text", "finetune", "armenian_qa.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
PROGRESS_FILE = OUTPUT_FILE.replace(".json", "_progress.json")


def make_prompt(topic_name: str, topic_desc: str, batch_num: int, total_batches: int) -> str:
    return f"""Դու հայ կրթական փորձագետ ես։ Ստեղծիր {BATCH_SIZE} հարց-պատասխան զույգ «{topic_name}» թեմայով։

Թեմայի նկարագրություն (անգլերեն): {topic_desc}

Պահանջներ.
- Հարցերն ու պատասխանները պետք է լինեն արևելահայերենով (Arevelahayeren)
- Նախատեսված է 12-18 տարեկան աշակերտների համար
- Հարցերը պետք է հետաքրքիր, բովանդակալից և տարբեր լինեն (batch {batch_num}/{total_batches})
- Պատասխանները՝ 2-4 նախադասություն, հստակ ու ճշգրիտ
- Բովանդակությունը պետք է լինի տարիքին համապատասխան ու անվտանգ
- Օգտագործիր բնական, ճիշտ հայերեն քերականություն

Վերադարձրու JSON ձևաչափով (ONLY the JSON array, no other text):
[
  {{"question": "Հայկական հարցը...", "answer": "Հայկական հարցը..."}},
  ...
]"""


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_batches": [], "pairs": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def generate_batch(client, topic_name: str, topic_desc: str, batch_num: int, total_batches: int) -> list:
    """Call Claude API and return list of {instruction, output} dicts."""
    prompt = make_prompt(topic_name, topic_desc, batch_num, total_batches)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Extract JSON if wrapped in markdown code block
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    pairs_raw = json.loads(raw)

    # Normalize to {instruction, output} format used by prepare_chat.py
    result = []
    for p in pairs_raw:
        q = p.get("question", p.get("instruction", "")).strip()
        a = p.get("answer", p.get("output", "")).strip()
        if q and a and len(a) > 20:
            result.append({"instruction": q, "input": "", "output": a})
    return result


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set!")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed.")
        print("  pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Plan: for each topic, ceil(PAIRS_PER_TOPIC / BATCH_SIZE) calls
    batches_per_topic = PAIRS_PER_TOPIC // BATCH_SIZE  # 10
    total_batches = len(TOPICS) * batches_per_topic     # 250

    print(f"\n{'='*55}")
    print(f"  Armenian Q&A Generator")
    print(f"{'='*55}")
    print(f"  Topics:       {len(TOPICS)}")
    print(f"  Per topic:    {PAIRS_PER_TOPIC}")
    print(f"  Total target: {len(TOPICS) * PAIRS_PER_TOPIC:,}")
    print(f"  API calls:    {total_batches}")
    print(f"  Model:        claude-haiku-4-5-20251001")
    print(f"  Est. cost:    ~$0.80")
    print(f"  Output:       {OUTPUT_FILE}")
    print(f"{'='*55}\n")

    progress = load_progress()
    completed = set(progress["completed_batches"])
    all_pairs = progress["pairs"]
    print(f"Resuming: {len(completed)}/{total_batches} batches done, {len(all_pairs):,} pairs so far\n")

    call_num = 0
    errors = 0

    for topic_name, topic_desc in TOPICS:
        for batch_idx in range(batches_per_topic):
            batch_key = f"{topic_name}::{batch_idx}"
            call_num += 1

            if batch_key in completed:
                continue  # Already done

            print(f"[{call_num:3d}/{total_batches}] {topic_name[:30]:<30} batch {batch_idx+1}/{batches_per_topic} ...", end="", flush=True)

            try:
                pairs = generate_batch(client, topic_name, topic_desc, batch_idx + 1, batches_per_topic)
                all_pairs.extend(pairs)
                completed.add(batch_key)
                progress["completed_batches"] = list(completed)
                progress["pairs"] = all_pairs
                save_progress(progress)
                print(f" ✓ {len(pairs)} pairs  (total: {len(all_pairs):,})")
                errors = 0

            except json.JSONDecodeError as e:
                print(f" ✗ JSON parse error: {e}")
                errors += 1

            except Exception as e:
                print(f" ✗ Error: {e}")
                errors += 1
                if "rate_limit" in str(e).lower() or "overloaded" in str(e).lower():
                    print("    Rate limit hit — waiting 30s...")
                    time.sleep(30)

            if errors >= 5:
                print("\nToo many consecutive errors. Check your API key and try again.")
                sys.exit(1)

            # Small delay to be nice to the API
            time.sleep(0.3)

    # Save final output in prepare_chat.py compatible format
    print(f"\n{'='*55}")
    print(f"  Generation Complete!")
    print(f"{'='*55}")
    print(f"  Total pairs:  {len(all_pairs):,}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"  Saved to:     {OUTPUT_FILE}")

    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    # Show sample
    print(f"\nSample Q&A (first 3):")
    for i, pair in enumerate(all_pairs[:3]):
        print(f"\n  [{i+1}] Q: {pair['instruction'][:80]}...")
        print(f"       A: {pair['output'][:100]}...")

    print(f"\nNext step: python data/prepare_chat.py --source data/armenian_qa.json")


if __name__ == "__main__":
    main()
