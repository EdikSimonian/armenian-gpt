# ArmGPT 5-Day Workshop

**Build your own Armenian AI from scratch.**

Audience: Students ages 12-18
Schedule: 5 days, 4 hours per day (20 hours total)
Hardware: 1x RTX 4090 (shared or one per group)
Goal: By the end, students have a working Armenian chatbot they trained themselves.

---

## Overview

| Day | Theme | Hands-On | Training Status |
|-----|-------|----------|-----------------|
| 1 | What is AI? How does it learn? | Setup, download data, train tiny model | tiny model (1 min) |
| 2 | Tokens, numbers, and patterns | Explore tokenizer, start xlarge training | xlarge starts (runs overnight) |
| 3 | Inside the Transformer | Monitor training, understand loss curves | xlarge running (~16 hrs in) |
| 4 | Teaching AI to talk | Fine-tune for chat, test conversations | fine-tune (~1 hr) |
| 5 | Your AI, your rules | Experiments, demo day, what's next | show off results |

> **Key insight:** The xlarge model takes ~8 hours on a 4090. We start it at the end of Day 2 and let it train overnight. By Day 3 morning it's done.

---

## Day 1: What is AI? (4 hours)

### Hour 1: What is a language model?

**Concepts to cover:**
- AI is pattern recognition, not "thinking"
- A language model predicts the next word (like phone autocomplete, but bigger)
- Show the live demo: https://huggingface.co/spaces/edisimon/armgpt-demo
- The entire pipeline in one slide:
  ```
  Armenian text -> Numbers -> Learn patterns -> Generate new text
  ```

**Activity:** Play "predict the next word" as a class.
- Write an Armenian sentence on the board, stop mid-word
- Students guess what comes next
- "You just did what a language model does!"

### Hour 2: Setup and get the data

**Do together:**
```bash
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt
pip install numpy sentencepiece huggingface_hub zstandard
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Explain while installing:**
- What is Python? (a language computers understand)
- What is a GPU? (a chip that does millions of math problems at once)
- What is training data? (examples the AI learns from)

**Download the tokenized data:**
```bash
export HF_TOKEN=hf_your_class_token
python 1_download.py --tokenized-only
```

**While it downloads (~10 min), explain:**
- We collected 63 GB of Armenian text from Wikipedia, news, books
- We cleaned it: removed junk, duplicates, non-Armenian text
- We turned it into numbers (tokens) - 8.3 billion of them
- That's what's downloading now: the numbers, ready to learn from

### Hour 3: Train your first model

**Train the tiny model together (takes ~1 minute):**
```bash
python 4_train.py --preset tiny --tokenizer bpe
```

**Watch the output together, explain each line:**
- `loss 4.40` - The model is guessing randomly (bad)
- `loss 3.20` - It's learning some patterns (better)
- `loss 2.80` - It knows common Armenian letter combinations (good)
- `tok/s` - How fast it's learning (tokens per second)

**Generate text from the tiny model:**
```bash
python 5_generate.py --prompt "Hayastan"
```

**Discussion:** "The text looks like garbage. Why?"
- The tiny model only has 200,000 parameters (like a brain with 200K neurons)
- It only trained for 1,000 steps
- GPT-4 has 1,000,000,000,000+ parameters and trained for months
- But the basic idea is exactly the same!

### Hour 4: Understanding size and scale

**Train the small model (takes ~30 min on 4090):**
```bash
python 4_train.py --preset small --tokenizer bpe
```

**While it trains, cover:**

| Human Brain | tiny model | small model | xlarge model | GPT-4 |
|-------------|-----------|-------------|-------------|-------|
| 86 billion neurons | 200K params | 10M params | 350M params | 1.8T params |

- More parameters = more patterns it can learn
- More training data = more patterns to learn from
- More training time = better at using those patterns

**Activity:** Compare tiny vs small generated text:
```bash
python 5_generate.py --checkpoint checkpoints/final.pt --prompt "Hayastan"
```

"See the difference? More parameters + more training = better Armenian."

**Homework concept:** "Tonight the computer will train a much bigger model. Tomorrow we'll see what 350 million parameters can do."

---

## Day 2: Tokens and Patterns (4 hours)

### Hour 1: How computers read Armenian

**The big question:** "Computers only understand numbers. How do we turn Armenian text into numbers?"

**Explain tokenization:**
```
"Hayastan" is not one word to the computer.

Character level:  H -> 1, a -> 2, y -> 3, ... (one number per letter)
BPE level:        "Hay" -> 234, "astan" -> 1523 (groups of letters)
```

**Hands-on: Explore the tokenizer**
```bash
python -c "
from core.bpe_tokenizer import BPETokenizer
tok = BPETokenizer()
tok.load('data/bpe_model.model')

# Try encoding Armenian words
words = ['Hayastan', 'Yerevan', 'barev']
for w in words:
    ids = tok.encode(w)
    print(f'{w} -> {ids} ({len(ids)} tokens)')
    print(f'  decoded back: {tok.decode(ids)}')
"
```

**Activity:** Students try encoding their own names and Armenian words. Who has the shortest token representation? Who has the longest?

**Key concepts:**
- Common words = fewer tokens (the model learned they go together)
- Rare words = more tokens (broken into smaller pieces)
- Our tokenizer has 16,000 tokens (like a 16,000-word dictionary of Armenian pieces)

### Hour 2: What is learning?

**Explain with an analogy:**
- Imagine you just moved to Armenia and don't speak Armenian
- Day 1: You hear sounds but can't predict what comes next
- Day 30: You recognize common words
- Day 365: You can finish someone's sentence
- That's exactly what the model does, but in hours instead of years

**The training loop (simplified):**
1. Show the model a sentence: "Hayastani mayraqaghaq@ ..."
2. Model guesses: "X" (random at first)
3. Correct answer was: "Yerevan"
4. Model adjusts its weights to be less wrong
5. Repeat 36,000 times

**Loss = how wrong the model is:**
- Loss 4.5 = random guessing
- Loss 3.0 = knows common patterns
- Loss 2.0 = writes decent Armenian
- Loss 1.5 = hard to tell from human text

### Hour 3: The Transformer (simplified)

**Draw on the board:**
```
Input: "Hayastani mayraqaghaq@ ..."

Step 1: EMBEDDING
  Each token becomes a list of 1024 numbers
  (like giving each word a unique "fingerprint")

Step 2: ATTENTION (the key idea!)
  Each word looks at all previous words and asks:
  "Which words are important for predicting what comes next?"
  
  "mayraqaghaq@" pays attention to "Hayastani"
  because the capital depends on which country!

Step 3: FEED-FORWARD
  Each word "thinks" about what it learned from attention

Step 4: Repeat steps 2-3 twenty-four times (24 layers)

Step 5: OUTPUT
  Predict the next token: "Yerevan@" (92% confident)
```

**Activity:** "Attention game"
- Write a sentence with a blank: "Hayastani ____ @ Yerevan @"
- Which words in the sentence help you fill in the blank?
- That's attention! The model learns which words to focus on.

### Hour 4: Start the real training

**The moment of truth - start the xlarge model:**
```bash
python 4_train.py --preset xlarge --tokenizer bpe
```

**Watch the first 100 steps together. Explain:**
- 350 million parameters being adjusted
- 1024-dimensional vectors (each token is a point in 1024D space)
- 24 transformer layers deep
- Processing ~150,000 tokens per second on the 4090

**Show the math:**
- 36,000 steps x ~220,000 tokens/step = ~8 billion tokens seen
- That's every token in our dataset once!
- Takes about 8 hours. It'll run overnight.

**Leave it running:**
```bash
# Check on it anytime with:
# tail -5 checkpoints/metrics.json
```

---

## Day 3: Inside the Training (4 hours)

> The xlarge model should be done or nearly done by morning (~8 hrs overnight).

### Hour 1: Check our model

**Check if training finished:**
```bash
ls checkpoints/final.pt
```

**If still running, check progress:**
```bash
# See the latest loss
tail -1 checkpoints/metrics.json
```

**Generate text with the xlarge model:**
```bash
python 5_generate.py --checkpoint checkpoints/final.pt --prompt "Hayastan" --length 500
```

**Compare all three models side by side:**
```bash
# Tiny (200K params, 1 min training)
python 5_generate.py --checkpoint checkpoints/step_1000.pt --prompt "Hayastan" --length 200

# Small (10M params, 30 min training)  
# (use small checkpoint if you saved it on Day 1)

# XLarge (350M params, 8 hrs training)
python 5_generate.py --checkpoint checkpoints/final.pt --prompt "Hayastan" --length 200
```

**Discussion:** What changed? The words are real. The sentences make sense. The grammar is correct. But does it "understand" Armenian? (Spoiler: no - it's pattern matching.)

### Hour 2: Understanding loss curves

**Open the metrics file and plot:**
```python
import json
import matplotlib.pyplot as plt

with open("checkpoints/metrics.json") as f:
    m = json.load(f)

plt.figure(figsize=(10, 5))
plt.plot(m["steps"], m["train_loss"], label="Train loss")
plt.plot(m["steps"], m["val_loss"], label="Val loss")
plt.xlabel("Training step")
plt.ylabel("Loss (lower = better)")
plt.title("ArmGPT Learning Curve")
plt.legend()
plt.grid(True)
plt.show()
```

**Explain the curve:**
- Steep drop at the start = learning basic patterns fast
- Gradual decline = learning subtler patterns
- Train vs val gap = overfitting (memorizing vs learning)
- Flat line at the end = model reached its capacity

**Key idea: perplexity**
- Perplexity = e^loss
- Perplexity of 10 means: at each step, the model is choosing between ~10 equally likely options
- Perplexity of 3 means: almost sure about the next word
- Human-level Armenian writing: perplexity ~20-50

### Hour 3: Temperature and creativity

**Experiment together:**
```bash
# Very focused (repetitive, safe)
python 5_generate.py --temperature 0.2 --length 300

# Balanced
python 5_generate.py --temperature 0.7 --length 300

# Creative (surprising, sometimes wrong)
python 5_generate.py --temperature 1.2 --length 300

# Chaos
python 5_generate.py --temperature 2.0 --length 300
```

**Explain:**
- Temperature 0.1 = always pick the most likely next word (boring but correct)
- Temperature 1.0 = sample normally from the probabilities
- Temperature 2.0 = make unlikely words more likely (creative but messy)

**Activity:** Students find their favorite temperature for different tasks:
- Writing a Wikipedia article? (low temperature)
- Writing a poem? (higher temperature)
- Writing something funny? (high temperature)

### Hour 4: How AI can be wrong

**Important discussion for young people:**

1. **AI doesn't know truth.** It knows patterns. If the training data says "Yerevan@ Hayastani mayraqaghaq@ e" enough times, it learns the pattern. But it could just as easily learn wrong patterns from wrong data.

2. **Bias in data = bias in AI.** If Wikipedia has more articles about some topics than others, the model knows more about those topics.

3. **AI can't think.** Ask it a math question - it will pattern-match, not calculate. Ask it about 2026 - it only knows what was in the training data.

**Activity:** Try to "trick" the model:
- Ask it something factually wrong in a confident way
- Ask it about very recent events
- Ask it to do math
- See how it responds

---

## Day 4: Teaching AI to Talk (4 hours)

### Hour 1: From autocomplete to chatbot

**The problem:**
- Our model continues text (autocomplete)
- We want it to answer questions (chat)
- Solution: show it thousands of question-answer examples

**Explain the format:**
```
<|user|>Inch e Hayastani mayraqaghaq@?<|end|><|assistant|>Hayastani mayraqaghaq@ Yerevan@ e:<|end|>
```

- `<|user|>` = "a human is speaking"
- `<|assistant|>` = "now the AI should respond"
- `<|end|>` = "this person is done talking"

**These are special tokens we add to the vocabulary:**
- Base vocabulary: 16,000 tokens (Armenian text pieces)
- Extended vocabulary: 16,003 tokens (+3 special chat tokens)

### Hour 2: Prepare chat data and fine-tune

**Download and prepare Q&A data:**
```bash
python 1_download.py --download --qa
python 2_prepare.py --qa
python 3_tokenize.py --qa --tokenizer bpe
```

**Explore the data:**
```python
import json
with open("data/text/finetune/qa_merged.json") as f:
    data = json.load(f)
print(f"Total Q&A pairs: {len(data)}")
print(f"\nExample:")
print(f"Q: {data[0]['instruction']}")
print(f"A: {data[0]['output'][:200]}")
```

**Start fine-tuning (~1 hour):**
```bash
python 6_finetune.py --tokenizer bpe
```

**While it trains, explain:**
- We're NOT training from scratch
- We're taking the model that already knows Armenian (8 hours of training)
- And teaching it the question-answer format (1 hour of fine-tuning)
- This is called "transfer learning" - reuse what you already learned

### Hour 3: Chat with your model

**When fine-tuning finishes:**
```bash
python 8_chat.py
```

**Try these together:**
- Simple facts: "Inch e Hayastani mayraqaghaq@?"
- Explanations: "Inchpes e ashxatum arcevi ijejn?"
- Creative: "Grir mi karjr patmutjyun Ararati masin"

**Discussion:** What does it do well? What does it get wrong?

**Compare base vs fine-tuned:**
```bash
# Base model (autocomplete - continues the text)
python 5_generate.py --prompt "Inch e Hayastani mayraqaghaq@"

# Fine-tuned model (answers the question)
python 8_chat.py
# > Inch e Hayastani mayraqaghaq@?
```

### Hour 4: How real AI products are built

**The three stages of building ChatGPT:**
1. **Pretraining** (what we did in 8 hrs) - Learn language from billions of words. OpenAI uses trillions.
2. **Fine-tuning** (what we did in 1 hr) - Learn to follow instructions. OpenAI uses millions of examples.
3. **RLHF** (we didn't do this) - Humans rate responses, model learns to give better ones.

**Scale comparison:**

| | ArmGPT | GPT-4 |
|---|---|---|
| Parameters | 350M | ~1.8T (5000x more) |
| Training data | 63 GB | ~13 TB (200x more) |
| Training time | 8 hours | ~3 months |
| Training cost | ~$3 | ~$100 million |
| Languages | Armenian only | 100+ languages |

"We built a miniature version of the same technology. The ideas are identical. The difference is scale."

---

## Day 5: Experiments and Demo Day (4 hours)

### Hour 1: Student experiments

**Each student/group picks one experiment:**

**Experiment A: Temperature artist**
- Generate Armenian text at 10 different temperatures
- Find the best temperature for: a Wikipedia article, a poem, a story
- Present your findings

**Experiment B: Size matters**
- Compare tiny, small, and xlarge outputs for the same 5 prompts
- At what size does the model start making sense?
- How many parameters do you need for grammar?

**Experiment C: Prompt engineer**
- Find the best way to ask the chat model questions
- Does it work better with short or long questions?
- What topics does it know best? Worst?
- Can you make it write a poem?

**Experiment D: Break the AI**
- Find the model's limits
- What questions confuse it?
- Can you make it contradict itself?
- What happens with nonsense input?

### Hour 2: Work on experiments

Students work on their experiments. Teacher circulates, helps, answers questions.

### Hour 3: Presentations

Each group presents their findings (5-10 min each):
- What did you try?
- What did you discover?
- What surprised you?

### Hour 4: What's next + wrap-up

**Where to go from here:**
- **Python:** Learn more at python.org - everything in AI starts with Python
- **Machine Learning:** Fast.ai (free course, very practical)
- **Transformers:** Watch Andrej Karpathy's "Let's Build GPT" on YouTube
- **Armenian NLP:** There's so much work to do for Armenian language technology

**Big ideas to take away:**
1. AI is math, not magic. You just built one.
2. Data matters more than algorithms. Garbage in = garbage out.
3. Scale changes everything. Same idea, bigger = dramatically better.
4. AI doesn't understand - it predicts. Knowing the difference matters.
5. The people who build AI get to decide what it does. That could be you.

**Certificates / show parents / deploy to HuggingFace Space.**

---

## Teacher Notes

### Before the workshop
- Set up the machine with CUDA, Python, and all dependencies
- Pre-download the tokenized data (`python 1_download.py --tokenized-only`)
- Test that training works: `python 4_train.py --preset tiny --tokenizer bpe`
- Have the HF_TOKEN ready as an environment variable
- Print copies of the pipeline diagram for Day 1

### Timing the xlarge training
- Start xlarge training at the **end of Day 2** (last 15 min)
- It runs for ~8 hours overnight on the 4090
- By Day 3 morning (16+ hours later) it will be done
- If it's not done, students can watch live training progress (also interesting)
- **Backup plan:** If the 4090 isn't available, use `large` preset (~3 hrs) and start it at lunch on Day 2

### If things go wrong
- `CUDA out of memory`: Reduce batch_size (`--batch_size 16`)
- Training interrupted: Resume with `--resume_from checkpoints/step_XXXXX.pt`
- Model generates garbage: Lower the temperature (`--temperature 0.3`)
- Slow download: Pre-download everything before the workshop starts

### Adapting for different ages
- **Ages 12-14:** Focus on the "what" (demos, experiments, discussion). Skip the math. More time on activities.
- **Ages 15-16:** Include the "how" (attention, embeddings, loss). Show code snippets.
- **Ages 17-18:** Include the "why" (gradients, backprop, cross-entropy). Let them modify code.
