# ArmGPT - Build Your Own Armenian Language Model

A simple GPT language model trained on Armenian text. Built for students (ages 12-18) to learn how AI language models work by reading, running, and modifying the code.

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Download the Data](#step-1-download-the-data)
  - [Step 2: Prepare the Data](#step-2-prepare-the-data)
  - [Step 3: Train the Model](#step-3-train-the-model)
  - [Step 4: Generate Text](#step-4-generate-text)
- [How It Works (The Concepts)](#how-it-works-the-concepts)
- [Tokenization Levels](#tokenization-levels)
- [Training Data Sources](#training-data-sources)
- [Model Presets](#model-presets)
- [All Command-Line Options](#all-command-line-options)
- [Performance Metrics](#performance-metrics)
- [Training on the Cloud](#training-on-the-cloud)
- [Resuming Training](#resuming-training)
- [Troubleshooting](#troubleshooting)
- [Stage 2: Make It Conversational](#stage-2-make-it-conversational)
- [Experiments to Try](#experiments-to-try)
- [Learning Resources](#learning-resources)
- [License](#license)

---

## What Is This?

A **language model** is a program that learns to predict the next character (or word) in a sequence of text. You give it millions of characters of Armenian text, and it learns patterns like:

- Which letters commonly follow other letters
- How Armenian words are typically structured
- How sentences and paragraphs flow

After training, it can **generate new Armenian text** that looks surprisingly real, one character at a time.

This entire project is ~400 lines of Python. Every part is commented so you can understand what's happening.

---

## Quick Start

### Option A: Google Colab (Easiest - No Setup!)

Click this badge to open a notebook in Google Colab with a free GPU. No installation needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EdikSimonian/armenian-gpt/blob/main/notebooks/armgpt_colab.ipynb)

Just run each cell from top to bottom. The notebook handles everything.

### Option B: Run on Your Computer

You need Python 3.8 or newer. Open a terminal and run:

```bash
# 1. Get the code
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armenian-gpt

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # On Mac/Linux
# venv\Scripts\activate         # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Armenian Wikipedia (~500 MB, takes a few minutes)
python data/download.py

# 5. Prepare the data (clean text, build vocabulary)
python data/prepare.py

# 6. Train the model (pick one)
python train.py --preset tiny     # ~1 min on CPU  (for testing)
python train.py --preset small    # ~30 min on GPU (recommended)

# 7. Generate Armenian text!
python generate.py --prompt "Հայաստան"
```

> **What is a virtual environment?** It's an isolated folder where Python installs packages just for this project, so it doesn't affect your system Python. You need to activate it every time you open a new terminal with `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows). You'll see `(venv)` in your prompt when it's active.

### What You Need

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | latest |
| RAM | 4 GB | 8 GB |
| Disk space | 1 GB | 3 GB |
| GPU | not required | any NVIDIA GPU (or use Colab) |

---

## Project Structure

```
armgpt/
├── config.py                  # All settings in one place (presets, hyperparameters)
├── model.py                   # The GPT model - the brain of ArmGPT (~150 lines)
├── train.py                   # Stage 1: Training loop (~150 lines)
├── generate.py                # Generate text with a trained model (~50 lines)
├── finetune.py                # Stage 2: Fine-tune on conversations
├── chat.py                    # Stage 2: Interactive chat interface
├── requirements.txt           # Python packages needed
├── data/
│   ├── download.py            # Downloads Armenian Wikipedia text
│   ├── prepare.py             # Cleans text, builds vocab, creates train/val splits
│   └── prepare_chat.py        # Stage 2: Downloads and prepares conversation data
├── tokenizers/
│   ├── char_tokenizer.py      # Level 1: one character = one token (beginners)
│   └── bpe_tokenizer.py       # Level 2: subword tokens with BPE (advanced)
└── notebooks/
    └── armgpt_colab.ipynb     # All-in-one Colab notebook with visualizations
```

**What each file does:**

| File | Purpose | Lines |
|---|---|---|
| `config.py` | Stores all settings (model size, learning rate, etc.) so you can change them in one place | ~130 |
| `model.py` | Defines the GPT neural network: attention, MLP, transformer blocks | ~150 |
| `train.py` | Loads data, trains the model, tracks metrics, saves checkpoints | ~150 |
| `generate.py` | Loads a trained model and generates new Armenian text | ~70 |
| `data/download.py` | Downloads and extracts Armenian Wikipedia articles | ~120 |
| `data/prepare.py` | Cleans text, builds character vocabulary, saves as binary files | ~100 |
| `tokenizers/char_tokenizer.py` | Maps each character to a number and back | ~50 |
| `tokenizers/bpe_tokenizer.py` | Trains a smarter tokenizer that groups common character sequences | ~80 |

---

## Step-by-Step Guide

### Step 1: Download the Data

```bash
python data/download.py
```

**What happens:**
1. Downloads the Armenian Wikipedia dump (~500 MB compressed) from `dumps.wikimedia.org`
2. Extracts plain text from the XML, removing all HTML/wiki markup
3. Saves everything to `data/raw_text.txt`

**Output:** A file with millions of characters of clean Armenian text from 300,000+ Wikipedia articles.

If the download is slow, you can also manually download the file from:
`https://dumps.wikimedia.org/hywiki/latest/hywiki-latest-pages-articles.xml.bz2`
and place it in the `data/` folder.

---

### Step 2: Prepare the Data

```bash
# Level 1 (default): character-level tokenization
python data/prepare.py

# Level 2 (advanced): BPE subword tokenization
python data/prepare.py --tokenizer bpe
```

**What happens:**
1. Reads `data/raw_text.txt`
2. Cleans the text:
   - Normalizes Unicode (NFC)
   - Keeps only Armenian characters (U+0530-U+058F), punctuation, digits, and spaces
   - Removes Latin, Cyrillic, emojis, and other non-Armenian content
3. Builds a vocabulary (mapping each unique character to a number)
4. Encodes the entire text as a sequence of numbers
5. Splits into 90% training / 10% validation
6. Saves binary files: `data/train.bin`, `data/val.bin`, `data/tokenizer.json`

**Output files:**

| File | What it contains |
|---|---|
| `data/train.bin` | Training data as encoded numbers (binary, ~90% of data) |
| `data/val.bin` | Validation data as encoded numbers (binary, ~10% of data) |
| `data/tokenizer.json` | The vocabulary mapping (character-to-number and back) |

---

### Step 3: Train the Model

```bash
# Quick test on CPU (~1 min)
python train.py --preset tiny

# Recommended: good results on a GPU (~30 min)
python train.py --preset small

# Best results on a strong GPU (~2 hours)
python train.py --preset medium
```

**What happens during training:**
1. Creates a GPT model with the chosen size
2. Loads training data into memory
3. Repeats thousands of times:
   - Grabs a random batch of text sequences
   - Model predicts the next character at each position
   - Compares predictions to the actual next characters
   - Calculates how wrong it was (the **loss**)
   - Adjusts the model weights to be less wrong next time
4. Every N steps: evaluates on validation data, generates sample text, saves a checkpoint

**What you'll see in the terminal:**

```
==================================================
  ArmGPT Training
==================================================
  Device:      cuda
  Model:       6 layers, 6 heads, 384 dim
  Block size:  256
  Batch size:  64
  Max iters:   5000
  Tokenizer:   char
==================================================

GPT model initialized: 10,234,880 parameters
Train data: 45,231,567 tokens
Val data:   5,025,729 tokens

step     0 | loss 4.4012 | lr 0.00e+00 | 52341 tok/s
step    10 | loss 4.1523 | lr 1.00e-04 | 48923 tok/s
step    20 | loss 3.8234 | lr 2.00e-04 | 49102 tok/s
...

==================================================
  Step 500 Evaluation
  Train loss:   2.3412
  Val loss:     2.4521
  Perplexity:   11.61
  Accuracy:     28.3%
==================================================

--- Sample (step 500) ---
Հայաստանի մարզdelays...
--- End sample ---
```

The loss starts around **4.5** (random guessing) and should drop below **2.0** with enough training.

---

### Step 4: Generate Text

```bash
# Basic generation
python generate.py --prompt "Հայաստան"

# More characters
python generate.py --prompt "Երևանը" --length 500

# More creative output
python generate.py --prompt "Հայաստանի" --temperature 1.2

# More conservative output
python generate.py --prompt "Հայաստանի" --temperature 0.3

# Multiple samples
python generate.py --prompt "Հայաստանի" --num_samples 3

# Use a specific checkpoint
python generate.py --checkpoint checkpoints/step_3000.pt --prompt "Հայաստան"
```

**All generation options:**

| Flag | Default | What it does |
|---|---|---|
| `--prompt` | `"Հայաստան"` | Starting text in Armenian |
| `--length` | `300` | Number of characters to generate |
| `--temperature` | `0.8` | Controls randomness (see below) |
| `--top_k` | `40` | Only sample from the top K most likely characters (0 = all) |
| `--num_samples` | `1` | How many separate texts to generate |
| `--checkpoint` | `checkpoints/final.pt` | Path to the trained model file |
| `--data_dir` | `data` | Where to find `tokenizer.json` |

**Temperature explained:**

| Temperature | Effect | Best for |
|---|---|---|
| 0.1 - 0.3 | Very safe, repetitive, stays close to training data | Seeing what the model learned best |
| 0.5 - 0.8 | Balanced, natural-sounding text | General use |
| 1.0 | Standard sampling, matches training distribution | Measuring model quality |
| 1.2 - 2.0 | Creative, surprising, sometimes nonsensical | Fun experimentation |

---

## How It Works (The Concepts)

### What is a Language Model?

Imagine you're typing on your phone and it suggests the next word. That's a tiny language model! ArmGPT does the same thing, but for Armenian characters, and it generates entire paragraphs.

### The Pipeline

```
Armenian Wikipedia  -->  Clean Text  -->  Numbers  -->  Train GPT  -->  Generate Text
     (raw data)         (prepare.py)    (tokenizer)    (train.py)    (generate.py)
```

### Inside the GPT Model

The model is called a **Transformer**. Here's what each part does:

```
Input: "Հdelays" (as numbers: [23, 41, 55, ...])
         |
   [Token Embedding]     Each number becomes a vector (list of 384 numbers)
         |
   [Position Embedding]  Adds position info (so the model knows word order)
         |
   [Transformer Block 1] \
   [Transformer Block 2]  |  Each block has two parts:
   [Transformer Block 3]  |   - Self-Attention: tokens look at each other
   [Transformer Block 4]  |   - MLP: each token thinks about what it saw
   [Transformer Block 5]  |
   [Transformer Block 6] /
         |
   [Output Head]          Predicts probability of each possible next character
         |
Output: "ა" has 73% chance, "ո" has 12% chance, "ե" has 8% chance, ...
```

**Self-Attention** is the key idea: each character can look at all previous characters to understand context. For example, to predict what comes after "Հայdelays", the model looks at "Հ", "ա", "յ" together to recognize this is the start of an Armenian word.

**Training** means showing the model millions of examples and adjusting its weights (numbers inside the model) so its predictions get more accurate over time.

---

## Tokenization Levels

### Level 1: Character-Level (Default)

Each character is one token. Simple and easy to understand.

```python
# Armenian alphabet + punctuation + space = ~80-100 tokens
"Բdelays" -> [12, 33, 45, 23, 67, 89, ...]   # each character is one number
```

**Pros:** Simple, easy to understand, no extra libraries
**Cons:** Longer sequences (every character is a separate token), slower to learn long-range patterns

### Level 2: BPE - Byte Pair Encoding (Advanced)

Groups common character combinations into single tokens. Requires `sentencepiece`.

```bash
# First, install sentencepiece
pip install sentencepiece

# Prepare data with BPE
python data/prepare.py --tokenizer bpe

# Train with BPE
python train.py --tokenizer bpe
```

```python
# Common words become 1-2 tokens instead of many characters
"Հайdelays" -> [234, 1523, ...]   # fewer tokens = faster training
```

**Pros:** Better text quality, shorter sequences, faster training
**Cons:** More complex concept, needs an extra library, harder to inspect

**When to switch to Level 2:** After you've experimented with character-level and want better results. BPE is what real language models (GPT-4, LLaMA, etc.) use.

---

## Training Data Sources

### Primary: Armenian Wikipedia (Default)

Downloaded automatically by `data/download.py`.

- ~325,000 articles
- ~2-3 GB of clean text
- Covers: history, science, geography, culture, biography, and more
- Free, no account needed

### Pre-cleaned Dataset (Recommended)

A pre-cleaned, deduplicated Armenian text corpus is published on HuggingFace as [**edisimon/armenian-clean-text**](https://huggingface.co/datasets/edisimon/armenian-clean-text). Grabbing it lets you skip steps 1 and 2 entirely.

- **Size:** ~29 GB of clean text (~17B characters)
- **License:** CC-BY-4.0
- **Processing:** NFC-normalized, deduplicated at the paragraph level (MD5), stripped of non-Armenian characters, whitespace and blank lines collapsed
- **Sources:** Armenian Wikipedia, CC-100, CulturaX, OSCAR-2301, mC4, HPLT v2.0, Glot500 (see the table below for details)

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='edisimon/armenian-clean-text', filename='clean_text.txt', repo_type='dataset', local_dir='data/')"
```

Then go straight to tokenization: `python 3_tokenize.py --tokenizer bpe`

### Optional: Additional Data Sources

For better results, you can add more data. After downloading additional data, append it to `data/raw_text.txt` before running `data/prepare.py`.

| Source | Size | How to get it |
|---|---|---|
| [OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | 4.9 GB / 336M tokens | HuggingFace (free account required) |
| [CC-100](https://data.statmt.org/cc-100/) | 776 MB | Direct download from `data.statmt.org/cc-100/` |
| [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | 2.4B tokens | HuggingFace (largest cleaned corpus) |
| [Armenian Wikisource](https://dumps.wikimedia.org/hywikisource/) | ~50 MB | Literary works, classical texts |
| [HPLT v1.2](https://huggingface.co/datasets/HPLT/hplt_monolingual_v1_2) | varies | Common Crawl + Internet Archive |

**Example: Adding CC-100 data**

```bash
# Download CC-100 Armenian
wget https://data.statmt.org/cc-100/hy.txt.xz
xz -d hy.txt.xz

# Append to raw text
cat hy.txt >> data/raw_text.txt

# Re-prepare the data
python data/prepare.py
```

---

## Model Presets

Three presets are available, from small to large:

| Preset | Layers | Heads | Dim | Context | Parameters | Training Time | Hardware |
|---|---|---|---|---|---|---|---|
| `tiny` | 1 | 2 | 64 | 64 chars | ~50K | ~1 min | Any CPU |
| `small` | 6 | 6 | 384 | 256 chars | ~10M | ~30 min | GPU (Colab T4) |
| `medium` | 8 | 8 | 512 | 512 chars | ~25M | ~2 hours | Good GPU (A100/V100) |

**What each setting means:**

| Setting | What it controls |
|---|---|
| `n_layer` | How many transformer blocks stacked on top of each other (more = smarter but slower) |
| `n_head` | How many attention heads per block (more = model can focus on more things at once) |
| `n_embd` | Size of the internal vectors (bigger = model can represent more complex patterns) |
| `block_size` | How many characters the model can see at once (its "memory window") |
| `batch_size` | How many text sequences to train on simultaneously (bigger = faster but needs more memory) |
| `max_iters` | Total number of training steps |
| `learning_rate` | How big the weight adjustments are each step (too high = unstable, too low = slow) |
| `dropout` | Fraction of neurons randomly turned off during training (prevents memorization) |

---

## All Command-Line Options

### train.py

```bash
python train.py [OPTIONS]

Options:
  --preset {tiny,small,medium}   Use a preset configuration
  --n_layer INT                  Number of transformer blocks
  --n_head INT                   Number of attention heads
  --n_embd INT                   Embedding dimension
  --block_size INT               Context window length
  --dropout FLOAT                Dropout rate (0.0 to 0.5)
  --batch_size INT               Batch size
  --max_iters INT                Total training steps
  --learning_rate FLOAT          Peak learning rate
  --tokenizer {char,bpe}         Which tokenizer to use
  --device {auto,cpu,cuda,mps}   Device for training
  --data_dir PATH                Path to data directory
  --resume_from PATH             Path to checkpoint to resume from
```

### generate.py

```bash
python generate.py [OPTIONS]

Options:
  --checkpoint PATH              Path to model checkpoint (default: checkpoints/final.pt)
  --prompt TEXT                   Starting text in Armenian (default: "Հайdelays")
  --length INT                   Characters to generate (default: 300)
  --temperature FLOAT            Randomness 0.1-2.0 (default: 0.8)
  --top_k INT                    Top-k sampling, 0=off (default: 40)
  --num_samples INT              Number of texts to generate (default: 1)
  --data_dir PATH                Path to tokenizer.json (default: data)
```

### data/prepare.py

```bash
python data/prepare.py [OPTIONS]

Options:
  --tokenizer {char,bpe}         Tokenizer type (default: char)
```

---

## Performance Metrics

During training, these metrics are tracked and saved to `checkpoints/metrics.json`:

| Metric | What it measures | Starting value | Good value | How it's calculated |
|---|---|---|---|---|
| **Train Loss** | Error on training data | ~4.5 | < 1.8 | Cross-entropy between predictions and actual next characters |
| **Val Loss** | Error on unseen data | ~4.5 | < 2.0 | Same as train loss but on validation data (data the model hasn't seen) |
| **Perplexity** | How "confused" the model is | ~90 | 4 - 8 | `e^(val_loss)` - lower means more confident predictions |
| **Accuracy** | % of correct predictions | ~1% | 30 - 45% | How often the model's top prediction matches the actual next character |
| **Tokens/sec** | Training speed | - | - | How many characters processed per second |

**How to interpret the numbers:**

- **Loss going down** = the model is learning
- **Train loss much lower than val loss** = the model is memorizing (overfitting) instead of learning general patterns. Try: more data, more dropout, or fewer parameters.
- **Loss stopped going down** = the model has learned as much as it can at this size. Try: bigger model, more data, or longer training.
- **Perplexity of 6** means the model is choosing between ~6 characters at each step (out of ~80). That's pretty good!

**Visualizing metrics:**

The Colab notebook automatically plots loss curves. For local training, you can use the metrics file:

```python
import json
import matplotlib.pyplot as plt

with open("checkpoints/metrics.json") as f:
    m = json.load(f)

plt.plot(m["steps"], m["train_loss"], label="Train")
plt.plot(m["steps"], m["val_loss"], label="Validation")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

## Training on the Cloud

### Google Colab (Free)

The easiest option. Use the provided notebook:

1. Open `notebooks/armgpt_colab.ipynb` in Colab
2. Go to **Runtime > Change runtime type > T4 GPU**
3. Run all cells

Free tier gives you ~4 hours of T4 GPU time per session.

### Any Cloud GPU (AWS, GCP, Lambda Labs, etc.)

The scripts work on any machine with Python and PyTorch:

```bash
# SSH into your cloud machine, then:
git clone https://github.com/EdikSimonian/armenian-gpt.git
cd armgpt
pip install -r requirements.txt
python data/download.py
python data/prepare.py
python train.py --preset small    # or medium for better results
```

### Apple Silicon (M1/M2/M3/M4 Mac)

PyTorch supports Apple's MPS (Metal Performance Shaders) backend. ArmGPT auto-detects it:

```bash
python train.py --preset small
# Output will show: Device: mps
```

---

## Resuming Training

Training saves checkpoints automatically. If training is interrupted (or you want to train longer), resume from the last checkpoint:

```bash
# Resume from a specific checkpoint
python train.py --resume_from checkpoints/step_3000.pt

# Resume and train for more steps
python train.py --resume_from checkpoints/final.pt --max_iters 10000
```

Checkpoints are saved every 1000 steps (configurable with `save_interval` in `config.py`).

Each checkpoint file contains:
- Model weights
- Optimizer state (so learning continues smoothly)
- The step number
- The full config used

---

## Troubleshooting

### "externally-managed-environment" error (macOS/Linux)

Modern Python installations don't let you install packages system-wide. Use a virtual environment:

```bash
# Create it (one time)
python3 -m venv venv

# Activate it (every time you open a new terminal)
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Now pip works
pip install -r requirements.txt
```

You'll see `(venv)` in your terminal prompt when the environment is active. If you get "command not found" errors, you probably forgot to activate it.

### "No module named 'torch'"

Make sure your virtual environment is activated, then install:
```bash
source venv/bin/activate
pip install torch
```

Or visit [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific instructions.

### "CUDA out of memory"

The model is too big for your GPU. Try:
```bash
# Use a smaller preset
python train.py --preset tiny

# Or reduce batch size
python train.py --preset small --batch_size 32

# Or reduce model size
python train.py --n_layer 4 --n_head 4 --n_embd 256
```

### "data/raw_text.txt not found"

You need to download the data first:
```bash
python data/download.py
```

### "data/train.bin not found"

You need to prepare the data first:
```bash
python data/prepare.py
```

### "checkpoints/final.pt not found"

You need to train the model first:
```bash
python train.py --preset tiny
```

### Download is too slow

The Wikipedia dump is ~500 MB. If your connection is slow:
1. Download it manually from a browser: `https://dumps.wikimedia.org/hywiki/latest/hywiki-latest-pages-articles.xml.bz2`
2. Save it to the `data/` folder
3. Run `python data/download.py` again (it will skip the download if the file exists)

### Training loss is not going down

- Make sure you have enough data (at least a few MB of text)
- Try a smaller model first (`--preset tiny`) to verify everything works
- Check that the data looks correct: open `data/raw_text.txt` and verify it contains Armenian text

### Generated text looks like garbage

- The model needs more training. Try increasing `--max_iters`
- Use a lower temperature: `--temperature 0.5`
- Use top-k sampling: `--top_k 20`
- Make sure you trained on enough data

---

## Stage 2: Make It Conversational

Stage 1 gives you a base model that continues text like autocomplete. Stage 2 fine-tunes it on question-answer pairs so it can have conversations — like a mini ChatGPT in Armenian.

**This is how real AI assistants are built:**
1. Stage 1: Train on lots of text (Wikipedia) -> model learns language patterns
2. Stage 2: Fine-tune on conversations (this step) -> model learns to answer questions

### Quick Start (Stage 2)

```bash
# 1. Install the datasets library (needed to download conversation data)
pip install datasets

# 2. Download and prepare conversation data (52K Armenian Q&A pairs)
python data/prepare_chat.py

# 3. Fine-tune your Stage 1 model on conversations
python finetune.py

# 4. Chat with your model!
python chat.py
```

### What Happens in Each Step

**`data/prepare_chat.py`** downloads [Alpaca-Armenian](https://huggingface.co/datasets/saillab/alpaca-armenian-cleaned) — 52,000 instruction/response pairs already translated to Armenian. It formats them with special tokens:

```
<|user|>Ի՞delays է Արarat:<|end|><|assistant|>Արdelays...<|end|>
```

**`finetune.py`** loads your Stage 1 model (`checkpoints/final.pt`), extends its vocabulary with the special tokens, and trains it on the conversation data. Uses a lower learning rate and fewer steps since the model already knows Armenian from Stage 1.

**`chat.py`** is an interactive chat loop. Type a question, get a response:

```
$ python chat.py

==================================================
  ArmGPT Chat
  Device: cuda | Temp: 0.7
  Type 'quit' to exit
==================================================

You: Ի՞delays է Հdelays
ArmGPT: Հdelays...

You: quit
Bye!
```

### Chat Options

```bash
python chat.py --temperature 0.3    # more focused responses
python chat.py --temperature 1.0    # more creative responses
python chat.py --max_length 500     # longer responses
python chat.py --checkpoint checkpoints/chat_final.pt  # specific checkpoint
```

### New Files (Stage 2)

| File | Purpose |
|---|---|
| `data/prepare_chat.py` | Download Alpaca-Armenian, format with chat tokens, save as binary |
| `finetune.py` | Fine-tune Stage 1 model on conversation data |
| `chat.py` | Interactive CLI chat interface |

---

## Experiments to Try

Once you have the basic model working, try these experiments to learn more:

### 1. Change the Model Size

How does the number of layers affect quality?

```bash
python train.py --n_layer 1 --max_iters 2000   # tiny model
python train.py --n_layer 4 --max_iters 2000   # medium model
python train.py --n_layer 8 --max_iters 2000   # large model
```

Compare the validation loss and generated text for each.

### 2. Change the Context Window

How much "memory" does the model need?

```bash
python train.py --block_size 32 --max_iters 2000    # sees 32 characters
python train.py --block_size 128 --max_iters 2000   # sees 128 characters
python train.py --block_size 512 --max_iters 2000   # sees 512 characters
```

### 3. Compare Tokenizers

Does BPE make a difference?

```bash
# Character-level
python data/prepare.py --tokenizer char
python train.py --tokenizer char --max_iters 3000

# BPE (requires: pip install sentencepiece)
python data/prepare.py --tokenizer bpe
python train.py --tokenizer bpe --max_iters 3000
```

### 4. Temperature Exploration

Generate with different temperatures and compare:

```bash
python generate.py --temperature 0.1 --length 300
python generate.py --temperature 0.5 --length 300
python generate.py --temperature 1.0 --length 300
python generate.py --temperature 2.0 --length 300
```

### 5. Track Overfitting

Train for a long time and watch train loss vs. val loss:

```bash
python train.py --preset small --max_iters 20000
```

If train loss keeps dropping but val loss goes up, the model is memorizing the training data instead of learning general patterns. This is called **overfitting**.

### 6. Learning Rate Experiments

```bash
python train.py --learning_rate 0.01 --max_iters 2000    # too high?
python train.py --learning_rate 0.001 --max_iters 2000   # default
python train.py --learning_rate 0.0001 --max_iters 2000  # too low?
```

---

## Learning Resources

### Videos
- [Karpathy: Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 2-hour tutorial building a GPT step by step
- [Karpathy: Neural Networks - Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Full course from basics to transformers
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Beautiful visual explanations

### Code
- [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) - GPT in 200 lines of pure Python (no libraries!)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - The project that inspired ArmGPT
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT implementation
- [picoGPT](https://github.com/jaymody/picoGPT) - GPT-2 inference in 60 lines of NumPy

### Papers (for the curious)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper (2017)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - The GPT-2 paper

### Armenian NLP
- [Armenian Wikipedia](https://hy.wikipedia.org/) - The source of our training data
- [Eastern Armenian National Corpus](http://www.eanc.net/) - Annotated Armenian text corpus
- [HuggingFace Armenian datasets](https://huggingface.co/datasets?language=language:hy) - 250+ Armenian datasets

---

## License

MIT - Use this code however you want. Learn from it, modify it, share it.
