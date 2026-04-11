---
language:
- hy
license: cc-by-4.0
tags:
- armenian
- text
- pretraining
- nlp
size_categories:
- 10B<n<100B
---

# Armenian Clean Text

A large-scale cleaned and deduplicated Armenian text corpus for language model pretraining.

## Dataset Details

- **Size**: ~29 GB of clean Armenian text (~17 billion characters)
- **Format**: Single `clean_text.txt` file, UTF-8 encoded
- **Processing**: Deduplicated (exact paragraph-level MD5), cleaned to retain only Armenian Unicode characters (U+0530-U+058F, U+FB13-U+FB17), digits, and basic punctuation

## Sources

| Source | Description | Link |
|--------|-------------|------|
| Armenian Wikipedia | ~325K articles covering history, science, geography, culture, biography | [hywiki dumps](https://dumps.wikimedia.org/hywiki/) |
| CC-100 | Armenian subset of Common Crawl monolingual data | [data.statmt.org/cc-100](https://data.statmt.org/cc-100/) |
| CulturaX | Large-scale cleaned multilingual corpus (Armenian subset) | [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| OSCAR-2301 | Web-crawled multilingual data (Armenian subset) | [oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| mC4 | Multilingual C4 (Armenian subset) | [allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| HPLT v2.0 | Common Crawl + Internet Archive cleaned data | [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) |
| Glot500 | Multilingual corpus for low-resource languages | [cis-lmu/Glot500](https://huggingface.co/datasets/cis-lmu/Glot500) |

## Processing Pipeline

1. **Download**: All sources streamed via HuggingFace datasets or direct download (never loads full datasets into RAM)
2. **Deduplication**: Exact paragraph-level dedup using MD5 hashes, minimum 50 character paragraph length
3. **Cleaning**: Unicode NFC normalization, strip all non-Armenian characters, collapse whitespace and blank lines

## Usage

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="edisimon/armenian-clean-text",
    filename="clean_text.txt",
    repo_type="dataset",
    local_dir="data/"
)
```

## Intended Use

Pretraining Armenian language models. Created for the [ArmGPT](https://github.com/EdikSimonian/armenian-gpt) project.
