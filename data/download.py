"""
Download Armenian text data for training.

This script downloads Armenian Wikipedia articles — a clean, free source
of Armenian text. No account needed.

Usage:
    python data/download.py

After running, you'll have a file called data/raw_text.txt with
millions of characters of Armenian text.
"""

import os
import bz2
import re
import urllib.request
import sys

# Armenian Wikipedia dump URL (latest articles)
WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/hywiki/latest/"
    "hywiki-latest-pages-articles.xml.bz2"
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(DATA_DIR, "raw_text.txt")
DUMP_FILE = os.path.join(DATA_DIR, "hywiki-latest-pages-articles.xml.bz2")


def download_dump():
    """Download the Armenian Wikipedia dump if not already present."""
    if os.path.exists(DUMP_FILE):
        print(f"Dump already downloaded: {DUMP_FILE}")
        return

    print(f"Downloading Armenian Wikipedia dump...")
    print(f"URL: {WIKI_DUMP_URL}")
    print("This may take a few minutes (file is ~500 MB)...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(WIKI_DUMP_URL, DUMP_FILE, reporthook=progress_hook)
    print("\nDownload complete!")


def strip_markup(text):
    """Remove MediaWiki markup from text, keeping plain Armenian content."""
    # Remove XML/HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove wiki templates {{...}} (non-greedy, handles simple cases)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove wiki links [[...]] but keep the display text
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove external links [http... text] but keep the text
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)
    # Remove category/file links
    text = re.sub(r"\[\[(?:Категория|Category|Պատկdelays|File|Файл|Image):[^\]]*\]\]", "", text)
    # Remove bold/italic markers
    text = re.sub(r"'{2,}", "", text)
    # Remove headings (== text ==)
    text = re.sub(r"={2,}(.+?)={2,}", r"\1", text)
    # Remove references
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    # Remove remaining HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Remove tables {| ... |}
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)
    # Remove lines starting with | or ! (table rows)
    text = re.sub(r"^\s*[|!].*$", "", text, flags=re.MULTILINE)
    # Remove multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just punctuation or whitespace
    text = re.sub(r"^\s*[*#:;]+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_articles(dump_path):
    """
    Extract plain text from the Wikipedia XML dump.
    Yields one article's text at a time.
    """
    print("Extracting articles from dump (this takes a few minutes)...")

    # Read the bz2-compressed XML and extract text between <text> tags
    current_text = []
    in_text = False
    article_count = 0

    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            # Detect start of article text
            if "<text" in line:
                in_text = True
                # Get content after the opening tag on this line
                match = re.search(r"<text[^>]*>(.*)", line)
                if match:
                    current_text.append(match.group(1))
                continue

            if in_text:
                # Detect end of article text
                if "</text>" in line:
                    current_text.append(line.split("</text>")[0])
                    full_text = "\n".join(current_text)
                    current_text = []
                    in_text = False

                    # Skip redirects and very short articles
                    if full_text.startswith("#REDIRECT") or \
                       full_text.startswith("#ՎԵՐԱՀՂՈՒՄ") or \
                       len(full_text) < 200:
                        continue

                    cleaned = strip_markup(full_text)
                    if len(cleaned) > 100:
                        article_count += 1
                        if article_count % 10000 == 0:
                            print(f"  Extracted {article_count} articles...")
                        yield cleaned
                else:
                    current_text.append(line)

    print(f"  Total articles extracted: {article_count}")


def main():
    # Step 1: Download
    download_dump()

    # Step 2: Extract and save
    print(f"\nExtracting text to {RAW_FILE}...")
    total_chars = 0

    with open(RAW_FILE, "w", encoding="utf-8") as out:
        for article_text in extract_articles(DUMP_FILE):
            out.write(article_text)
            out.write("\n\n")  # separate articles with blank lines
            total_chars += len(article_text)

    mb = total_chars / (1024 * 1024)
    print(f"\nDone! Saved {total_chars:,} characters ({mb:.1f} MB) to {RAW_FILE}")
    print(f"You can now run: python data/prepare.py")


if __name__ == "__main__":
    main()
