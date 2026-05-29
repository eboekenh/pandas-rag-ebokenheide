"""Synthetic Q&A generator that uses a local Ollama model.

Reads the scraped pages from ``data/raw/`` and generates question-answer pairs
for the hybrid retriever's Q&A store, written to
``data/processed/qa_dataset_ollama.csv``.

Usage:
    python -m src.qa_generator

Make sure Ollama is running and the model is pulled:
    ollama serve
    ollama pull llama3.1:8b
"""
import os
import sys
import json
import re
import time

import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_DIR,
    QA_DATASET_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_QA_MODEL,
)

WRITE_EVERY = 3  # flush buffer to disk after every N pairs

QA_PROMPT = """You are building a Q&A dataset for a RAG chatbot that helps coders, data scientists, data analysts, and data engineers use the pandas library effectively. Your goal is to generate question-answer pairs that teach users HOW to use pandas in real data workflows — not just what a function is called.

Given the pandas documentation below, generate exactly {n} question-answer pairs.

RULES:
- Questions must be general and standalone — a user must be able to ask them WITHOUT having read the docs
- NEVER reference "the example", "the code", "in the example code", or example variable names (df, df2, s, s1, result, etc.)
- Questions must be practical and relevant to data work: purpose, how to use it, key parameters, return value, or common use cases
- Answers must be explanatory: describe what the concept/feature does, what the result looks like, and include a short code example when relevant
- If code examples are provided, use them to write a practical usage question (not about what the example variable contains)

Return ONLY a valid JSON object with a "pairs" key. No explanations, no markdown.

Format:
{{"pairs": [
  {{"question": "...", "answer": "..."}},
  ...
]}}

TEXT:
{chunk}"""


def load_raw_pages(raw_dir: str) -> list[dict]:
    pages = []
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        url = ""
        title = filename
        content_start = 0
        for i, line in enumerate(raw.splitlines()):
            if line.startswith("URL: "):
                url = line[5:].strip()
            elif line.startswith("TITLE: "):
                title = line[7:].strip()
            elif line.startswith("-" * 10):
                content_start = i + 1
                break

        content = "\n".join(raw.splitlines()[content_start:]).strip()
        if content:
            pages.append({"filename": filename, "url": url, "title": title, "content": content})

    print(f"Loaded {len(pages)} pages from {raw_dir}")
    return pages


def chunk_for_qa(content: str, max_chars: int = 3000) -> list[str]:
    """Split content into chunks that respect section boundaries."""
    raw_sections = re.split(r"(?=\n## )", content)
    sections = [s.strip() for s in raw_sections if s.strip()]

    chunks: list[str] = []
    current = ""

    for section in sections:
        first_line = section.split("\n")[0].strip()
        heading = first_line[3:] if first_line.startswith("## ") else first_line

        if len(section) <= max_chars:
            if len(current) + len(section) + 2 <= max_chars:
                current += ("\n\n" if current else "") + section
            else:
                if current:
                    chunks.append(current.strip())
                current = section
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            paragraphs = section.split("\n\n")
            sub = ""
            is_first = True
            for para in paragraphs:
                if len(sub) + len(para) + 2 <= max_chars:
                    sub += ("\n\n" if sub else "") + para
                else:
                    if sub:
                        chunks.append(sub.strip())
                    if len(para) > max_chars:
                        sentences = re.split(r"(?<=[.!?])\s+", para)
                        sub = "" if is_first else f"[continued: {heading}]\n"
                        for sent in sentences:
                            if len(sub) + len(sent) + 1 <= max_chars:
                                sub += (" " if sub else "") + sent
                            else:
                                if sub:
                                    chunks.append(sub.strip())
                                sub = f"[continued: {heading}]\n{sent}"
                    else:
                        sub = para if is_first else f"[continued: {heading}]\n\n{para}"
                is_first = False
            if sub:
                chunks.append(sub.strip())

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 100]


def generate_qa_for_chunk(
    chunk: str,
    source_url: str,
    n_pairs: int = 5,
    max_retries: int = 2,
) -> list[dict]:
    """Call Ollama to generate n_pairs Q&A pairs from chunk."""
    prompt = QA_PROMPT.format(n=n_pairs, chunk=chunk)

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_QA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.4},
                },
                timeout=300,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            # Extract first JSON object if extra text is present
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            # Fix common JSON issues from small models
            raw = raw.replace("\\'", "'")
            # Remove trailing commas before ] or }
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            # Replace smart quotes with straight quotes
            raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
            # Fix unescaped newlines inside strings
            raw = re.sub(r'(?<!\\)\n', ' ', raw)

            parsed = json.loads(raw)
            pairs = parsed.get("pairs", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(pairs, list):
                raise ValueError("Response does not contain a pairs array")

            valid = []
            for p in pairs:
                if isinstance(p, dict) and "question" in p and "answer" in p:
                    valid.append({
                        "question": str(p["question"]).strip(),
                        "answer": str(p["answer"]).strip(),
                        "source_page": source_url,
                    })
            return valid

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                print(f"  [RETRY {attempt + 1}] JSON parse error: {e}")
            else:
                print(f"  [SKIP] Failed after {max_retries + 1} attempts: {e}")
                return []
        except requests.RequestException as e:
            print(f"  [ERROR] Ollama request failed: {e}")
            print("  Make sure Ollama is running: ollama serve")
            return []

    return []


def run_qa_generator(raw_dir: str = None, out_path: str = None) -> None:
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if out_path is None:
        # Save to a separate file so the original dataset is not overwritten
        out_path = QA_DATASET_PATH.replace(".csv", "_ollama.csv")

    pages = load_raw_pages(raw_dir)

    if not pages:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}. Run scraper.py first.")

    # Resume: skip pages already processed
    already_done: set[str] = set()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path)
            already_done = set(existing_df["source_page"].dropna().unique())
            print(f"Resuming: {len(already_done)} page(s) already processed, skipping them.")
        except Exception:
            pass

    remaining = [p for p in pages if p["url"] not in already_done]
    print(f"Pages to process: {len(remaining)} / {len(pages)}")
    print(f"Using Ollama model: {OLLAMA_QA_MODEL}  ({OLLAMA_BASE_URL})")

    buffer: list[dict] = []
    total_pairs = 0

    def flush(buf: list[dict]) -> None:
        if not buf:
            return
        df = pd.DataFrame(buf, columns=["question", "answer", "source_page"])
        write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
        df.to_csv(out_path, mode="a", index=False, header=write_header)
        buf.clear()

    for page in tqdm(remaining, desc="Generating Q&A"):
        chunks = chunk_for_qa(page["content"], max_chars=1500)

        for chunk in tqdm(chunks, desc=f"  Chunks [{page['title'][:30]}]", leave=False):
            n_pairs = max(2, min(5, len(chunk) // 300))
            pairs = generate_qa_for_chunk(
                chunk=chunk,
                source_url=page["url"],
                n_pairs=n_pairs,
            )
            buffer.extend(pairs)
            total_pairs += len(pairs)

            if len(buffer) >= WRITE_EVERY:
                flush(buffer)
                tqdm.write(f"  Saved {total_pairs} pairs so far -> {out_path}")

    flush(buffer)
    print(f"\nDone! Total Q&A pairs generated: {total_pairs}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    run_qa_generator()
