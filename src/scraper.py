import os
import time
import json
import re
import sys
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCRAPE_TARGETS, RAW_DATA_DIR, SCRAPE_DELAY

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; educational-research-bot/1.0; "
        "RAG chatbot assignment)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def url_to_slug(url: str) -> str:
    path = urlparse(url).path
    slug = path.strip("/").replace("/", "__")
    slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", slug)
    slug = re.sub(r"\.[a-z]+$", "", slug)  # remove URL extension (e.g. .html)
    return slug


def fetch_page(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.exceptions.HTTPError as e:
        print(f"  [SKIP] HTTP {e.response.status_code} – {url}")
    except requests.exceptions.RequestException as e:
        print(f"  [SKIP] Request failed – {url}: {e}")
    return None


def _table_to_text(table_tag) -> str:
    """Convert an HTML table to a pipe-separated text representation."""
    lines = []
    for row in table_tag.find_all("tr"):
        cells = row.find_all(["th", "td"])
        cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]
        if any(cell_texts):
            lines.append(" | ".join(cell_texts))
    return "\n".join(lines)


def clean_html(soup: BeautifulSoup, url: str) -> dict:
    # Remove non-content elements
    for tag in soup.select("nav, footer, script, style, .headerlink, "
                           ".toctree-wrapper, #searchbox, .sphinxsidebar, "
                           ".bd-sidebar, .bd-toc, .prev-next-area"):
        tag.decompose()

    # PyData Sphinx Theme main content selector
    content_div = (
        soup.select_one("div.bd-content")
        or soup.select_one("article.bd-article")
        or soup.select_one("div#main-content")
        or soup.select_one("div.body")
        or soup.find("main")
    )

    if not content_div:
        content_div = soup.find("body") or soup

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path

    # Step 1: Unwrap inline formatting so text flows naturally with surroundings
    for tag in content_div.find_all(["em", "strong", "a", "b", "i", "u",
                                      "small", "abbr", "cite", "span"]):
        tag.unwrap()
    # Unwrap inline <code> that are NOT inside a <pre> block
    for tag in content_div.find_all("code"):
        if not tag.find_parent("pre"):
            tag.unwrap()

    # Step 2: Convert <pre> code blocks to single readable lines
    for pre in content_div.find_all("pre"):
        code_text = re.sub(r"\s+", " ", pre.get_text(separator="", strip=False)).strip()
        if code_text:  # skip empty <pre> blocks (e.g. image placeholders)
            pre.replace_with(f"\nCode: {code_text}\n")
        else:
            pre.decompose()

    # Step 3: Convert tables to pipe-separated readable text
    for table in content_div.find_all("table"):
        table.replace_with(f"\n{_table_to_text(table)}\n")

    # Step 4: Collapse each block element to a single line.
    # h2-h6 get a '## ' prefix so chunk_for_qa can split at section boundaries.
    # <dt> uses separator=" " so param name and type don't merge (e.g. "sep str" not "sepstr").
    for tag in content_div.find_all(["h2", "h3", "h4", "h5", "h6"]):
        text = re.sub(r"\s+", " ", tag.get_text(separator="", strip=False)).strip()
        if text:
            tag.replace_with(f"\n## {text}\n")
    for tag in content_div.find_all(["p", "h1", "li", "dd"]):
        text = re.sub(r"\s+", " ", tag.get_text(separator="", strip=False)).strip()
        if text:
            tag.replace_with(f"\n{text}\n")
    for tag in content_div.find_all("dt"):
        text = re.sub(r"\s+", " ", tag.get_text(separator=" ", strip=False)).strip()
        if text:
            tag.replace_with(f"\n{text}\n")

    raw_text = content_div.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    content = "\n".join(cleaned_lines).strip()

    return {
        "url": url,
        "title": title,
        "content": content,
        "scraped_at": datetime.utcnow().isoformat(),
    }


def save_page(page_data: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = url_to_slug(page_data["url"])
    filepath = os.path.join(out_dir, f"{slug}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"URL: {page_data['url']}\n")
        f.write(f"TITLE: {page_data['title']}\n")
        f.write(f"SCRAPED_AT: {page_data['scraped_at']}\n")
        f.write("-" * 60 + "\n")
        f.write(page_data["content"])

    return filepath


def run_scraper(targets: list[str] = None, out_dir: str = None) -> list[str]:
    if targets is None:
        targets = SCRAPE_TARGETS
    if out_dir is None:
        out_dir = RAW_DATA_DIR

    saved_paths = []
    failed = []

    print(f"Scraping {len(targets)} pages → {out_dir}")
    for url in tqdm(targets, desc="Scraping"):
        soup = fetch_page(url)
        if soup is None:
            failed.append(url)
            time.sleep(SCRAPE_DELAY)
            continue

        page_data = clean_html(soup, url)

        if len(page_data["content"]) < 100:
            print(f"  [WARN] Very short content ({len(page_data['content'])} chars) – {url}")

        path = save_page(page_data, out_dir)
        saved_paths.append(path)
        time.sleep(SCRAPE_DELAY)

    print(f"\nDone: {len(saved_paths)} saved, {len(failed)} failed.")
    if failed:
        print("Failed URLs:")
        for u in failed:
            print(f"  - {u}")

    return saved_paths


if __name__ == "__main__":
    paths = run_scraper()
    print(f"\nSaved files:")
    for p in paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p}  ({size_kb:.1f} KB)")
