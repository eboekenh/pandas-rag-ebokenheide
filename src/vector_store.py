"""
vector_store.py
---------------
Builds and loads the two ChromaDB vector stores used by this RAG chatbot:

  doc_store  – chunked raw documentation pages (used for Stage 2 RAG fallback).
  qa_store   – pre-generated Q/A pairs, questions embedded as vectors,
               answers stored as metadata (used for Stage 1 lookup).

Both stores use the same embedding model (multi-qa-MiniLM-L6-cos-v1) with
``normalize_embeddings=True``, and ChromaDB's default L2 distance metric.
A single cached embedder instance (``get_embedder()``) is shared across all
store operations to avoid reloading the model on every call.

Typical usage
-------------
First-time setup (run once, or after data changes):
    python -m src.vector_store

Runtime (called by retriever.py on startup):
    doc_store = load_doc_store()
    qa_store  = load_qa_store()
"""

import logging
import os
import shutil
import sys
import functools

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_DIR,
    QA_DATASET_PATH,
    CHROMA_PERSIST_DIR,
    DOC_COLLECTION_NAME,
    QA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Load and cache the sentence-transformer embedding model.

    Uses ``functools.lru_cache`` so the model is loaded from disk only once
    per process, regardless of how many times this function is called.
    L2-normalised output vectors (``normalize_embeddings=True``) bound the
    maximum L2 distance between any two vectors to ``sqrt(2)``.  This is
    required for LangChain's relevance score formula
    ``1 - L2_distance / sqrt(2)`` to stay in ``[0, 1]``; without
    normalisation the distance can exceed ``sqrt(2)`` and scores turn negative.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_raw_pages(raw_dir: str) -> list[dict]:
    """
    Read all ``.txt`` files from *raw_dir* and parse them into page dicts.

    Each file is expected to follow the format written by ``scraper.py``::

        URL: <url>
        TITLE: <title>
        ----------
        <content ...>

    Returns
    -------
    list[dict]
        Each dict has keys ``url``, ``title``, and ``content``.
        Files with no content after the separator are silently skipped.
    """
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
            pages.append({"url": url, "title": title, "content": content})

    return pages


def chunk_documents(pages: list[dict]) -> list[Document]:
    """
    Split page content into overlapping chunks and wrap them as LangChain Documents.

    Uses ``RecursiveCharacterTextSplitter`` with the CHUNK_SIZE and CHUNK_OVERLAP
    values from config.  Each chunk inherits the source URL and page title from
    its parent page as metadata, which is later cited in chatbot responses.

    Parameters
    ----------
    pages : list[dict]
        Output of ``load_raw_pages()``.

    Returns
    -------
    list[Document]
        Flat list of all chunks across all pages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for page in pages:
        chunks = splitter.split_text(page["content"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"source": page["url"], "title": page["title"]},
            ))
    return docs


def build_doc_store(raw_dir: str | None = None, persist_dir: str | None = None) -> Chroma:
    """
    Build the documentation vector store from raw scraped ``.txt`` files.

    Reads all pages from *raw_dir*, chunks them, embeds each chunk, and
    persists the resulting ChromaDB collection to *persist_dir*.
    Overwrites any existing store at that path without warning.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing the raw ``.txt`` files.  Defaults to RAW_DATA_DIR.
    persist_dir : str, optional
        Directory to persist the ChromaDB collection.  Defaults to
        ``<CHROMA_PERSIST_DIR>/docs``.

    Returns
    -------
    Chroma
        The built and persisted vector store.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* contains no ``.txt`` files.
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "docs")

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    pages = load_raw_pages(raw_dir)
    if not pages:
        raise FileNotFoundError(f"No .txt files in {raw_dir}. Run scraper.py first.")

    logger.info(f"Chunking {len(pages)} pages...")
    docs = chunk_documents(pages)
    logger.info(f"Created {len(docs)} chunks")

    embedder = get_embedder()

    logger.info(f"Building doc store → {persist_dir}")
    _BATCH = 500
    store = None
    for i in tqdm(range(0, len(docs), _BATCH), desc="Embedding doc chunks", unit="batch"):
        batch = docs[i : i + _BATCH]
        if store is None:
            store = Chroma.from_documents(
                documents=batch,
                embedding=embedder,
                collection_name=DOC_COLLECTION_NAME,
                persist_directory=persist_dir,
            )
        else:
            store.add_documents(batch)
    logger.info(f"Doc store ready: {len(docs)} chunks indexed")
    return store


def build_qa_store(qa_csv: str | None = None, persist_dir: str | None = None) -> Chroma:
    """
    Build the Q/A vector store from the pre-generated Q/A dataset CSV.

    Only the *question* text is embedded — the answer and source URL are
    stored as document metadata so they can be returned without an LLM call
    when the Stage 1 relevance threshold is met (see HYBRID_THRESHOLD in config).

    Parameters
    ----------
    qa_csv : str, optional
        Path to the Q/A dataset CSV.  Must have columns ``question``,
        ``answer``, and ``source_page``.  Defaults to QA_DATASET_PATH.
    persist_dir : str, optional
        Directory to persist the ChromaDB collection.  Defaults to
        ``<CHROMA_PERSIST_DIR>/qa``.

    Returns
    -------
    Chroma
        The built and persisted vector store.

    Raises
    ------
    FileNotFoundError
        If *qa_csv* does not exist.
    ValueError
        If *qa_csv* is missing required columns.
    """
    if qa_csv is None:
        qa_csv = QA_DATASET_PATH
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "qa")

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    if not os.path.exists(qa_csv):
        raise FileNotFoundError(f"{qa_csv} not found. Run qa_generator.py first.")

    df = pd.read_csv(qa_csv)
    required = {"question", "answer", "source_page"}
    if not required.issubset(df.columns):
        raise ValueError(f"qa_dataset.csv must have columns: {required}")

    # Embed only the question – answer stored as metadata
    docs = []
    for _, row in df.iterrows():
        docs.append(Document(
            page_content=str(row["question"]),
            metadata={
                "answer": str(row["answer"]),
                "source": str(row["source_page"]),
            },
        ))

    embedder = get_embedder()

    logger.info(f"Building Q&A store → {persist_dir} ({len(docs)} pairs)")
    _BATCH = 500
    store = None
    for i in tqdm(range(0, len(docs), _BATCH), desc="Embedding Q&A pairs", unit="batch"):
        batch = docs[i : i + _BATCH]
        if store is None:
            store = Chroma.from_documents(
                documents=batch,
                embedding=embedder,
                collection_name=QA_COLLECTION_NAME,
                persist_directory=persist_dir,
            )
        else:
            store.add_documents(batch)
    logger.info(f"Q&A store ready: {len(docs)} pairs indexed")
    return store


def load_doc_store(persist_dir: str | None = None) -> Chroma:
    """
    Load an existing documentation vector store from disk.

    Parameters
    ----------
    persist_dir : str, optional
        Path to the persisted ChromaDB collection.  Defaults to
        ``<CHROMA_PERSIST_DIR>/docs``.

    Returns
    -------
    Chroma
        The loaded vector store, ready for similarity search.

    Raises
    ------
    FileNotFoundError
        If *persist_dir* does not exist (store has not been built yet).
    """
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "docs")
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Doc store not found at '{persist_dir}'. Run build_doc_store() first."
        )
    return Chroma(
        collection_name=DOC_COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=persist_dir,
    )


def load_qa_store(persist_dir: str | None = None) -> Chroma:
    """
    Load an existing Q/A vector store from disk.

    Parameters
    ----------
    persist_dir : str, optional
        Path to the persisted ChromaDB collection.  Defaults to
        ``<CHROMA_PERSIST_DIR>/qa``.

    Returns
    -------
    Chroma
        The loaded vector store, ready for similarity search.

    Raises
    ------
    FileNotFoundError
        If *persist_dir* does not exist (store has not been built yet).
    """
    if persist_dir is None:
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, "qa")
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Q/A store not found at '{persist_dir}'. Run build_qa_store() first."
        )
    return Chroma(
        collection_name=QA_COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_doc_store()
    build_qa_store()
    print("\nBoth vector stores built successfully.")
    print(f"Persisted to: {CHROMA_PERSIST_DIR}/")
