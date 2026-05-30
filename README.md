# RAG-Powered Pandas Chatbot with Hybrid Q&A Retrieval

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions about the official [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) using a fully local [Ollama](https://ollama.com) pipeline — no cloud API keys required.

Built for **AGAI-03 Assignment 1 – Website-Specific RAG Chatbot using Scraping + Hybrid Retrieval**.

---

## What It Does

| Phase | Description |
|-------|-------------|
| **Scrape** | Downloads and cleans 23 pandas User Guide pages |
| **Generate** | Creates ~1,800 synthetic Q&A pairs via local Ollama (`llama3.1:8b`) |
| **Index** | Builds two ChromaDB vector stores (docs + Q&A pairs) |
| **Retrieve** | Hybrid retrieval: Q&A semantic match first, doc-chunk fallback |
| **Answer** | Ollama LLM synthesises grounded answers with source citations |
| **UI** | Streamlit chat interface with sidebar stats and confidence badges |

---

## Repository Structure

```
pandas-rag-chatbot/
├── app.py                        # Streamlit main application ← run this
├── config.py                     # Central settings: paths, thresholds, models
├── requirements.txt              # Python dependencies
├── .gitignore
├── project_report.docx           # Project report (source)
├── report.pdf                    # Project report (PDF deliverable)
│
├── src/
│   ├── scraper.py                # Web scraper (requests + BeautifulSoup4 + lxml)
│   ├── qa_generator.py           # Synthetic Q&A generator (local Ollama)
│   ├── vector_store.py           # ChromaDB indexing (docs + Q&A collections)
│   ├── retriever.py              # HybridRetriever with two-stage logic
│   └── chatbot.py                # OllamaChatbot orchestrator with memory
│
├── data/
│   ├── raw/                      # 23 scraped .txt files (one per pandas page)
│   └── processed/
│       └── qa_dataset_ollama.csv # ~1,347 generated Q&A pairs
│
└── chroma_db/                    # Persistent ChromaDB storage (docs + qa)
```

---

## Tech Stack

| Area | Technology |
|------|-----------|
| Language | Python 3.10+ |
| Web scraping | `requests`, `beautifulsoup4`, `lxml` |
| Data | `pandas`, `tqdm` |
| Embeddings | `multi-qa-MiniLM-L6-cos-v1` (sentence-transformers) |
| Vector DB | ChromaDB |
| RAG framework | LangChain |
| LLM (chatbot + Q&A generation) | Ollama `llama3.1:8b` (local) |
| UI | Streamlit |

---

## Setup & Installation

### 1. Clone & install dependencies

```bash
git clone https://github.com/eboekenh/pandas-rag-chatbot.git
cd pandas-rag-chatbot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```bash
ollama serve                    # start the Ollama daemon
ollama pull llama3.1:8b         # chatbot + Q&A generation model
```

> No API keys are needed — everything runs locally.

### 3. Run the app

The repository already ships with the scraped data, the Q&A dataset and a
pre-built `chroma_db/`, so you can launch directly:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Rebuilding the Pipeline from Scratch (optional)

Run each step in order. Skip any step whose output already exists.

### Step 1 – Scrape pandas documentation

```bash
python -m src.scraper
```

Downloads and cleans 23 pages from `pandas.pydata.org/docs/user_guide/` into `data/raw/`.

### Step 2 – Generate Q&A pairs

```bash
python -m src.qa_generator
```

Saves Q&A pairs to `data/processed/qa_dataset_ollama.csv` (columns: `question`, `answer`, `source_page`).

### Step 3 – Build vector stores

```bash
python -m src.vector_store
```

Creates two ChromaDB collections in `chroma_db/`:
- `pandas_docs` – chunked raw documentation (chunk size 1,000, overlap 200)
- `pandas_qa` – embedded Q&A pairs (question → embedding, answer → metadata)

### Step 4 – Launch the chatbot

```bash
streamlit run app.py
```

---

## Hybrid Retrieval Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────┐
│  Query expansion (optional)      │
│  Ollama rephrases the query into │
│  2 doc-style variants to improve │
│  recall on colloquial phrasing   │
└──────────────┬───────────────────┘
               ▼
┌─────────────────────────────┐
│  Stage 1: Q&A Store Search  │
│  (pandas_qa ChromaDB coll.) │
│  best relevance score ≥ 0.6 │
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     │ YES        │ NO
     ▼            ▼
qa_match mode   ┌─────────────────────────────┐
matched Q&A     │  Stage 2: Doc Store Search  │
pairs → LLM     │  (pandas_docs, top-k=5)     │
(stored answer  │  Build context prompt       │
 as fallback)   │  → Ollama LLM generates     │
                │    grounded answer          │
                └─────────────────────────────┘
                       (doc_search mode)
```

In **both** modes the Ollama LLM produces the final answer: in `qa_match` mode
it is grounded on the matched Q&A pairs (with the stored answer as a fallback
if Ollama is unavailable); in `doc_search` mode it is grounded on the retrieved
documentation chunks.

**Embedding model:** `multi-qa-MiniLM-L6-cos-v1`
**Match threshold:** 0.6 (configurable via `HYBRID_THRESHOLD` in `config.py`)
**Doc chunk size:** 1,000 characters, 200 overlap

> **Note on the score:** the relevance score is LangChain's normalised score
> (`1 − L2_distance / √2`) over L2-normalised embeddings, in the range `[0, 1]`.
> It behaves like a semantic-similarity score but is not raw cosine similarity.

---

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HYBRID_THRESHOLD` | `0.6` | Min relevance score to use a Q&A match |
| `TOP_K_DOCS` | `5` | Number of doc chunks for fallback retrieval |
| `CHUNK_SIZE` | `1000` | Characters per doc chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `multi-qa-MiniLM-L6-cos-v1` | Sentence-transformers model |
| `MAX_HISTORY_TURNS` | `6` | Rolling conversation memory window |
| `OLLAMA_QA_MODEL` | `llama3.1:8b` | Model for chatbot + Q&A generation |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature for the chatbot |
| `SCRAPE_DELAY` | `2` | Seconds between HTTP requests |

---

## Pages Scraped (23 total)

| # | Section | URL slug |
|---|---------|----------|
| 1 | 10 Minutes to pandas | `10min.html` |
| 2 | Intro to Data Structures | `dsintro.html` |
| 3 | Essential Basic Functionality | `basics.html` |
| 4 | IO Tools | `io.html` |
| 5 | Indexing & Selecting Data | `indexing.html` |
| 6 | MultiIndex / Advanced Indexing | `advanced.html` |
| 7 | Merge, Join, Concatenate | `merging.html` |
| 8 | Reshaping & Pivot Tables | `reshaping.html` |
| 9 | Working with Text Data | `text.html` |
| 10 | Working with Missing Data | `missing_data.html` |
| 11 | Duplicate Labels | `duplicates.html` |
| 12 | Categorical Data | `categorical.html` |
| 13 | Chart Visualisation | `visualization.html` |
| 14 | Group By: Split-Apply-Combine | `groupby.html` |
| 15 | Windowing Operations | `window.html` |
| 16 | Time Series / Date Functionality | `timeseries.html` |
| 17 | Time Deltas | `timedeltas.html` |
| 18 | Options & Settings | `options.html` |
| 19 | Enhancing Performance | `enhancingperf.html` |
| 20 | Scaling to Large Datasets | `scale.html` |
| 21 | Sparse Data Structures | `sparse.html` |
| 22 | Frequently Asked Questions | `gotchas.html` |
| 23 | Cookbook | `cookbook.html` |

---

## Chatbot UI Features

- **Chat bubbles** – user right-aligned, assistant left-aligned
- **Retrieval mode badge** – `Q&A Match` (green) or `Doc Search` (blue) with confidence %
- **Source citation** – URL of the pandas page used
- **Matched question** – expander shown when in `qa_match` mode
- **Sidebar stats** – live pages-scraped, Q&A pair count, doc chunk count, memory turns
- **Sample questions** – one-click example queries
- **Q&A Preview** – expandable random sample from the dataset
- **Clear Chat** – resets conversation memory
- **Out-of-scope guard** – non-pandas questions get a polite refusal (no hallucination)
- **Ollama status guard** – blocks startup if Ollama isn't running

---

## Data Assets

| Asset | Location | Description |
|-------|----------|-------------|
| Raw docs | `data/raw/*.txt` | 23 cleaned text files |
| Q&A dataset | `data/processed/qa_dataset_ollama.csv` | ~1,347 generated Q&A pairs |
| Vector store | `chroma_db/` | ChromaDB persistent storage (docs + qa) |

---

## Limitations

- Static dataset — no automatic update when pandas releases a new version
- Local Ollama inference is slower than cloud APIs (CPU-only, ~5–15 s per response)
- Hybrid threshold (0.6) was set empirically; formal optimisation would require a labelled eval set
- Coverage limited to 23 selected User Guide pages; API Reference not included
- In-memory conversation history — lost on page refresh

---

## Future Improvements

- Cross-encoder reranking for better doc-store precision
- BM25 + vector fusion (Reciprocal Rank Fusion)
- RAGAS evaluation framework for systematic quality measurement
- Persistent chat sessions (SQLite)
- Expand to API Reference and other Python data science libraries

---

## License

Documentation scraped from [pandas.pydata.org](https://pandas.pydata.org) is published under the **BSD 3-Clause License**. This project is for educational use (AGAI-03 Assignment 1).
