import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
QA_DATASET_PATH = "data/processed/qa_dataset_ollama.csv"
CHROMA_PERSIST_DIR = "chroma_db"

# --- Chroma Collections ---
DOC_COLLECTION_NAME = "pandas_docs"
QA_COLLECTION_NAME = "pandas_qa"

# --- Embedding ---
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
HYBRID_THRESHOLD = 0.6
TOP_K_DOCS = 5


# --- Ollama (used by qa_generator, retriever and chatbot) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_QA_MODEL = "llama3.1:8b"

# --- LLM (Ollama, used by chatbot) ---
LLM_TEMPERATURE = 0.3

# --- Chat Memory ---
MAX_HISTORY_TURNS = 6

# --- Scraper ---
SCRAPE_DELAY = 2  # seconds between requests

SCRAPE_TARGETS = [
    "https://pandas.pydata.org/docs/user_guide/10min.html",
    "https://pandas.pydata.org/docs/user_guide/dsintro.html",
    "https://pandas.pydata.org/docs/user_guide/basics.html",
    "https://pandas.pydata.org/docs/user_guide/io.html",
    "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "https://pandas.pydata.org/docs/user_guide/advanced.html",
    "https://pandas.pydata.org/docs/user_guide/merging.html",
    "https://pandas.pydata.org/docs/user_guide/reshaping.html",
    "https://pandas.pydata.org/docs/user_guide/text.html",
    "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "https://pandas.pydata.org/docs/user_guide/duplicates.html",
    "https://pandas.pydata.org/docs/user_guide/categorical.html",
    "https://pandas.pydata.org/docs/user_guide/visualization.html",
    "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "https://pandas.pydata.org/docs/user_guide/window.html",
    "https://pandas.pydata.org/docs/user_guide/timeseries.html",
    "https://pandas.pydata.org/docs/user_guide/timedeltas.html",
    "https://pandas.pydata.org/docs/user_guide/options.html",
    "https://pandas.pydata.org/docs/user_guide/enhancingperf.html",
    "https://pandas.pydata.org/docs/user_guide/scale.html",
    "https://pandas.pydata.org/docs/user_guide/sparse.html",
    "https://pandas.pydata.org/docs/user_guide/gotchas.html",
    "https://pandas.pydata.org/docs/user_guide/cookbook.html",
]
