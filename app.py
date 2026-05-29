"""Streamlit front-end for the pandas RAG chatbot.

Run with:
    streamlit run app.py

This is a thin UI layer: all retrieval and generation logic lives in
``src/chatbot.py`` (``OllamaChatbot``).  The app streams the LLM response
token-by-token and renders retrieval-mode badges, source citations and a
sidebar with project info and live statistics.  All inference runs locally
through Ollama — no API key required.
"""
import os
import socket
import sys

import streamlit as st
import pandas as pd

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Pandas RAG Chatbot",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

from config import OLLAMA_BASE_URL, OLLAMA_QA_MODEL, EMBEDDING_MODEL, MAX_HISTORY_TURNS
from src.chatbot import OllamaChatbot, extract_source_url


# ── Ollama availability check ─────────────────────────────────────────────────
def _ollama_running() -> bool:
    """Return True if Ollama is reachable on its configured host/port."""
    try:
        netloc = OLLAMA_BASE_URL.split("//", 1)[-1]
        host = netloc.split(":")[0]
        port = int(netloc.rsplit(":", 1)[-1]) if ":" in netloc else 11434
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# ── Cached resource loader ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading chatbot (embedding model + vector stores + LLM)...")
def load_chatbot() -> OllamaChatbot:
    return OllamaChatbot()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []


# ── Ollama guard — show error before loading heavy resources ──────────────────
if not _ollama_running():
    st.error(
        "**Ollama is not running.**\n\n"
        "Please start it first:\n"
        "```bash\n"
        "ollama serve\n"
        f"ollama pull {OLLAMA_QA_MODEL}\n"
        "```\n"
        "Then reload this page.",
        icon="🐼",
    )
    st.stop()


# ── Load resources (cached after first run) ───────────────────────────────────
bot = load_chatbot()
stats = bot.get_stats()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐼 Pandas RAG Chatbot")
    st.caption(f"Local LLM · Ollama {OLLAMA_QA_MODEL} · Hybrid RAG")
    st.divider()

    st.markdown("**Website:**")
    st.markdown("[pandas User Guide ↗](https://pandas.pydata.org/docs/user_guide/index.html)")
    st.divider()

    st.markdown("**🟢 Ollama Status**")
    st.success(f"Connected · {OLLAMA_QA_MODEL}")
    st.divider()

    # Live statistics
    st.markdown("**📊 Statistics**")
    c1, c2 = st.columns(2)
    c1.metric("Pages scraped", stats["pages_scraped"])
    c2.metric("Q&A pairs", stats["qa_pairs"])
    c3, c4 = st.columns(2)
    c3.metric("Doc chunks", stats["doc_chunks"])
    c4.metric("Memory turns", len(st.session_state.memory) // 2)
    st.divider()

    # Sample questions
    st.markdown("**💡 Sample Questions**")
    sample_questions = [
        "How do I create a DataFrame from a dictionary?",
        "What is the difference between merge and join?",
        "How do I handle missing values in pandas?",
        "How does groupby work in pandas?",
        "How do I filter rows based on a condition?",
        "What is a MultiIndex?",
        "How do I sort a DataFrame?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"s_{q[:20]}"):
            st.session_state.pending_input = q

    st.divider()

    # Dataset sample viewer
    qa_path = "data/processed/qa_dataset_ollama.csv"
    if os.path.exists(qa_path):
        with st.expander("📄 View Sample Q&A Pairs"):
            try:
                df = pd.read_csv(qa_path)
                for _, row in df.sample(min(5, len(df))).reset_index(drop=True).iterrows():
                    st.markdown(f"**Q:** {row['question']}")
                    a = str(row["answer"])
                    st.caption(f"A: {a[:200]}{'...' if len(a) > 200 else ''}")
                    st.divider()
            except Exception as e:
                st.error(f"Could not load Q&A file: {e}")

    st.divider()

    with st.expander("ℹ️ About"):
        st.markdown(f"""
**Hybrid Retrieval:**
1. Searches the Q&A store first (semantic relevance score).
2. Falls back to full document search if no confident match is found.
3. The LLM synthesises the final answer from the retrieved context.

**Models:**
- LLM: Ollama {OLLAMA_QA_MODEL}
- Embeddings: {EMBEDDING_MODEL}

**Vector DB:** ChromaDB (2 collections: pandas_qa, pandas_docs)
        """)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🐼 Ask me anything about pandas")
st.caption(f"Powered by RAG · Hybrid Q&A + Document Retrieval · Ollama {OLLAMA_QA_MODEL}")

if not st.session_state.messages:
    st.info(
        "👋 Hi! I can answer questions about the **pandas library** based on the official User Guide. "
        "I run **entirely on your local machine** via Ollama — no API key needed.\n\n"
        "Try asking: *How do I read a CSV file?* or *What is a DataFrame?*"
    )


def _render_meta(meta: dict) -> None:
    """Render the retrieval-mode badge, source citation and matched-question expander."""
    cols = st.columns([1, 1, 2])
    if meta.get("mode") == "qa_match":
        cols[0].success(f"✅ Q&A Match ({meta['confidence']:.0%})")
    else:
        cols[0].info(f"🔍 Doc Search ({meta['confidence']:.0%})")
    if meta.get("sources"):
        cols[1].caption("Source: " + " ".join(f"[↗]({s})" for s in meta["sources"]))
    if meta.get("matched_question"):
        with st.expander("🔗 Matched Q&A"):
            st.caption(f"**Matched question:** {meta['matched_question']}")


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])

# Handle sample-question button clicks
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Ask about pandas...") or pending

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Pre-save user message so follow-up context is available on the next turn.
    st.session_state.memory.append({"role": "user", "content": user_input})

    # Prepare retrieval using prior turns only (exclude the just-added message).
    messages, meta = bot.prepare(user_input, st.session_state.memory[:-1])

    with st.chat_message("assistant"):
        if messages is None:
            answer = "I don't have enough information about that in the pandas documentation."
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(chunk.content for chunk in bot.llm.stream(messages))
                extracted = extract_source_url(answer)
                if extracted:
                    meta["sources"] = extracted
            except Exception as exc:
                answer = f"Ollama error: {exc}"
                st.error(answer)
        _render_meta(meta)

    # Save assistant response and trim memory.
    st.session_state.memory.append({"role": "assistant", "content": answer})
    if len(st.session_state.memory) > MAX_HISTORY_TURNS * 2:
        del st.session_state.memory[:2]

    # Save to history for replay on next rerun.
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    st.rerun()
