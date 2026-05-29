"""RAG chatbot orchestrator for the pandas documentation assistant.

Exposes a single ``OllamaChatbot`` class that ties together the two-stage
hybrid retriever and a locally-running Ollama LLM:

* **Stage 1 (qa_match)** – ``HybridRetriever`` finds one or more matches in
  the pre-built Q/A store whose relevance score meets ``HYBRID_THRESHOLD``.
  The matched Q/A pairs are passed to the LLM as high-confidence context to
  produce a polished answer.  If the LLM is unavailable, the stored answer is
  returned directly as a fallback.
* **Stage 2 (doc_search)** – The retriever falls back to the document store,
  and the top-k chunks are forwarded to the Ollama LLM to synthesise a
  grounded answer with a source citation.

Conversation history is kept in ``self.memory`` and injected into every LLM
prompt up to ``MAX_HISTORY_TURNS`` turns.  All generation runs locally through
Ollama – no cloud API keys are required.

The Streamlit UI (``app.py``) imports this module and uses ``prepare()`` for
token-by-token streaming; ``chat()`` is a convenience method that returns a
fully-generated answer in one call (useful for testing the pipeline).
"""
import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_QA_MODEL,
    LLM_TEMPERATURE,
    MAX_HISTORY_TURNS,
    QA_DATASET_PATH,
    RAW_DATA_DIR,
)

from langchain_core.documents import Document
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# A doc_search match whose best chunk scores below this is treated as
# "no relevant documentation found" (out-of-scope guard).
DOC_SEARCH_MIN_CONFIDENCE = 0.20

SYSTEM_PROMPT = """You are a helpful pandas expert assistant.

## Your job
Answer the user's CURRENT question using the Documentation Context provided below.
The Chat History is only for resolving pronouns like "it", "this", "that" — do NOT let previous topics override the current question.

## Rules
1. Always answer the CURRENT question. Ignore unrelated history.
2. Give a clear, detailed explanation — not just a one-liner.
3. Only include code examples if the Documentation Context explicitly contains them. Never invent output values.
4. Use simple language so beginners can understand.
5. If the Documentation Context does not contain enough information, say: "I don't have enough information about that in the pandas documentation."
6. End your answer with: Source: <url>"""

# Pandas-specific terms used to decide whether a query is in-scope and whether
# a short query is a follow-up that needs prior context prepended.
PANDAS_KEYWORDS = {
    "pandas", "dataframe", "df", "series", "column", "columns", "row", "rows",
    "index", "merge", "join", "concat", "groupby", "group", "sort", "filter",
    "read_csv", "read_excel", "to_csv", "iloc", "loc", "apply", "map", "lambda",
    "pivot", "melt", "stack", "unstack", "resample", "rolling", "shift", "diff",
    "fillna", "dropna", "isna", "isnull", "notnull", "duplicated", "drop_duplicates",
    "rename", "reset_index", "set_index", "astype", "dtypes", "dtype", "shape",
    "values", "head", "tail", "describe", "info", "value_counts", "unique",
    "nunique", "aggregate", "agg", "transform", "explode", "crosstab",
    "data", "dataset", "table", "cell", "missing", "nan", "null", "csv", "excel",
    "boolean", "mask", "where", "query", "eval", "cut", "qcut", "multiindex",
}


def is_followup(query: str) -> bool:
    """True if *query* is a short message without pandas-specific terms.

    Short messages such as "what about the second one?" rely on the previous
    turn for meaning, so the retriever prepends the last user topic.
    """
    if len(query.split()) >= 8:
        return False
    return not any(kw in query.lower() for kw in PANDAS_KEYWORDS)


def is_pandas_related(query: str) -> bool:
    """True if *query* contains at least one pandas-specific keyword."""
    words = set(query.lower().replace("?", "").replace(",", "").split())
    return bool(words & PANDAS_KEYWORDS)


def extract_source_url(answer: str) -> list[str]:
    """Extract the single Source URL the LLM cited at the end of its answer."""
    match = re.search(r"[Ss]ource:\s*(https?://\S+)", answer)
    if match:
        return [match.group(1).rstrip(".,)>")]
    return []


class OllamaChatbot:
    """Retrieval-augmented generation chatbot for pandas documentation queries.

    Combines a two-stage hybrid retriever with a local Ollama LLM and a
    rolling conversation-memory window.

    Attributes
    ----------
    retriever : HybridRetriever
        Handles Q/A-store lookup and doc-store RAG fallback.
    llm : ChatOllama
        Local Ollama LLM used for answer synthesis in both modes.
    memory : list[dict]
        Conversation history as a list of ``{role, content}`` dicts,
        capped at ``MAX_HISTORY_TURNS * 2`` entries.
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = self._load_llm()
        self.memory: list[dict] = []

    @staticmethod
    def _load_llm():
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_QA_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_doc_messages(self, query: str, context_docs: list[Document], memory: list[dict]) -> list[dict]:
        """Assemble messages for doc_search mode (raw doc chunks as context)."""
        context_blocks = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_blocks.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        context_text = "\n\n".join(context_blocks)

        user_content = (
            f"## Documentation Context\n{context_text}\n\n"
            f"## Current Question\n{query}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(memory[-(MAX_HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_qa_messages(self, query: str, qa_matches: list, memory: list[dict]) -> list[dict]:
        """Assemble messages for qa_match mode (matched Q/A pairs as context).

        ``qa_matches`` is a list of ``(Document, score)`` tuples, all with a
        relevance score >= ``HYBRID_THRESHOLD``.
        """
        context_blocks = []
        for i, (doc, score) in enumerate(qa_matches, 1):
            q = doc.page_content
            a = doc.metadata.get("answer", "")
            src = doc.metadata.get("source", "")
            context_blocks.append(
                f"[Match {i} | Score: {score:.2f} | Source: {src}]\n"
                f"Q: {q}\n"
                f"A: {a}"
            )
        context_text = "\n\n".join(context_blocks)
        user_content = (
            f"Context (top relevant Q&A pairs from documentation):\n{context_text}\n\n"
            f"Question: {query}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(memory[-(MAX_HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieval_query(self, query: str, memory: list[dict]) -> str:
        """For follow-ups, prepend the last user topic to improve retrieval."""
        if not is_followup(query):
            return query
        for msg in reversed(memory):
            if msg["role"] == "user":
                return f"{msg['content']} {query}"
        return query

    def prepare(self, query: str, memory: list[dict] | None = None) -> tuple[list[dict] | None, dict]:
        """Run hybrid retrieval and build the LLM messages WITHOUT calling the LLM.

        Returns ``(messages, meta)``.  ``messages`` is ``None`` when the query
        is out of scope (the caller should then return the standard refusal).
        ``meta`` carries the retrieval mode, confidence, sources and the
        matched question (for qa_match).  This split lets the UI stream the
        response token-by-token via ``self.llm.stream(messages)``.
        """
        if memory is None:
            memory = self.memory

        has_prior_context = is_followup(query) and any(m["role"] == "user" for m in memory)
        if not is_pandas_related(query) and not has_prior_context:
            return None, {"mode": "doc_search", "confidence": 0.0, "sources": [], "matched_question": None}

        retrieval = self.retriever.retrieve(self._retrieval_query(query, memory))
        confidence = retrieval["confidence"]
        doc_confidence = retrieval.get("doc_confidence")

        if retrieval["mode"] == "qa_match":
            qa_matches = retrieval["qa_matches"]
            messages = self._build_qa_messages(query, qa_matches, memory)
            seen, sources = set(), []
            for doc, _ in qa_matches:
                src = doc.metadata.get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    sources.append(src)
            meta = {
                "mode": "qa_match",
                "confidence": round(confidence, 3),
                "sources": sources,
                "matched_question": retrieval["matched_question"] or "",
            }
            return messages, meta

        # doc_search
        context_docs = retrieval["context_docs"]
        low_confidence = not context_docs or (doc_confidence or 0.0) < DOC_SEARCH_MIN_CONFIDENCE
        if low_confidence and not has_prior_context:
            messages, sources = None, []
        else:
            messages = self._build_doc_messages(query, context_docs, memory)
            sources = list({
                doc.metadata.get("source", "")
                for doc in context_docs
                if doc.metadata.get("source")
            })
        meta = {
            "mode": "doc_search",
            "confidence": round(doc_confidence or 0.0, 3),
            "sources": sources,
            "matched_question": None,
        }
        return messages, meta

    # ------------------------------------------------------------------
    # One-shot chat (non-streaming)
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> dict:
        """Process a user message and return a fully-generated answer.

        Runs the two-stage retrieval pipeline, calls the Ollama LLM, updates
        ``self.memory`` and returns a dict with keys ``answer``, ``sources``,
        ``mode``, ``confidence`` and ``matched_question``.
        """
        messages, meta = self.prepare(user_message, self.memory)

        if messages is None:
            answer = "I don't have enough information about that in the pandas documentation."
        else:
            try:
                answer = self.llm.invoke(messages).content
                extracted = extract_source_url(answer)
                if extracted:
                    meta["sources"] = extracted
            except Exception as exc:  # pragma: no cover - network/runtime guard
                logger.error("Ollama call failed: %s", exc)
                answer = "I'm sorry, I couldn't reach the local Ollama model. Is it running?"

        self.memory.append({"role": "user", "content": user_message})
        self.memory.append({"role": "assistant", "content": answer})
        if len(self.memory) > MAX_HISTORY_TURNS * 2:
            del self.memory[:2]

        return {
            "answer": answer,
            "sources": meta["sources"],
            "mode": meta["mode"],
            "confidence": meta["confidence"],
            "matched_question": meta["matched_question"],
        }

    def reset_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory = []

    def get_stats(self) -> dict:
        """Return best-effort runtime statistics about the chatbot's data.

        Each lookup is wrapped in a silent try/except so a failure in one does
        not affect the others.  Returns counts of Q/A pairs, doc chunks, pages
        scraped and conversation turns held in memory.
        """
        qa_count = doc_count = page_count = 0

        try:
            import pandas as pd
            if os.path.exists(QA_DATASET_PATH):
                qa_count = len(pd.read_csv(QA_DATASET_PATH))
        except Exception:
            pass

        try:
            if os.path.exists(RAW_DATA_DIR):
                page_count = len([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")])
        except Exception:
            pass

        try:
            doc_count = self.retriever.doc_store._collection.count()
        except Exception:
            pass

        return {
            "qa_pairs": qa_count,
            "doc_chunks": doc_count,
            "pages_scraped": page_count,
            "memory_turns": len(self.memory) // 2,
        }
