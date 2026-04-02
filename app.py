"""Streamlit UI for the 6-Layer Agentic RAG Pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import glob as globmod
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Pipeline",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Available model choices per provider ─────────────────────────────────────
LLM_CHOICES: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
    "ollama": ["llama3.1", "llama3.2", "qwen2.5", "mistral", "phi3", "gemma2"],
}

EMBEDDING_CHOICES: dict[str, list[str]] = {
    "openai": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    "gemini": ["models/text-embedding-004", "models/embedding-001"],
    "ollama": ["Qwen3-Embedding-0.6B", "nomic-embed-text", "mxbai-embed-large", "all-minilm"],
}

RERANKER_CHOICES = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


# ── Sidebar: Configuration ──────────────────────────────────────────────────
def render_sidebar() -> dict:
    """Render the config sidebar and return the chosen settings dict."""
    with st.sidebar:
        st.title("Settings")

        # ── Provider ──
        st.header("Provider")
        provider = st.selectbox(
            "LLM & Embedding Provider",
            options=["openai", "gemini", "ollama"],
            index=0,
            help="All LLM calls and embeddings will use this provider.",
        )

        # ── LLM Model ──
        st.header("LLM Model")
        llm_options = LLM_CHOICES.get(provider, [])
        llm_model = st.selectbox("Chat model", options=llm_options, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

        # ── Embedding Model ──
        st.header("Embedding Model")
        emb_options = EMBEDDING_CHOICES.get(provider, [])
        embedding_model = st.selectbox(
            "Embedding model",
            options=emb_options,
            index=0,
            help="Used for document indexing and dense retrieval.",
        )

        # ── Reranker ──
        st.header("Reranker")
        reranker = st.selectbox("Cross-encoder model", options=RERANKER_CHOICES, index=0)

        # ── Retrieval ──
        st.header("Retrieval")
        col1, col2 = st.columns(2)
        with col1:
            retrieval_top_k = st.number_input("Dense top-k", 1, 100, 20)
            bm25_top_k = st.number_input("BM25 top-k", 1, 100, 20)
        with col2:
            rerank_top_k = st.number_input("Rerank top-k", 1, 50, 5)
            max_retries = st.number_input("Max retries", 0, 5, 2)

        # ── Chunking ──
        st.header("Chunking")
        chunk_size = st.number_input("Chunk size (tokens)", 64, 2048, 256, step=32)
        chunk_overlap = st.number_input("Chunk overlap (tokens)", 0, 512, 32, step=8)

        # ── Evaluation ──
        st.header("Evaluation")
        enable_ragas = st.toggle("Enable Ragas evaluation", value=True)

        # ── Data ──
        st.header("Data")
        data_dir = st.text_input("Documents directory", value="./data")

        st.divider()
        reindex = st.button("Re-index documents", type="secondary", use_container_width=True)

    return {
        "provider": provider,
        "llm_model": llm_model,
        "temperature": temperature,
        "embedding_model": embedding_model,
        "reranker": reranker,
        "retrieval_top_k": retrieval_top_k,
        "bm25_top_k": bm25_top_k,
        "rerank_top_k": rerank_top_k,
        "max_retries": max_retries,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "enable_ragas": enable_ragas,
        "data_dir": data_dir,
        "reindex": reindex,
    }


# ── Apply config to global settings ─────────────────────────────────────────
def apply_config(cfg: dict) -> None:
    """Push sidebar selections into the runtime Settings singleton."""
    from config import settings

    provider = cfg["provider"]
    settings.default_llm_provider = provider
    settings.default_embedding_provider = provider

    # LLM model
    if provider == "openai":
        settings.openai_model = cfg["llm_model"]
        settings.openai_embedding_model = cfg["embedding_model"]
    elif provider == "gemini":
        settings.gemini_model = cfg["llm_model"]
        settings.gemini_embedding_model = cfg["embedding_model"]
    elif provider == "ollama":
        settings.ollama_model = cfg["llm_model"]
        settings.ollama_embedding_model = cfg["embedding_model"]

    settings.reranker_model = cfg["reranker"]
    settings.retrieval_top_k = cfg["retrieval_top_k"]
    settings.bm25_top_k = cfg["bm25_top_k"]
    settings.rerank_top_k = cfg["rerank_top_k"]
    settings.max_retries = cfg["max_retries"]


# ── Build / cache store & graph ──────────────────────────────────────────────
def _config_fingerprint(cfg: dict) -> str:
    """Return a string that changes when store-affecting config changes."""
    return f"{cfg['provider']}|{cfg['embedding_model']}|{cfg['data_dir']}"


def get_store(cfg: dict, force_reindex: bool = False):
    """Return a DocumentStore, re-creating when provider/embedding changes."""
    from indexing.store import DocumentStore

    fingerprint = _config_fingerprint(cfg)
    prev = st.session_state.get("_store_fingerprint")

    if force_reindex or prev != fingerprint or "store" not in st.session_state:
        store = DocumentStore()

        # Ingest documents
        data_dir = cfg["data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        documents = _load_documents(data_dir)

        if documents:
            from indexing.chunker import recursive_chunk

            chunks = recursive_chunk(
                documents,
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
            )
            store.add_documents(chunks)
            st.session_state["_index_stats"] = {
                "docs": len(documents),
                "chunks": len(chunks),
            }
        else:
            st.session_state["_index_stats"] = {"docs": 0, "chunks": 0}

        st.session_state["store"] = store
        st.session_state["_store_fingerprint"] = fingerprint
        # Invalidate graph cache
        st.session_state.pop("graph", None)

    return st.session_state["store"]


def get_graph(cfg: dict):
    """Return a compiled LangGraph, rebuilding if needed."""
    from pipeline.graph import build_rag_graph

    if "graph" not in st.session_state:
        store = st.session_state["store"]
        st.session_state["graph"] = build_rag_graph(
            store, enable_ragas=cfg["enable_ragas"],
        )
    return st.session_state["graph"]


# ── Document loading (reuse logic from main.py) ─────────────────────────────
def _load_documents(data_dir: str) -> list:
    from langchain_core.documents import Document

    docs: list[Document] = []
    loaders = {
        "**/*.pdf": _load_pdf,
        "**/*.txt": _load_text,
        "**/*.md": _load_text,
        "**/*.docx": _load_docx,
    }
    for pattern, fn in loaders.items():
        for fp in globmod.glob(os.path.join(data_dir, pattern), recursive=True):
            try:
                docs.extend(fn(fp))
            except Exception:
                pass
    return docs


def _load_pdf(path: str):
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(path).load()


def _load_text(path: str):
    from langchain_core.documents import Document
    with open(path, "r", encoding="utf-8") as f:
        return [Document(page_content=f.read(), metadata={"source": path})]


def _load_docx(path: str):
    from langchain_core.documents import Document
    import docx2txt
    return [Document(page_content=docx2txt.process(path), metadata={"source": path})]


# ── Main UI ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("6-Layer Agentic RAG Pipeline")
    st.caption("HyDE -> Lazy Router -> Hybrid Retrieval -> Distillation -> Critique -> Output")

    # Sidebar config
    cfg = render_sidebar()
    apply_config(cfg)

    # Initialise store & graph
    with st.spinner("Initialising index & pipeline..."):
        store = get_store(cfg, force_reindex=cfg["reindex"])
        graph = get_graph(cfg)

    # Index stats bar
    stats = st.session_state.get("_index_stats", {})
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Provider", cfg["provider"].upper())
    col_b.metric("Embedding", cfg["embedding_model"].split("/")[-1])
    col_c.metric("Documents", stats.get("docs", "—"))
    col_d.metric("Chunks", stats.get("chunks", "—"))

    st.divider()

    # ── Chat interface ───────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                _render_citations(msg["citations"])
            if msg["role"] == "assistant" and msg.get("ragas"):
                _render_ragas(msg["ragas"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                _render_metadata(msg["metadata"])

    # Input
    if query := st.chat_input("Ask a question..."):
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result_state = _run_query(graph, query, cfg)

            # Answer
            if result_state.should_abstain:
                st.error(result_state.answer)
            else:
                answer_text = result_state.answer.split("\n--- Sources ---")[0]
                st.markdown(answer_text)

            citations = result_state.citations or []
            ragas = result_state.ragas_scores or {}
            metadata = {
                "retries": result_state.retry_count,
                "sub_questions": result_state.sub_questions,
                "abstained": result_state.should_abstain,
            }

            if citations and not result_state.should_abstain:
                _render_citations(citations)
            if ragas:
                _render_ragas(ragas)
            if result_state.retry_count > 0 or result_state.sub_questions:
                _render_metadata(metadata)

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": result_state.answer.split("\n--- Sources ---")[0],
            "citations": citations,
            "ragas": ragas,
            "metadata": metadata,
        })


def _run_query(graph, query: str, cfg: dict):
    """Execute the pipeline graph and return PipelineState."""
    from pipeline.state import PipelineState

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.get("messages", [])
    ]

    state = PipelineState(
        original_query=query,
        conversation_history=history[-20:],
    )
    result = graph.invoke(state.model_dump())
    return PipelineState(**result)


def _render_citations(citations: list[dict]) -> None:
    with st.expander("Sources", expanded=False):
        for cite in citations:
            score = cite.get("rerank_score", 0)
            name = Path(cite.get("source", "unknown")).name
            st.markdown(
                f"**[{cite['source_id']}]** `{name}` — "
                f"relevance: `{score:.3f}`\n\n"
                f"> {cite.get('snippet', '')[:200]}..."
            )


def _render_ragas(scores: dict) -> None:
    if scores.get("abstained"):
        return
    with st.expander("Quality Metrics (Ragas)", expanded=False):
        cols = st.columns(len([k for k in scores if k != "abstained"]))
        for col, (metric, score) in zip(
            cols, ((k, v) for k, v in scores.items() if k != "abstained")
        ):
            col.metric(metric, f"{score:.3f}")


def _render_metadata(meta: dict) -> None:
    parts = []
    if meta.get("retries", 0) > 0:
        parts.append(f"Retries: {meta['retries']}")
    if meta.get("sub_questions"):
        parts.append(f"Sub-questions: {', '.join(meta['sub_questions'])}")
    if meta.get("abstained"):
        parts.append("Abstained: yes")
    if parts:
        st.caption(" | ".join(parts))


if __name__ == "__main__":
    main()
