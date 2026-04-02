"""FastAPI backend for the 6-Layer Agentic RAG Pipeline.

Endpoints
---------
POST /api/configure        -- push provider/model/param settings
POST /api/ingest           -- upload & index files
GET  /api/status           -- index + provider status
POST /api/query            -- run pipeline and return answer
POST /api/clear            -- clear conversation history
DELETE /api/index          -- wipe vector store

Run with:
    uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import glob as globmod
import io
import os
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG Pipeline API",
    description="6-Layer Agentic RAG: HyDE → Router → Hybrid Retrieval → Distillation → Critique → Output",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state ─────────────────────────────────────────────────
_state: dict[str, Any] = {
    "store": None,
    "graph": None,
    "config": {
        "provider": "openai",
        "llm_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "temperature": 0.0,
        "reranker": "BAAI/bge-reranker-v2-m3",
        "retrieval_top_k": 20,
        "bm25_top_k": 20,
        "rerank_top_k": 5,
        "max_retries": 2,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "enable_ragas": True,
    },
    "index_stats": {"docs": 0, "chunks": 0},
    "conversation_history": [],
}

# ── Schemas ──────────────────────────────────────────────────────────────────

class ConfigRequest(BaseModel):
    provider: str = "openai"
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    reranker: str = "BAAI/bge-reranker-v2-m3"
    retrieval_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5
    max_retries: int = 2
    chunk_size: int = 256
    chunk_overlap: int = 32
    enable_ragas: bool = True


class QueryRequest(BaseModel):
    query: str
    use_history: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _apply_settings(cfg: dict) -> None:
    """Push config dict into the runtime Settings singleton."""
    from config import settings

    provider = cfg["provider"]
    settings.default_llm_provider = provider
    settings.default_embedding_provider = provider

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


def _get_store(force_rebuild: bool = False):
    """Return (and cache) a DocumentStore."""
    from indexing.store import DocumentStore

    if _state["store"] is None or force_rebuild:
        _apply_settings(_state["config"])
        _state["store"] = DocumentStore()
        _state["graph"] = None  # invalidate graph
    return _state["store"]


def _get_graph():
    """Return (and cache) the compiled LangGraph."""
    from pipeline.graph import build_rag_graph

    if _state["graph"] is None:
        store = _get_store()
        _state["graph"] = build_rag_graph(
            store, enable_ragas=_state["config"]["enable_ragas"]
        )
    return _state["graph"]


def _load_document(path: str) -> list:
    """Load a single file into LangChain Documents."""
    from langchain_core.documents import Document

    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(path).load()
    elif ext == ".docx":
        import docx2txt
        content = docx2txt.process(path)
        return [Document(page_content=content, metadata={"source": path})]
    else:  # .txt, .md, etc.
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [Document(page_content=f.read(), metadata={"source": path})]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return current provider, model, and index statistics."""
    cfg = _state["config"]
    return {
        "provider": cfg["provider"],
        "llm_model": cfg["llm_model"],
        "embedding_model": cfg["embedding_model"],
        "reranker": cfg["reranker"],
        "enable_ragas": cfg["enable_ragas"],
        "index": _state["index_stats"],
        "ready": True,
    }


@app.post("/api/configure")
def configure(req: ConfigRequest):
    """Update provider/model settings.  Rebuilds graph if provider changes."""
    old_provider = _state["config"]["provider"]
    old_embedding = _state["config"]["embedding_model"]

    _state["config"].update(req.model_dump())
    _apply_settings(_state["config"])

    # Invalidate store/graph when provider or embedding model changes
    if req.provider != old_provider or req.embedding_model != old_embedding:
        _state["store"] = None
        _state["graph"] = None

    # Invalidate graph when ragas setting changes
    if _state["graph"] is not None and req.enable_ragas != _state["config"].get("enable_ragas"):
        _state["graph"] = None

    return {"ok": True, "config": _state["config"]}


@app.post("/api/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(256),
    chunk_overlap: int = Form(32),
):
    """Upload files, chunk them, and index into the vector store."""
    from indexing.chunker import recursive_chunk

    _state["config"]["chunk_size"] = chunk_size
    _state["config"]["chunk_overlap"] = chunk_overlap
    _apply_settings(_state["config"])

    store = _get_store()
    all_docs = []
    failed = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            dest = os.path.join(tmpdir, upload.filename or f"file_{uuid.uuid4()}")
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            try:
                docs = _load_document(dest)
                # Preserve original filename in metadata
                for d in docs:
                    d.metadata["source"] = upload.filename or d.metadata.get("source", "unknown")
                all_docs.extend(docs)
            except Exception as e:
                failed.append({"file": upload.filename, "error": str(e)})

    if not all_docs:
        raise HTTPException(status_code=422, detail="No documents could be parsed.")

    chunks = recursive_chunk(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store.add_documents(chunks)

    _state["index_stats"]["docs"] += len(all_docs)
    _state["index_stats"]["chunks"] += len(chunks)

    # Rebuild graph so new index is used
    _state["graph"] = None

    return {
        "ok": True,
        "ingested_docs": len(all_docs),
        "total_chunks": len(chunks),
        "failed": failed,
        "index": _state["index_stats"],
    }


@app.post("/api/query")
def query(req: QueryRequest):
    """Run the 6-layer RAG pipeline and return the answer."""
    from pipeline.state import PipelineState

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        graph = _get_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline init failed: {e}")

    history = _state["conversation_history"][-20:] if req.use_history else []

    state = PipelineState(
        original_query=req.query,
        conversation_history=history,
    )

    try:
        result = graph.invoke(state.model_dump())
        result_state = PipelineState(**result)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": tb},
        )

    # Update conversation history
    _state["conversation_history"].append({"role": "user", "content": req.query})
    _state["conversation_history"].append({"role": "assistant", "content": result_state.answer})
    if len(_state["conversation_history"]) > 40:
        _state["conversation_history"] = _state["conversation_history"][-40:]

    # Build citation payload
    citations = [
        {
            "id": c["source_id"],
            "file": Path(c["source"]).name,
            "score": c.get("rerank_score", 0.0),
            "snippet": c.get("snippet", "")[:200],
        }
        for c in (result_state.citations or [])
    ]

    # Ragas scores (drop internal flags)
    ragas = {
        k: v
        for k, v in (result_state.ragas_scores or {}).items()
        if k != "abstained"
    }

    return {
        "answer": result_state.answer,
        "abstained": result_state.should_abstain,
        "citations": citations,
        "ragas": ragas,
        "meta": {
            "retry_count": result_state.retry_count,
            "sub_questions": result_state.sub_questions,
            "rewritten_query": result_state.rewritten_query,
            "needs_retrieval": result_state.needs_retrieval,
            "critique_passed": result_state.critique_passed,
        },
    }


@app.post("/api/clear")
def clear_history():
    """Clear conversation history."""
    _state["conversation_history"].clear()
    return {"ok": True}


@app.delete("/api/index")
def delete_index():
    """Wipe the vector store and reset index stats."""
    _state["store"] = None
    _state["graph"] = None
    _state["index_stats"] = {"docs": 0, "chunks": 0}

    import shutil
    from config import settings
    if os.path.exists(settings.chroma_persist_dir):
        shutil.rmtree(settings.chroma_persist_dir)

    return {"ok": True}


@app.get("/")
def root():
    return {"message": "Agentic RAG Pipeline API — see /docs for interactive spec."}
