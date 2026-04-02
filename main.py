"""Main entry point — document ingestion + interactive RAG query loop."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def load_documents(data_dir: str) -> list:
    """Load documents from a directory (PDF, TXT, DOCX, MD)."""
    from langchain_core.documents import Document

    docs: list[Document] = []
    patterns = {
        "**/*.pdf": _load_pdf,
        "**/*.txt": _load_text,
        "**/*.md": _load_text,
        "**/*.docx": _load_docx,
    }

    for pattern, loader_fn in patterns.items():
        for filepath in glob.glob(os.path.join(data_dir, pattern), recursive=True):
            try:
                loaded = loader_fn(filepath)
                docs.extend(loaded)
                console.print(f"  [green]✓[/green] {filepath} ({len(loaded)} pages)")
            except Exception as e:
                console.print(f"  [red]✗[/red] {filepath}: {e}")

    return docs


def _load_pdf(path: str) -> list:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(path).load()


def _load_text(path: str) -> list:
    from langchain_core.documents import Document

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": path})]


def _load_docx(path: str) -> list:
    from langchain_core.documents import Document
    import docx2txt

    content = docx2txt.process(path)
    return [Document(page_content=content, metadata={"source": path})]


def ingest(data_dir: str, store) -> None:
    """Load, chunk, and index documents."""
    from indexing.chunker import recursive_chunk

    console.print(Panel("📄 Loading documents...", style="bold blue"))
    documents = load_documents(data_dir)

    if not documents:
        console.print("[yellow]No documents found. Add files to the data/ directory.[/yellow]")
        return

    console.print(f"\n[bold]Loaded {len(documents)} document(s). Chunking...[/bold]")
    chunks = recursive_chunk(documents)
    console.print(f"[bold]Created {len(chunks)} chunks. Indexing...[/bold]")

    store.add_documents(chunks)
    console.print(Panel(f"✅ Indexed {len(chunks)} chunks into vector store", style="bold green"))


def display_result(state) -> None:
    """Pretty-print the pipeline result."""
    console.print()

    if state.should_abstain:
        console.print(Panel(
            state.answer,
            title="🚫 Abstained",
            style="bold red",
        ))
    else:
        console.print(Panel(
            Markdown(state.answer.split("\n--- Sources ---")[0]),
            title="💡 Answer",
            style="bold green",
        ))

    # Citations table
    if state.citations and not state.should_abstain:
        table = Table(title="📚 Sources")
        table.add_column("#", style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Snippet", max_width=60)

        for cite in state.citations:
            table.add_row(
                str(cite["source_id"]),
                Path(cite["source"]).name,
                f"{cite.get('rerank_score', 0):.3f}",
                cite["snippet"][:60] + "...",
            )
        console.print(table)

    # Ragas scores
    if state.ragas_scores:
        table = Table(title="📊 Quality Metrics (Ragas)")
        table.add_column("Metric", style="bold")
        table.add_column("Score", style="green")

        for metric, score in state.ragas_scores.items():
            if metric == "abstained":
                continue
            color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
            table.add_row(metric, f"[{color}]{score:.3f}[/{color}]")
        console.print(table)

    # Pipeline metadata
    if state.retry_count > 0:
        console.print(f"[dim]Retries: {state.retry_count}[/dim]")
    if state.sub_questions:
        console.print(f"[dim]Sub-questions: {', '.join(state.sub_questions)}[/dim]")


def interactive_loop(graph, enable_ragas: bool) -> None:
    """Run interactive query loop."""
    from pipeline.state import PipelineState

    conversation_history: list[dict[str, str]] = []

    console.print(Panel(
        "🤖 RAG Pipeline Ready!\n"
        "Type your question, or:\n"
        "  /quit  — exit\n"
        "  /clear — clear conversation history\n"
        "  /providers — list available model providers",
        style="bold cyan",
    ))

    while True:
        try:
            query = console.input("\n[bold cyan]❓ Query:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/quit":
            break
        if query == "/clear":
            conversation_history.clear()
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue
        if query == "/providers":
            from mcp.registry import provider_registry
            console.print(f"Available: {', '.join(provider_registry.available_providers)}")
            continue

        # Build initial state
        state = PipelineState(
            original_query=query,
            conversation_history=conversation_history.copy(),
        )

        console.print("[dim]Processing...[/dim]")

        try:
            # Run the graph
            result = graph.invoke(state.model_dump())

            # Convert result dict back to PipelineState for display
            result_state = PipelineState(**result)
            display_result(result_state)

            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": result_state.answer})

            # Keep history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="6-Layer Agentic RAG Pipeline")
    parser.add_argument(
        "--data-dir", "-d",
        default="./data",
        help="Directory containing documents to index (default: ./data)",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Disable Ragas post-evaluation",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "ollama"],
        help="Override default LLM provider",
    )
    args = parser.parse_args()

    # Override provider if specified
    if args.provider:
        from config import settings
        settings.default_llm_provider = args.provider
        settings.default_embedding_provider = args.provider

    console.print(Panel(
        "[bold]🏗️ 6-Layer Agentic RAG Pipeline[/bold]\n"
        "HyDE → Lazy Router → Hybrid Retrieval → Distillation → Critique → Output",
        style="bold magenta",
    ))

    # Initialize store
    from indexing.store import DocumentStore

    store = DocumentStore()

    # Ingest documents
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    if args.reindex or not os.path.exists(store.persist_dir):
        ingest(data_dir, store)
    else:
        console.print("[dim]Using existing index. Pass --reindex to rebuild.[/dim]")
        # Rebuild BM25 from ChromaDB
        try:
            existing = store.vectorstore.get()
            if existing and existing.get("documents"):
                from langchain_core.documents import Document

                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(
                        existing["documents"], existing["metadatas"]
                    )
                ]
                store._build_bm25(docs)
                console.print(f"[dim]Loaded {len(docs)} chunks from existing index.[/dim]")
        except Exception:
            console.print("[yellow]Could not load existing index. Run with --reindex.[/yellow]")

    # Build graph
    from pipeline.graph import build_rag_graph

    enable_ragas = not args.no_ragas
    graph = build_rag_graph(store, enable_ragas=enable_ragas)

    console.print(f"[dim]Provider: {args.provider or 'default from .env'}[/dim]")
    console.print(f"[dim]Ragas eval: {'enabled' if enable_ragas else 'disabled'}[/dim]")

    # Run interactive loop
    interactive_loop(graph, enable_ragas)
    console.print("\n[bold]Goodbye! 👋[/bold]")


if __name__ == "__main__":
    cli()
