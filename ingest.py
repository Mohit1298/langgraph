"""Document ingestion and retrieval.

Loads PDFs from ./docs, splits them into chunks, embeds them with OpenAI, and
builds an in-memory Chroma vector store. The vector store is built once and
cached in a module-level singleton so Streamlit reruns don't re-embed.

Also exposes `get_chat_model()` as the single place the chat LLM is configured,
so it can be swapped (e.g. for Claude) by changing one function.
"""

from __future__ import annotations

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
EMBED_MODEL = "text-embedding-3-small"

# Module-level singletons (built once per process).
_vectorstore: Chroma | None = None
_loaded_sources: list[str] = []


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Single source of truth for the chat LLM.

    Reads CHAT_MODEL from the environment (default: gpt-4o-mini). Swap the
    implementation here to move to Claude or another provider.
    """
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


def _load_documents() -> list:
    """Load and split every PDF in ./docs into chunks."""
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {DOCS_DIR}. Drop some .pdf files there and retry."
        )

    docs = []
    for path in pdf_paths:
        # PyPDFLoader splits per-page and records page numbers in metadata.
        docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Normalize the source metadata to just the filename for clean citations.
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk.metadata["source"] = os.path.basename(src) if src else "unknown"

    return chunks


def build_vectorstore(force: bool = False) -> Chroma:
    """Build (or return the cached) in-memory Chroma vector store."""
    global _vectorstore, _loaded_sources
    if _vectorstore is not None and not force:
        return _vectorstore

    chunks = _load_documents()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    _vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    _loaded_sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
    return _vectorstore


def get_retriever(k: int = 4):
    """Return a retriever over the (cached) vector store."""
    return build_vectorstore().as_retriever(search_kwargs={"k": k})


def get_loaded_sources() -> list[str]:
    """List of source filenames currently in the vector store (after build)."""
    return list(_loaded_sources)


def has_documents() -> bool:
    """True if there is at least one PDF available to ingest."""
    return bool(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))


def format_context(documents) -> str:
    """Render retrieved docs into a context string with source/page markers."""
    blocks = []
    for d in documents:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        # PyPDF pages are 0-indexed; present them 1-indexed for humans.
        page_str = f"{page + 1}" if isinstance(page, int) else "?"
        marker = f"(source: {source} p.{page_str})"
        blocks.append(f"{marker}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)
