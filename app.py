"""Streamlit chat UI for the RAG assistant (Phase 1: core RAG).

Phase 1 wires retrieve -> generate directly. Phase 2 swaps in the LangGraph
agentic pipeline (graph.py) without changing this UI's shape.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from ingest import (
    format_context,
    get_chat_model,
    get_loaded_sources,
    get_retriever,
    has_documents,
)
from prompts import GENERATE_PROMPT

load_dotenv()

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="📚")
st.title("📚 Agentic RAG Assistant")


def _sources_from_docs(documents) -> list[str]:
    seen = []
    for d in documents:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_str = f" p.{page + 1}" if isinstance(page, int) else ""
        label = f"{src}{page_str}"
        if label not in seen:
            seen.append(label)
    return seen


def answer_question(question: str) -> dict:
    """Phase 1 simple path: retrieve top-k chunks, then generate a grounded answer."""
    retriever = get_retriever(k=4)
    documents = retriever.invoke(question)
    chain = GENERATE_PROMPT | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": format_context(documents), "question": question}
    )
    return {"answer": answer, "sources": _sources_from_docs(documents)}


# --- Sidebar: environment + loaded documents -------------------------------
with st.sidebar:
    st.header("Status")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY detected")
    else:
        st.error("OPENAI_API_KEY missing — set it in .env")

    if has_documents():
        st.caption("Source documents are loaded on first question.")
    else:
        st.warning("No PDFs in ./docs — add some to enable answers.")

# --- Chat history -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# --- Handle input -----------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not os.getenv("OPENAI_API_KEY"):
            answer = "I need an `OPENAI_API_KEY` in your `.env` to answer."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        elif not has_documents():
            answer = "No documents found. Add PDFs to the `docs/` folder first."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    result = answer_question(prompt)
                except Exception as e:  # noqa: BLE001 - surface errors in UI
                    result = {"answer": f"Error: {e}", "sources": []}
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"- {s}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
