"""Streamlit chat UI for the agentic (corrective) RAG assistant.

Calls the compiled LangGraph app (graph.run_query) which runs
retrieve -> grade -> (rewrite loop / web search) -> generate, and renders the
answer, its sources, and the agent's reasoning trace.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from graph import run_query
from ingest import get_loaded_sources, has_documents

load_dotenv()

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="📚")
st.title("📚 Agentic RAG Assistant")


def answer_question(question: str) -> dict:
    """Run the agentic RAG graph: retrieve -> grade -> (rewrite/web) -> generate."""
    return run_query(question)


# --- Sidebar: environment + loaded documents -------------------------------
with st.sidebar:
    st.header("Status")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY detected")
    else:
        st.error("OPENAI_API_KEY missing — set it in .env")

    if os.getenv("TAVILY_API_KEY"):
        st.success("Web-search fallback enabled (Tavily)")
    else:
        st.info("No TAVILY_API_KEY — web-search fallback disabled (graceful).")

    st.divider()
    st.subheader("Source documents")
    if not has_documents():
        st.warning("No PDFs in ./docs — add some to enable answers.")
    else:
        sources = get_loaded_sources()
        if sources:
            for s in sources:
                st.markdown(f"- {s}")
        else:
            st.caption("Documents are embedded on the first question.")


def _render_trace(steps: list[str]) -> None:
    if steps:
        with st.expander("🧭 Agent reasoning trace"):
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")


def _render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- {s}")


# --- Chat history -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))
        _render_trace(msg.get("steps", []))

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
            with st.spinner("Retrieving, grading, and generating..."):
                try:
                    result = answer_question(prompt)
                except Exception as e:  # noqa: BLE001 - surface errors in UI
                    result = {"answer": f"Error: {e}", "sources": [], "steps": []}
            st.markdown(result["answer"])
            _render_sources(result.get("sources", []))
            _render_trace(result.get("steps", []))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "steps": result.get("steps", []),
                }
            )
