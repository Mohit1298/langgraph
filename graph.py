"""LangGraph agentic (corrective) RAG state machine.

Flow:

    START -> retrieve -> grade_documents -> [decide_to_generate]
             generate -> END
             transform_query -> retrieve        (bounded rewrite loop)
             web_search -> generate              (only if TAVILY_API_KEY set)

The `retries` counter guarantees termination: the rewrite loop can run at most
MAX_RETRIES times before the graph is forced to either web-search (if enabled)
or generate an honest "not in the documents" answer. This is a *bounded*
agentic loop -- there is no possibility of infinite recursion.
"""

from __future__ import annotations

import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ingest import format_context, get_chat_model, get_retriever
from prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT

load_dotenv()

MAX_RETRIES = 2
# Computed at build time (see build_graph) so the .env is loaded first.
WEB_SEARCH_ENABLED = False


class GraphState(TypedDict):
    """Shared state threaded through the graph."""

    question: str          # current (possibly rewritten) query
    original_question: str  # the user's original query
    documents: List         # retrieved + filtered docs
    generation: str         # final answer
    retries: int            # rewrite attempts so far
    web_search: str         # "Yes" / "No"
    steps: List[str]        # human-readable trace for the UI


class GradeDocuments(BaseModel):
    """Structured binary relevance score for a single chunk."""

    binary_score: str = Field(
        description="Is the document relevant to the question? 'yes' or 'no'."
    )


# --- Nodes ------------------------------------------------------------------
def retrieve(state: GraphState) -> dict:
    """Fetch chunks for the current question."""
    docs = get_retriever(k=4).invoke(state["question"])
    steps = state.get("steps", []) + [f"retrieve: {len(docs)} chunks"]
    return {"documents": docs, "steps": steps}


def grade_documents(state: GraphState) -> dict:
    """Keep only chunks the LLM judges relevant; flag web search if none remain."""
    grader = GRADE_PROMPT | get_chat_model().with_structured_output(GradeDocuments)

    relevant = []
    for d in state["documents"]:
        try:
            score = grader.invoke(
                {"document": d.page_content, "question": state["question"]}
            )
            if score.binary_score.strip().lower() == "yes":
                relevant.append(d)
        except Exception:  # noqa: BLE001 - a grading failure shouldn't crash the graph
            relevant.append(d)  # fail open: keep the doc rather than lose it

    web_search = "No" if relevant else "Yes"
    steps = state.get("steps", []) + [
        f"grade: {len(relevant)}/{len(state['documents'])} relevant"
    ]
    return {"documents": relevant, "web_search": web_search, "steps": steps}


def transform_query(state: GraphState) -> dict:
    """Rewrite the query for better retrieval; increment the retry counter."""
    rewriter = REWRITE_PROMPT | get_chat_model() | StrOutputParser()
    better = rewriter.invoke({"question": state["question"]}).strip()
    steps = state.get("steps", []) + [f"rewrite -> {better!r}"]
    return {
        "question": better,
        "retries": state.get("retries", 0) + 1,
        "steps": steps,
    }


def web_search(state: GraphState) -> dict:
    """Tavily web-search fallback. Only reached when WEB_SEARCH_ENABLED."""
    from tavily import TavilyClient

    docs = list(state.get("documents", []))
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp = client.search(state["original_question"], max_results=4)
        for r in resp.get("results", []):
            docs.append(
                Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url", "web"), "page": None},
                )
            )
        note = f"web_search: +{len(resp.get('results', []))} results"
    except Exception as e:  # noqa: BLE001 - degrade gracefully, never crash
        note = f"web_search failed ({e}); continuing without web results"

    return {"documents": docs, "steps": state.get("steps", []) + [note]}


def generate(state: GraphState) -> dict:
    """Produce the final grounded answer (answers honestly if no context)."""
    docs = state.get("documents", [])
    context = format_context(docs) if docs else "(no relevant context found)"
    chain = GENERATE_PROMPT | get_chat_model() | StrOutputParser()
    gen = chain.invoke(
        {"context": context, "question": state["original_question"]}
    )
    return {"generation": gen, "steps": state.get("steps", []) + ["generate"]}


# --- Conditional edge -------------------------------------------------------
def decide_to_generate(
    state: GraphState,
) -> Literal["generate", "transform_query", "web_search"]:
    """Route after grading.

    - Relevant docs found -> generate.
    - Otherwise, if we still have rewrite budget -> transform_query (loop back).
    - Budget exhausted -> web_search if enabled, else generate honestly.
    """
    if state.get("web_search", "No") == "No":
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "transform_query"
    return "web_search" if WEB_SEARCH_ENABLED else "generate"


# --- Build ------------------------------------------------------------------
def build_graph():
    """Wire the nodes/edges and compile the graph.

    The web_search node is only added when TAVILY_API_KEY is present; otherwise
    the conditional edge routes exhausted retries straight to generate.
    """
    global WEB_SEARCH_ENABLED
    WEB_SEARCH_ENABLED = bool(os.getenv("TAVILY_API_KEY"))

    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("generate", generate)
    if WEB_SEARCH_ENABLED:
        g.add_node("web_search", web_search)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")

    route_map = {"generate": "generate", "transform_query": "transform_query"}
    if WEB_SEARCH_ENABLED:
        route_map["web_search"] = "web_search"
    g.add_conditional_edges("grade_documents", decide_to_generate, route_map)

    g.add_edge("transform_query", "retrieve")
    if WEB_SEARCH_ENABLED:
        g.add_edge("web_search", "generate")
    g.add_edge("generate", END)

    return g.compile()


# Module-level compiled-app singleton.
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(question: str) -> dict:
    """Convenience wrapper: run the graph and return answer + sources + trace."""
    result = get_app().invoke(
        {
            "question": question,
            "original_question": question,
            "documents": [],
            "generation": "",
            "retries": 0,
            "web_search": "No",
            "steps": [],
        }
    )

    sources = []
    for d in result.get("documents", []):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        page_str = f" p.{page + 1}" if isinstance(page, int) else ""
        label = f"{src}{page_str}"
        if label not in sources:
            sources.append(label)

    return {
        "answer": result.get("generation", ""),
        "sources": sources,
        "steps": result.get("steps", []),
        "web_search_enabled": WEB_SEARCH_ENABLED,
    }
