"""A simple LangGraph workflow that uses a Pydantic model as its state.

The graph takes a number, then a conditional edge routes it to a different
node depending on whether the number is even or odd. Each node updates the
shared Pydantic state.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    """Shared state passed between nodes.

    Using a Pydantic model gives us validation and clear typing for free.
    """

    number: int
    category: str = ""
    message: str = ""
    steps: list[str] = Field(default_factory=list)


def ingest(state: WorkflowState) -> dict:
    """Entry node: record that we received the number."""
    return {"steps": state.steps + [f"ingested {state.number}"]}


def route_even_or_odd(state: WorkflowState) -> Literal["handle_even", "handle_odd"]:
    """Conditional edge: decide which branch to take."""
    return "handle_even" if state.number % 2 == 0 else "handle_odd"


def handle_even(state: WorkflowState) -> dict:
    return {
        "category": "even",
        "message": f"{state.number} is even, so half of it is {state.number // 2}.",
        "steps": state.steps + ["handled as even"],
    }


def handle_odd(state: WorkflowState) -> dict:
    return {
        "category": "odd",
        "message": f"{state.number} is odd, so triple it plus one is {state.number * 3 + 1}.",
        "steps": state.steps + ["handled as odd"],
    }


def build_graph():
    """Wire up the nodes, the conditional edge, and compile the graph."""
    graph = StateGraph(WorkflowState)

    graph.add_node("ingest", ingest)
    graph.add_node("handle_even", handle_even)
    graph.add_node("handle_odd", handle_odd)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges(
        "ingest",
        route_even_or_odd,
        {"handle_even": "handle_even", "handle_odd": "handle_odd"},
    )
    graph.add_edge("handle_even", END)
    graph.add_edge("handle_odd", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    for n in (4, 7):
        result = app.invoke(WorkflowState(number=n))
        # invoke returns a dict-like state; rebuild a model for nice access.
        final = WorkflowState(**result)
        print(f"input={n}")
        print(f"  category: {final.category}")
        print(f"  message:  {final.message}")
        print(f"  steps:    {final.steps}")
        print()
