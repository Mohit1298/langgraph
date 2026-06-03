# LangGraph + Pydantic: Simple Conditional Workflow

A minimal [LangGraph](https://langchain-ai.github.io/langgraph/) workflow that uses a
**Pydantic** model as its state and demonstrates a **conditional edge**.

## What it does

```
START → ingest → (conditional) → handle_even → END
                              └─→ handle_odd  → END
```

1. `ingest` receives a number and records the step.
2. A conditional edge (`route_even_or_odd`) inspects the state and routes to
   either `handle_even` or `handle_odd`.
3. The chosen branch updates the shared Pydantic `WorkflowState`.

No API keys are required — the logic is pure Python.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python workflow.py
```

Expected output:

```
input=4
  category: even
  message:  4 is even, so half of it is 2.
  steps:    ['ingested 4', 'handled as even']

input=7
  category: odd
  message:  7 is odd, so triple it plus one is 22.
  steps:    ['ingested 7', 'handled as odd']
```
