# Agentic (Corrective) RAG Assistant

A self-correcting RAG chatbot over your PDFs, built with **LangGraph**.

Unlike naive *retrieve-then-generate* RAG, this system uses a **LangGraph state
machine** that grades its own retrieved context, **rewrites the query and retries**
when retrieval is weak (bounded), optionally **falls back to web search**, and
finally generates a grounded answer **with source citations**. This is the
Corrective RAG (CRAG) / Agentic RAG pattern.

## Architecture

```
                ┌───────────┐
   START ─────► │ retrieve  │ ◄───────────────┐
                └─────┬─────┘                 │
                      ▼                        │ (rewritten query)
              ┌────────────────┐               │
              │ grade_documents│               │
              └───────┬────────┘               │
                      ▼                         │
            «decide_to_generate»                │
        ┌─────────────┼──────────────┐         │
        │ relevant    │ no relevant  │ no relevant
        │ docs        │ docs &       │ docs &
        ▼             ▼ retries<2    ▼ retries exhausted
   ┌──────────┐  ┌───────────────┐  ┌────────────┐
   │ generate │  │transform_query│  │ web_search │ (only if TAVILY_API_KEY)
   └────┬─────┘  └───────┬───────┘  └─────┬──────┘
        │                └──────────────► (loop back to retrieve)
        ▼                                  │
       END  ◄───────────────────────────── generate
```

When `TAVILY_API_KEY` is **absent**, the `web_search` node is not added to the
graph and exhausted retries route straight to `generate`, which answers honestly
that the documents don't cover the question. The app never crashes on a missing
key.

### Nodes (`graph.py`)

| Node | Responsibility |
|------|----------------|
| `retrieve` | Fetch top-`k` chunks from the Chroma vector store for the current (possibly rewritten) query. |
| `grade_documents` | Ask the LLM for a **structured yes/no** relevance verdict per chunk; keep only relevant ones. If none survive, flag `web_search = "Yes"`. |
| `transform_query` | Rewrite the query to be clearer/more retrievable and increment the `retries` counter. |
| `web_search` | (Optional) Tavily search; append results as documents. Degrades gracefully on error. |
| `generate` | Produce the final grounded answer with `(source: <file> p.<n>)` citations, or answer honestly when there's no relevant context. |

### Conditional edge: `decide_to_generate`

- **Relevant docs found** → `generate`.
- **No relevant docs, retries < 2** → `transform_query` → back to `retrieve` (bounded loop).
- **Retries exhausted** → `web_search` if enabled, otherwise `generate` (honest "not in the docs" answer).

### State (`GraphState`)

`question`, `original_question`, `documents`, `generation`, `retries`,
`web_search`, and `steps` (a human-readable trace rendered in the UI).

## How it mitigates hallucination

- **Relevance grading gate.** Every retrieved chunk is independently graded by
  the LLM before it can influence the answer. Off-topic chunks are dropped, so
  the generator isn't tempted to stitch together irrelevant context.
- **Grounded-only generation.** The generation prompt forbids outside knowledge,
  requires per-claim citations (`source` + page), and instructs the model to say
  "the documents don't cover this" rather than guess.
- **Self-correction instead of confident nonsense.** Weak retrieval triggers a
  query rewrite (and optional web search) rather than forcing an answer from poor
  context.
- **Bounded agentic loop.** The `retries` counter caps rewrites at 2, so the
  graph **always terminates** — no infinite recursion — and degrades to an honest
  non-answer when it genuinely can't ground a response.

## Project structure

```
.
├── docs/             # drop your source PDFs here
├── ingest.py         # load → chunk → embed → in-memory Chroma + get_retriever()
├── prompts.py        # GRADE / REWRITE / GENERATE prompt templates
├── graph.py          # LangGraph: state, nodes, conditional edge, compiled app
├── app.py            # Streamlit chat UI (answer + sources + reasoning trace)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY (TAVILY_API_KEY optional)
```

Drop a few PDFs into `docs/`, then launch the UI:

```bash
streamlit run app.py
```

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | yes | Generation, grading, rewriting, embeddings. |
| `TAVILY_API_KEY` | no | Enables the web-search fallback node. |
| `CHAT_MODEL` | no | Override the chat model (default `gpt-4o-mini`). Swap in one place — `get_chat_model()` in `ingest.py` — to move to Claude. |

Models: `gpt-4o-mini` for chat/grading, `text-embedding-3-small` for embeddings
(cheap + fast).

## Demo script

1. **In-domain question** → grounded answer with correct citations; the trace
   shows `retrieve → grade → generate`.
2. **Out-of-domain question** → grading drops all chunks → `transform_query`
   rewrite (and/or `web_search`) → graceful, honest answer. The self-correction
   is visible in the **🧭 Agent reasoning trace** expander.
