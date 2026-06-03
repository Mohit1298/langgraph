# Agentic (Corrective) RAG Assistant

A self-correcting RAG chatbot over your PDFs, built with **LangGraph**. See the
full architecture and talking points at the bottom — this header is expanded in
Phase 3.

## Phase 1 — Core RAG (current)

- `ingest.py` — load PDFs from `docs/`, chunk, embed (`text-embedding-3-small`),
  build an in-memory Chroma vector store, expose `get_retriever(k=4)`.
- `prompts.py` — grading / rewrite / generation prompts.
- `app.py` — Streamlit chat UI with grounded answers + source citations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your OPENAI_API_KEY
```

Drop a few PDFs into `docs/`, then run:

```bash
streamlit run app.py
```

Ask a question answerable from your PDFs and you'll get a grounded answer with
`(source: <file> p.<n>)` citations.
