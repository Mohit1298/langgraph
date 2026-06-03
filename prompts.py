"""All prompt templates in one place.

Three prompts drive the agentic RAG graph:
- GRADE_PROMPT    -> binary relevance grading (structured yes/no)
- REWRITE_PROMPT  -> query rewriting for better retrieval
- GENERATE_PROMPT -> grounded answer generation with citations
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a retrieved document chunk is "
            "relevant to a user question.\n"
            "If the chunk contains keywords or semantic meaning related to the "
            "question, grade it as relevant.\n"
            "This is a coarse filter to weed out clearly irrelevant chunks, not a "
            "strict test.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is "
            "relevant to the question.",
        ),
        (
            "human",
            "Retrieved document chunk:\n\n{document}\n\nUser question: {question}",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query-rewriting assistant. Rewrite the user's question to be "
            "clearer and more effective for semantic similarity search over a "
            "document vector store.\n"
            "Preserve the original intent. Expand abbreviations, add salient "
            "synonyms/keywords, and remove conversational filler.\n"
            "Return ONLY the rewritten question, with no preamble.",
        ),
        ("human", "Original question:\n\n{question}\n\nRewritten question:"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions using ONLY the provided "
            "context.\n"
            "Rules:\n"
            "1. Use only the information in the context. Do not use outside knowledge "
            "and do not fabricate.\n"
            "2. Cite the source for each claim using the format (source: <filename> "
            "p.<page>), taken from the context's source markers.\n"
            "3. If the context is insufficient to answer, say so honestly and clearly "
            "state that the documents do not cover it.\n"
            "Be concise and accurate.",
        ),
        (
            "human",
            "Context:\n\n{context}\n\nQuestion: {question}\n\nGrounded answer with "
            "citations:",
        ),
    ]
)
