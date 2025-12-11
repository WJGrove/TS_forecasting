# demand_forecasting/rag/chat_cli.py

from __future__ import annotations

import textwrap

import numpy as np
from openai import OpenAI

from .config import RAGConfig
from .indexing import build_index, load_index, search, _embed_texts, RAGIndex, CodeChunk


def _ensure_index(cfg: RAGConfig) -> RAGIndex:
    """Load the index if it exists; otherwise build it."""
    try:
        cfg.print(f"Trying to load index from {cfg.index_path} ...")
        return load_index(cfg)
    except FileNotFoundError:
        cfg.print("Index not found; building a new one.")
        return build_index(cfg)


def _embed_query(query: str, cfg: RAGConfig) -> np.ndarray:
    """Embed a single query string."""
    embeds = _embed_texts([query], cfg)
    return embeds[0]


def _build_context_block(chunks: list[CodeChunk]) -> str:
    """
    Build a context string that the LLM will see, including filenames
    and object names so it can cite them in answers.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.file_path}"
        if chunk.object_name:
            header += f" :: {chunk.object_name}"
        header += f" (lines {chunk.start_line}-{chunk.end_line})"
        parts.append(header)
        parts.append(chunk.text)
        parts.append("\n" + "-" * 80 + "\n")
    return "\n".join(parts)


def _chat_with_context(cfg: RAGConfig, question: str, context: str) -> str:
    """
    Call the chat model with the given context + question.

    All chat-model-specific logic lives here so it's easy to change models/providers.
    """
    client = OpenAI()

    system_msg = textwrap.dedent(
        """
        You are an assistant helping a developer understand and refactor a time series
        forecasting codebase. You are given context chunks from the repository
        (Python modules for preprocessing, diagnostics, plotting, and forecasting).

        Rules:
        - Only use information from the provided context when describing code.
        - Prefer to cite which file and function/class you're referring to.
        - If the answer is not clearly supported by the context, say you are unsure.
        - Be concise but precise. The user is experienced in mathematical and statistical
        concepts but is less experienced with development.
        """
    ).strip()

    user_msg = textwrap.dedent(
        f"""
        Here is context from the repository:

        {context}

        ---
        Question: {question}
        """
    ).strip()

    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,  # low for code Q&A
    )

    return resp.choices[0].message.content or ""


def main() -> None:
    """
    Simple CLI:

    - Ensures an index exists (build/build_index if needed).
    - Starts an interactive loop:
        > What does TSPreprocessor.run do?
        ...
    """
    cfg = RAGConfig()

    cfg.print("=== Repo RAG Assistant ===")
    index = _ensure_index(cfg)

    cfg.print(
        f"Index ready: {len(index.chunks)} chunks from {len(set(c.file_path for c in index.chunks))} files."
    )
    print()
    print("Type questions about your codebase (or 'exit' to quit).")
    print()

    while True:
        try:
            question = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Embed query
        q_embed = _embed_query(question, cfg)

        # Retrieve top-k chunks
        results = search(index, q_embed, top_k=cfg.top_k)
        cfg.print("Top matches:")
        for chunk, score in results:
            cfg.print(
                f"  {chunk.file_path} :: {chunk.object_name}  (score={score:.3f})"
            )

        # Build context block
        top_chunks = [c for (c, _) in results]
        context_str = _build_context_block(top_chunks)

        # Call chat model
        answer = _chat_with_context(cfg, question, context_str)

        print()
        print(textwrap.fill(answer, width=100))
        print()


if __name__ == "__main__":
    main()
