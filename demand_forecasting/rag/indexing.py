# demand_forecasting/rag/indexing.py

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI

from .config import RAGConfig


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    id: str
    file_path: str
    object_name: str | None
    start_line: int
    end_line: int
    text: str


@dataclass
class RAGIndex:
    """In-memory index of code chunks and their embeddings."""

    config: RAGConfig
    chunks: List[CodeChunk]
    embeddings: np.ndarray  # shape: (n_chunks, dim)


def _iter_python_files(cfg: RAGConfig) -> List[Path]:
    """Return a list of Python files to index based on the config globs, recursively."""
    root = cfg.source_root
    paths: List[Path] = []
    for pattern in cfg.include_globs:
        # rglob searches all subdirectories under root
        paths.extend(root.rglob(pattern))
    # Deduplicate and sort for stability
    return sorted({p.resolve() for p in paths if p.is_file()})


def _split_file_into_chunks(path: Path, cfg: RAGConfig) -> List[CodeChunk]:
    """
    Very simple function/class-based splitting:

    - Starts a new chunk whenever a line begins with 'def ' or 'class ' (ignoring leading spaces).
    - Each chunk is then optionally re-split if it's too long in characters.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    boundaries: List[int] = []
    names: List[str | None] = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            boundaries.append(i)
            # Roughly extract the name after 'def ' or 'class '
            try:
                token = stripped.split()[1]
                name = token.split("(")[0].split(":")[0]
            except Exception:
                name = None
            names.append(name)

    # If no defs/classes found, treat whole file as one chunk
    if not boundaries:
        chunk = CodeChunk(
            id=f"{path}:{1}-{len(lines)}",
            file_path=str(path),
            object_name=None,
            start_line=1,
            end_line=len(lines),
            text=text,
        )
        return _maybe_subsplit_chunk(chunk, cfg)

    # Add sentinel at end
    boundaries.append(len(lines))
    names.append(None)

    chunks: List[CodeChunk] = []

    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        obj_name = names[idx]

        chunk_text = "\n".join(lines[start:end])
        chunk = CodeChunk(
            id=f"{path}:{start+1}-{end}",
            file_path=str(path),
            object_name=obj_name,
            start_line=start + 1,
            end_line=end,
            text=chunk_text,
        )
        chunks.extend(_maybe_subsplit_chunk(chunk, cfg))

    return chunks


def _maybe_subsplit_chunk(chunk: CodeChunk, cfg: RAGConfig) -> List[CodeChunk]:
    """
    If a chunk is longer than chunk_size_chars, split it into overlapping
    character-based subchunks. This prevents huge functions from blowing up
    token counts.
    """
    text = chunk.text
    max_len = cfg.chunk_size_chars
    overlap = cfg.chunk_overlap_chars

    if len(text) <= max_len:
        return [chunk]

    subchunks: List[CodeChunk] = []
    idx = 0
    part = 1

    while idx < len(text):
        sub_text = text[idx : idx + max_len]
        new_id = f"{chunk.id}#part{part}"
        subchunks.append(
            CodeChunk(
                id=new_id,
                file_path=chunk.file_path,
                object_name=chunk.object_name,
                start_line=chunk.start_line,  # we don't track exact lines for subparts
                end_line=chunk.end_line,
                text=sub_text,
            )
        )
        idx += max_len - overlap
        part += 1

    return subchunks


def _embed_texts(texts: List[str], cfg: RAGConfig) -> np.ndarray:
    """
    Embed a list of texts using OpenAI embeddings.

    All embedding logic is centralized here so it's easy to swap providers/models.
    """
    client = OpenAI()

    # OpenAI API accepts up to N inputs per call; for simplicity, we batch naively.
    batch_size = 64
    all_embeds: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=cfg.embedding_model,
            input=batch,
        )
        # resp.data[i].embedding is a list[float]
        embeds = np.array([d.embedding for d in resp.data], dtype="float32")
        all_embeds.append(embeds)

    return np.vstack(all_embeds)


def build_index(cfg: RAGConfig) -> RAGIndex:
    """
    Build a fresh index from the configured source_root and save it to disk.

    Usage:
        from demand_forecasting.rag.config import RAGConfig
        from demand_forecasting.rag.indexing import build_index

        cfg = RAGConfig()
        index = build_index(cfg)
    """
    cfg.print("Scanning Python files for indexing...")
    paths = _iter_python_files(cfg)
    if not paths:
        raise RuntimeError(
            "No Python files found for RAG indexing. Check include_globs."
        )

    cfg.print(f"Found {len(paths)} file(s).")

    all_chunks: List[CodeChunk] = []
    for path in paths:
        cfg.print(f"  -> Splitting {path}")
        chunks = _split_file_into_chunks(path, cfg)
        cfg.print(f"     {len(chunks)} chunk(s) from this file.")
        all_chunks.extend(chunks)

    cfg.print(f"Total chunks: {len(all_chunks)}")
    texts = [c.text for c in all_chunks]

    cfg.print("Embedding chunks...")
    embeddings = _embed_texts(texts, cfg)
    cfg.print(f"Embeddings shape: {embeddings.shape}")

    index = RAGIndex(config=cfg, chunks=all_chunks, embeddings=embeddings)

    cfg.print(f"Saving index to {cfg.index_path} ...")
    with cfg.index_path.open("wb") as f:
        pickle.dump(index, f)

    cfg.print("Index build complete.")
    return index


def load_index(cfg: RAGConfig) -> RAGIndex:
    """Load an existing index from disk."""
    if not cfg.index_path.exists():
        raise FileNotFoundError(
            f"Index file {cfg.index_path} not found. Run build_index(cfg) first."
        )
    with cfg.index_path.open("rb") as f:
        index: RAGIndex = pickle.load(f)
    return index


def search(
    index: RAGIndex, query_embedding: np.ndarray, top_k: int | None = None
) -> List[Tuple[CodeChunk, float]]:
    """
    Simple cosine similarity search over the in-memory embeddings.

    Returns a list of (CodeChunk, score) sorted by descending score.
    """
    embeds = index.embeddings  # (n, dim)
    q = query_embedding.astype("float32")
    # cosine similarity = (q · v) / (||q|| * ||v||)
    v_norms = np.linalg.norm(embeds, axis=1)
    q_norm = np.linalg.norm(q)
    # Avoid division by zero
    denom = (v_norms * q_norm) + 1e-8
    sims = embeds @ q / denom

    k = top_k or index.config.top_k
    # argsort descending
    top_idx = np.argsort(-sims)[:k]

    results: List[Tuple[CodeChunk, float]] = []
    for i in top_idx:
        results.append((index.chunks[i], float(sims[i])))
    return results
