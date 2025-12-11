# demand_forecasting/rag/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class RAGConfig:
    """
    Configuration for the repo RAG assistant.

    This is intentionally simple and keeps model names + retrieval settings
    in one place so it's easy to swap models/providers later.
    """

    # Root folder to scan for code (relative to repo root)
    source_root: Path = Path("demand_forecasting")

    # Which files to include (relative to source_root)
    include_globs: List[str] = field(
        default_factory=lambda: [
            "ts0_*.py",
            "ts1_*.py",
            "ts2_*.py",
            "ts3_*.py",
            "ts_preprocessing_utils.py",
            "ts_forecast_utils.py",
        ]
    )

    # Where to persist the built index
    index_path: Path = Path(".rag_index.pkl")  # Hidden file in repo root

    # OpenAI models (embedding vs chat are decoupled)
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Chunking config
    chunk_size_chars: int = 2000
    chunk_overlap_chars: int = 200

    # Retrieval config
    top_k: int = 8

    # conversation history config
    max_history_turns: int = 6

    # Debug / verbosity
    verbose: bool = True

    def print(self, msg: str) -> None:
        """Helper to centralize verbose printing."""
        if self.verbose:
            print(msg)
