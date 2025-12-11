# RAG & Aider Interview Cheat Sheet

This is a high-level reference you can skim before or during interviews. Think of it as raw material
you can turn into 60–90 second stories as needed.

---

## 1. RAG Assistant over My Forecasting Repo

### Goal / Use Case

- Built a small **Retrieval-Augmented Generation (RAG)** assistant over my personal forecasting
  codebase.
- Main purpose: help me **understand, refactor, and document** a panel time-series forecasting
  pipeline (preprocessing, diagnostics, plotting, forecasting).
- Example questions I ask it:
  - “What does `TSPreprocessor.run` do end-to-end?”
  - “Which helpers does it use and where are they defined?”
  - “How are short series treated differently from long series?”

### Architecture & Tech Stack

- **Language:** Python.
- **Indexing:**
  - Walk selected folders in the repo (e.g. time-series pipeline modules and helper utilities).
  - Parse Python files into **object-level chunks** (functions, classes, methods), not just fixed-size
    text chunks.
  - For each chunk I store:
    - `file_path`
    - `object_name` (e.g. `TSPreprocessor.run`)
    - start / end line numbers
    - the source text itself
  - Embed these chunks with an OpenAI **embedding model** (e.g. `text-embedding-3-small`).
  - Store embeddings + metadata in a local `.pkl` index (numpy arrays + Python objects).
- **Retrieval:**
  - Embed the user question and compute **cosine similarity** to all chunk embeddings.
  - Take top-k chunks (tuned from 3 to 8 for better recall on multi-method questions).
  - Build a compact context block like:
    - `[1] ts0_preprocessing.py :: TSPreprocessor.run (lines …)`
    - `[2] ts_preprocessing_utils.py :: interpolate_groupwise_numeric …`
- **Generation:**
  - Call an OpenAI chat model (a cost-effective “mini” model) with:
    - A focused **system prompt**:
      - Only use provided context when describing code.
      - Cite file and function/class names when possible.
      - Explain math/stats when relevant; assume the user knows stats but is less experienced with
        software engineering.
    - A **user message** containing the context block plus the question.
    - A short **conversation history**, so follow-up questions like “And how is that decision made?”
      stay grounded.

### Design Decisions & Trade-offs

- **Local numpy index** instead of Pinecone/FAISS:
  - Repo is small, so an in-memory index is simple and fast enough.
  - Retrieval and generation are separated cleanly, so I can swap in a vector DB later without
    rewriting the RAG logic.
- **Object-level chunking:**
  - Embedding whole functions/classes preserves semantics and lets the model answer questions like
    “What helpers does this method call?” more reliably.
  - Trade-off: fewer, larger chunks; I mitigate that with a small `top_k` and structured headers.
- **Config-driven models:**
  - Both the embedding and chat models are configured via a `RAGConfig` object.
  - Makes it easy to start with cheaper models, and later upgrade to more capable models without
    changing the main code.

### How I’d Explain It as an AI Engineer

- It’s a minimal but real **RAG pipeline**:
  - Document store = my repo.
  - Indexing = chunk + embed + store.
  - Retrieval = vector search over embeddings + metadata.
  - Generation = LLM with a domain-specific system prompt and history.
- Shows I understand:
  - The separation of **retrieval vs. generation**.
  - How to structure context (chunks, metadata, `top_k`) around a concrete use case.
  - How to parameterize models/costs and keep the system easily swappable.

---

## 2. Aider as an “Agentic” Coding Assistant

### Goal / Use Case

- Use **Aider** as a tool-using coding assistant to:
  - Incrementally refactor legacy forecasting code.
  - Enforce consistency (type hints, docstrings, naming conventions).
  - Help spin up tests and small restructurings faster than by hand.
- It lives in my normal dev loop alongside:
  - Git (version control),
  - VSCode (editor),
  - and my RAG bot (for code understanding).

### Setup & Integration

- Installed Aider into the same virtual environment as my repo.
- Run it from the **repo root** so it can:
  - See the git repository.
  - Propose changes as diffs instead of big copy/paste patches.
- Configure it with:
  - A reasonably capable, cost-effective OpenAI model as the **coding model**.
  - Optionally, a higher-capacity model for more “architect”-style refactors (if needed).

### How I Use It in Practice

Typical session pattern:

1. Start Aider in the repo root.
2. Give it a **scoped request**, for example:
   - “Add type hints and detailed docstrings to `TSPreprocessor.run` and confirm the return type.”
   - “Refactor `train_test_split_panel` into a utility module and update all imports.”
   - “Generate a first-pass test file for `compute_wape` with some edge cases.”
3. Aider:
   - Reads the relevant files.
   - Proposes diffs in git-friendly form.
4. I:
   - Review diffs,
   - Run tests or quick sanity checks,
   - Commit if I’m happy.

### Why This Counts as “Agentic”

- Aider doesn’t just answer questions—it **acts** on the codebase:
  - Reads files,
  - Edits them,
  - Interacts with git.
- In AI-engineering terms, it’s a **tool-using agent**:
  - An LLM controlling tools like “edit this file,” “show me the diff,” etc., to achieve developer
    goals.
- Conceptually, this is a first step toward:
  - Auto-refactoring agents,
  - Internal AI code reviewers,
  - Or agents that help maintain and evolve MLOps / forecasting pipelines.

### How I’d Tie It Back to the Job Description

- Integrating LLMs into **real workflows**, not just notebooks.
- Combining RAG (code understanding) with agentic editing (Aider) to:
  - Understand the existing system,
  - Safely evolve it,
  - And keep everything under version control.
