# RAG & Aider – Interview Cheat Sheet (Short)

## 1. RAG Assistant over My Forecasting Repo

### Goal / Use Case
- Built a small **RAG assistant** over my personal **panel time-series forecasting** repo.
- Use it to **understand, refactor, and document** the pipeline: preprocessing, diagnostics, plotting, forecasting.
- Typical questions:
  - “What does `TSPreprocessor.run` do end-to-end?”
  - “Which helpers does it use and where are they defined?”

### Architecture & Stack
- **Language:** Python.
- **Indexing:**
  - Walks selected folders (e.g. `time_series_pipeline/`, `rag/`) under `demand_forecasting/`.
  - Splits Python files into **object-level chunks** (functions/classes/methods), not arbitrary text.
  - Stores for each chunk:
    - `file_path`
    - `object_name` (e.g. `TSPreprocessor.run`)
    - start/end lines
    - source text
  - Embeds chunks with **OpenAI embeddings** (e.g. `text-embedding-3-small`).
  - Saves embeddings + metadata in a local `.rag_index.pkl` (NumPy array + Python objects).
- **Retrieval:**
  - Embed the query, compute **cosine similarity** vs. all chunk embeddings (pure NumPy).
  - Take top-k chunks (e.g. 8) and build a context block like:
    - `[1] ts0_preprocessing.py :: TSPreprocessor.run (lines …)`
- **Generation:**
  - Call an OpenAI **chat model** (e.g. `gpt-4o-mini`) with:
    - A focused **system prompt** (only use provided context, cite files/functions, explain math/stats when relevant).
    - **User message** = context block + question.
    - **Short conversation history** so follow-ups like “and how is that decision made?” stay grounded.

### Design Choices / Trade-offs
- **Local NumPy index** instead of Pinecone/FAISS:
  - Repo is small → in-memory is simple and fast.
  - Retrieval and generation are cleanly separated, so swapping in a vector DB later is easy.
- **Object-level chunking**:
  - Preserves semantic boundaries (methods, classes) → better for “how does this function work?” questions.
- **Config-driven models**:
  - `RAGConfig` holds `embedding_model`, `chat_model`, `chunk_size`, `top_k`, temperature, etc.
  - Easy to start cheap and upgrade models later with no code changes.

### How I’d Describe It as an AI Engineer
- It’s a minimal but real **RAG pipeline**:
  - Document store = repo.
  - Index = chunk + embed + store.
  - Retrieval = vector search + metadata.
  - Generation = LLM with domain-specific instructions + history.
- Shows understanding of:
  - **Retrieval vs. generation** separation.
  - How to structure context (`top_k`, chunking strategy) around a concrete use case.
  - Cost/latency trade-offs via model and index choices.

---

## 2. Aider as an “Agentic” Coding Assistant

### Goal / Use Case
- Use **Aider** as a **tool-using coding assistant** to:
  - Refactor legacy forecasting code into cleaner modules.
  - Enforce consistency (type hints, docstrings, naming).
  - Speed up small restructures and test scaffolding.

### How It Fits My Workflow
- Installed Aider inside the repo’s **virtual environment**.
- Run from the **repo root**, so it can:
  - See the git repo and propose changes as **diffs**.
- Configured with:
  - A cost-effective OpenAI model (e.g. `gpt-4o`/`gpt-4o-mini`) for editing.

Typical loop:

1. Start Aider in repo root.
2. Give a **scoped request**, e.g.:
   - “Add type hints and docstrings to `TSPreprocessor` and keep the public API stable.”
   - “Refactor `train_test_split_panel` into a shared utility and update imports.”
   - “Generate a test file for `compute_wape` with edge cases.”
3. Aider:
   - Reads relevant files,
   - Proposes git-style diffs.
4. I:
   - Review diffs,
   - Run tests / quick sanity checks,
   - Commit if I’m happy. (I can also allow Aider to commit directly if I’m confident.)

### Why It’s “Agentic”
- The LLM doesn’t just answer questions; it **acts on the codebase**:
  - Reads files, edits them, interacts with git.
- In AI-engineer terms, it’s a **tool-using agent**:
  - An LLM driving tools like “edit file,” “show diff,” “commit,” to achieve development goals.
- This pattern generalizes to:
  - Auto-refactoring agents,
  - AI code reviewers,
  - Agents that maintain ML / forecasting pipelines over time.

### Tie-back to the Job Description
- **RAG:** I’ve implemented a working RAG system for a real codebase, with configurable models and retrieval parameters.
- **Agentic workflows:** I use Aider as a lightweight agent that edits code and works with git.
- **Engineering mindset:** Everything is config-driven, version-controlled, and designed so components (models, index backend) can be swapped as the stack evolves.
