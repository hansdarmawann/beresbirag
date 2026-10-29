# Changelog

All notable changes to the `beresbirag` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-29

### Added
- **Repository Hygiene & Type Checking (TASK-13)**:
  - Centralized context formatting in `beresbirag.retrieval.formatting` used uniformly across retriever and RAG chain.
  - Instance-level caching of `QdrantVectorStore` and collection readiness checks in `QdrantManager`.
  - Configurable `BSB_DATA_DIR` environment setting eliminating Git worktree path probing in library code.
  - Repository-wide code formatting via `ruff format` and strict static typing via `mypy`.
  - GitHub Actions CI workflow enhancements: author allowlist (`Hans Darmawan`, `dependabot[bot]`), `--no-merges` validation, and automated mypy / ruff format checks.
  - Generalized `manage_worktrees.ps1` script supporting Waves 1–8 with automatic `.env` `BSB_DATA_DIR` initialization and cleanup of merged branches.
  - Formal MIT License (`LICENSE`) and dataset source notice (`NOTICE`).
- **Production Operations & Hardening (TASK-12)**:
  - Shared FastAPI lifespan management with graceful resource shutdown.
  - Kubernetes-compatible `/livez` liveness and `/readyz` readiness probe endpoints.
  - Header-based `X-API-Key` authentication and in-memory token-bucket rate limiting (`RATE_LIMIT_PER_MINUTE`).
  - Server-Sent Events (SSE) streaming query endpoint (`/query/stream`).
  - Structured JSON logging (`LOG_FORMAT=json`) with distributed `request_id` context propagation.
  - OpenXML streaming ingestion with index metadata verification (`{collection}__meta`).
  - Corpus integrity SHA-256 verification tool (`scripts/verify_corpus.py`) and UI feedback-to-eval converter (`scripts/feedback_to_eval.py`).
  - Dependency pinning via `requirements.lock` and multi-stage container optimization.
- **Citation Validation & Router Evaluation (TASK-11)**:
  - Dual-mode citation validation policy (`annotate` and `strict`).
  - 150-case multi-corpus retrieval and QA benchmark evaluation suite.
- **Interactive Streamlit Web UI (TASK-10)**:
  - Multi-tab browser interface supporting retrieve-only and full LLM QA modes.
  - Provider-safe model selection preventing invalid provider/model pairings.
  - Integrated citation inspector and 👍/👎 user feedback collection.
- **Offline & Low-Resource Profile (TASK-09)**:
  - Full local execution support via Ollama (`qwen2.5:1.5b`) and native local Qdrant.
  - Low-resource profile (`RESOURCE_PROFILE=low`) tuned for 4 GB RAM budgets.
  - Deterministic fast-path query handler for exact scripture lookups and concordance counts.
- **P0 Correctness & Alignment Fixes (TASK-08)**:
  - Qdrant BM25 modifier IDF server-side scoring fix.
  - Automatic embedding provider dimension guardrails (`CollectionConfigError`).
  - Book-scoped hybrid retrieval and exact reference filtering.

## [0.1.0] - 2026-10-28

### Added
- Initial release of the Berean Standard Bible (BSB) Hybrid RAG system.
- Support for USJ 3.1 Scripture, PDF Book Outlines, Excel Topical Index, and Excel Concordance.
- Qdrant Hybrid Search (Dense + Sparse BM25 RRF) with FlashRank passage reranking.
- Typer CLI (`beresbirag`) and baseline FastAPI REST endpoints.
