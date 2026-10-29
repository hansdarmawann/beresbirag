# Operations Guide (`beresbirag`)

This document covers production deployment, operational hardening, observability, reproducible dependencies/data, Docker profiles, and the UI feedback curation workflow.

---

## 1. Operational Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_CONCURRENT_REQUESTS` | `4` (`1` when `RESOURCE_PROFILE=low`) | Bounded semaphore limit across `/retrieve`, `/query`, and `/query/stream`. |
| `QUEUE_TIMEOUT_S` | `30.0` | Maximum seconds a request waits for a concurrency slot before returning `503 {"detail": "Server busy, retry later"}`. |
| `API_KEYS` | `""` | Comma-separated API keys. When set, `POST` endpoints require a matching `X-API-Key` header (`secrets.compare_digest`); when empty, auth is disabled and a startup `WARNING` is logged. |
| `RATE_LIMIT_PER_MINUTE` | `30` | In-memory token-bucket rate limit per minute keyed by `X-API-Key` (or client IP when auth is off). `0` disables rate limiting. |
| `ALLOWED_MODELS` | `""` | Comma-separated `provider:model` allowlist. When empty, defaults to all `DEFAULT_MODELS` entries plus `LLM_PROVIDER:LLM_MODEL`. Disallowed pairs return `422`. |
| `MAX_QUERY_CHARS` | `1000` | Maximum query character length on `/retrieve`, `/query`, and `/query/stream`. Longer queries return `422`. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FORMAT` | `text` | Log output format: `text` or `json` (one JSON object per line). |
| `UI_PASSWORD` | `""` | Password gate for the Streamlit UI (`secrets.compare_digest`). Required whenever Streamlit binds to a non-loopback address (`0.0.0.0`). |
| `FEEDBACK_DIR` | `./feedback` | Directory where Streamlit UI 👍/👎 feedback entries are appended (`feedback.jsonl`). |

---

## 2. Shared Lifespan Resources & Concurrency

On startup, the FastAPI `lifespan` initializes:
- **One shared `BSBHybridRetriever`** (`app.state.retriever`), avoiding per-request reloading of Qdrant clients, FastEmbed dense/BM25 models, and FlashRank cross-encoders.
- **Thread-safe `BSBRAGPipeline` cache** (`app.state.pipelines` guarded by `threading.Lock`), keyed by `(provider, model, effort)`.
- **Bounded concurrency semaphore** (`threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)`). When all slots are occupied for longer than `QUEUE_TIMEOUT_S`, the server responds with `503 {"detail": "Server busy, retry later"}`.

---

## 3. Health & Readiness Semantics

- **`GET /livez`** (and backward-compatible alias **`GET /health`**):
  - Returns `200 {"status": "alive"}` (or `{"status": "ok", "collection": "..."}` on `/health`) with zero external dependencies.
- **`GET /readyz`**:
  - Verifies four readiness conditions:
    1. `qdrant_reachable`: Qdrant server responds.
    2. `collection_populated`: Target collection exists and has `points > 0`.
    3. `collection_config`: `validate_collection()` passes (dense vector dimension + sparse BM25 `modifier=idf`).
    4. `index_metadata`: Companion `{collection}__meta` metadata matches active `dense_model` and `sparse_model`.
  - Reports `llm_provider` status as `info` without failing readiness, since `/retrieve` operates without an LLM.
  - Returns `200` when all checks pass, or `503` listing `failing_checks`.
- `/livez`, `/readyz`, and `/health` are never authenticated or rate limited.

---

## 4. Authentication, Rate Limiting & Input Limits

- **API Key Authentication (`API_KEYS`)**:
  - When `API_KEYS` is non-empty, every `POST` endpoint (`/retrieve`, `/query`, `/query/stream`) requires `X-API-Key` matched via `secrets.compare_digest`. Missing or invalid keys return `401`.
- **Token-Bucket Rate Limiting (`RATE_LIMIT_PER_MINUTE`)**:
  - Enforced per API key (when authenticated) or per client IP (when unauthenticated).
  - Exceeding the bucket returns `429 Too Many Requests` with a `Retry-After` header in seconds.
- **Model Allowlist & Query Bounds**:
  - `top_k` is bounded to `1..20`.
  - Queries exceeding `MAX_QUERY_CHARS` or requesting a `provider:model` outside `ALLOWED_MODELS` return `422`.

---

## 5. Error Handling & Structured Observability

- **Sanitized Error Responses**:
  - `MissingAPIKeyError`, `ProviderUnavailableError`, `CollectionConfigError` → `503`.
  - `ProviderModelMismatchError` → `422`.
  - Any unexpected exception → `500 {"detail": "Internal error", "error_id": "<uuid4>"}`. Internal exception strings or file paths are never leaked to clients; full tracebacks are logged server-side with `error_id`.
- **Request Correlation & Logging Fields**:
  - Middleware accepts an incoming `X-Request-ID` header (or generates a UUID4) and echoes `X-Request-ID` in the response.
  - **Per-request log fields**: `request_id, method, path, status, duration_ms`.
  - **Per-answered-query log fields**: `request_id, intent, answer_mode, provider, model, top_k, retrieval_ms, generation_ms, tokens_in, tokens_out`. Raw query text is never logged at `INFO`.
  - **`RAGResponse.timings`**: Every response includes `{"retrieval_ms": ..., "generation_ms": ..., "total_ms": ...}`.

---

## 6. Server-Sent Events (`POST /query/stream`)

`POST /query/stream` accepts the same JSON body and guards as `POST /query` and returns `text/event-stream` emitting events in order:

1. **`event: meta`**:
   ```json
   {"intent": "exact_passage", "answer_mode": "llm", "retrieved": [{"rank": 1, "canonical_ref": "Genesis 1:1", "source_type": "scripture"}]}
   ```
2. **`event: token`** (one or more):
   ```json
   {"token": "In the beginning "}
   ```
3. **`event: done`**:
   Full serialized `RAGResponse` JSON (including `citations`, `citation_report`, and `timings`).
4. **`event: error`** (only if an exception occurs after streaming begins):
   ```json
   {"error_id": "<uuid4>"}
   ```

---

## 7. Streaming Ingestion, Index Metadata & Corpus Verification

### Constant-Memory Streaming Ingestion
`IngestionPipeline.run()` consumes source chunk iterators in `batch_size` slices via `itertools.islice` with `total=None` Rich progress tracking, ensuring peak buffered chunks never exceed `batch_size`.

### Companion Index Metadata (`{collection}__meta`)
At the end of ingestion, `QdrantManager.write_index_metadata()` writes a single point (`make_point_uuid("index-meta")`) to `{collection}__meta` recording:
- `embedding_provider`, `dense_model`, `dense_dim`, `sparse_model`, `sparse_modifier: "idf"`
- `chunking`: `{"max_verses_per_chunk", "verse_overlap"}`
- `corpus_manifest`: `{"<file>": "<sha256>"}`
- `beresbirag_version`, `ingested_at`

`validate_collection()` verifies that `dense_model` and `sparse_model` in `{collection}__meta` match the active settings, and `scripts/qdrant_snapshot.py` exports and imports both collections along with the metadata payload in the JSON sidecar.

### Reproducible Corpus Verification (`data/corpus_manifest.json`)
```powershell
# Verify local raw BSB files against data/corpus_manifest.json (exits 1 on mismatch)
conda run -n beresbirag python ./scripts/verify_corpus.py

# Regenerate file sizes and SHA-256 hashes in data/corpus_manifest.json
conda run -n beresbirag python ./scripts/verify_corpus.py --write

# Fail ingestion immediately if any raw corpus file mismatches the manifest
conda run -n beresbirag python ./scripts/ingest_all.py --recreate --strict-corpus
```

---

## 8. Reproducible Dependency Lockfile (`requirements.lock`)

Direct dependencies in `pyproject.toml` are upper-bounded to the installed major/minor series, and full transitive versions are pinned in `requirements.lock`.

To regenerate `requirements.lock` after updating `pyproject.toml`:
```powershell
conda run -n beresbirag pip-compile --extra dev --extra ui --extra offline --strip-extras -o requirements.lock pyproject.toml
```

---

## 9. Docker Compose Profiles

> **Note:** Docker targets servers and CI. For the 4 GB RAM offline target, use native Qdrant + Ollama as documented in [docs/OFFLINE.md](OFFLINE.md).

```powershell
# 1. Validate Compose configuration and build the hardened non-root image
docker compose config --quiet
docker build -t beresbirag:ops .

# 2. Start Qdrant (pinned v1.20.0) and the FastAPI service
docker compose up -d

# 3. Verify raw BSB files and run full ingestion via the 'ingest' profile
docker compose --profile ingest run --rm ingest

# 4. Launch the password-protected Streamlit UI via the 'ui' profile
$env:UI_PASSWORD="your-strong-password"
docker compose --profile ui up -d
```

---

## 10. UI Feedback → Evaluation Loop

1. **Collect Feedback in Streamlit**:
   - Users rate turns with 👍 or 👎 and optional comments in the Streamlit UI.
   - Entries are appended as JSON lines to `{FEEDBACK_DIR}/feedback.jsonl` with `ts, query, answer, answer_mode, provider, model, citation_report, timings, rating, comment`.
2. **Generate Draft Evaluation Cases**:
   ```powershell
   conda run -n beresbirag python ./scripts/feedback_to_eval.py
   ```
   - Converts only negative (`down` / `👎`) entries into `./eval/answer_cases.draft.jsonl` with `must_cite=[]` and `tags=["from-feedback"]`.
   - Never overwrites `./eval/answer_cases.jsonl`. Curate `must_cite` references manually before promoting cases into `eval/answer_cases.jsonl`.

