# Offline BSB search and answers

The target is a dual-core x86-64 Windows 10 computer with 4 GB RAM, no GPU,
and no network at runtime. Prepare models, Python packages and the indexed
corpus while connected, then use loopback Ollama and a native Qdrant server.

## Hardware budget and expectations

| Component | Estimated RAM |
| --- | --- |
| Windows 10 | 1.5–2.0 GB |
| Browser tab | 0.2–0.4 GB |
| Native Qdrant | 0.15–0.3 GB |
| Python app | 0.6–0.9 GB |
| LLM 0.5B Q4 | 0.5–0.7 GB |
| LLM 1–1.7B Q4 | 1.0–1.5 GB |

These totals can exceed 4 GB. Start with 0.5B if paging occurs; 1.5B is a
quality candidate, not a guarantee of comfortable operation. Docker Desktop
is unsuitable for this target: its WSL2 VM can consume another 1–2 GB.
Close other apps and browser tabs, keep `LLM_NUM_CTX` at or below 3072, and
expect much slower inference without AVX2.

**Latency estimates to be measured on the target**, not benchmark results:

- 0.5B: approximately 15–35 seconds per answer.
- 1–1.7B: approximately 45–90 seconds per answer.
- Retrieval only: approximately 1–3 seconds.

CPU prompt processing is a major cost. The measured corpus is approximately
24,300 points: scripture 4,655 chunks (~330 average tokens), topical 4,655
(~740 average, ~1,300 p90), concordance 14,782 (~170 average), and outlines
242 (~310 average). Six results can provide 2,000–4,500 tokens, occasionally
~7,800. Low mode uses three results, a 1,600-token context budget and a
400-token cap per chunk, with 10 initial retrieval candidates.

## Connected build machine

Use the same embedding model, Qdrant version, Python version and dependency
versions on both machines. Build on Windows x86-64 if preparing Windows
wheels. Run from the repository root in the `beresbirag` Conda environment:

```powershell
conda run -n beresbirag pip install ".[offline]"
$env:OFFLINE_MODE = "false"
$env:EMBEDDING_PROVIDER = "fastembed"
$env:QDRANT_MODE = "remote"
$env:MODEL_CACHE_DIR = "./.model_cache"
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
conda run -n beresbirag python ./scripts/prefetch_models.py --ollama-model qwen2.5:1.5b
conda run -n beresbirag python ./scripts/ingest_all.py --recreate
conda run -n beresbirag python ./scripts/qdrant_snapshot.py export --out ./snapshots/bsb_rag.snapshot
```

Start Qdrant before ingestion. `--recreate` replaces the configured collection;
use the intended build collection. Prefetch downloads the configured FastEmbed
dense/sparse models and both FlashRank MiniLM-L-12 and TinyBERT-L-2 models.
It pulls the requested Ollama tag if the Ollama CLI is installed; otherwise it
prints the command to run manually. `OFFLINE_MODE` must be false during prefetch.

Copy the snapshot **and** `bsb_rag.snapshot.json`, the entire `.model_cache/`
directory (including Hugging Face snapshot/blob files), and the application.
The sidecar records collection, point count, dense dimension, embedding provider,
dense/sparse model names, and creation time. Export while ingestion is stopped
so the reported point count describes the snapshot consistently.

If the target cannot connect even during setup, prepare a wheel directory on
a matching Windows/Python machine:

```powershell
conda run -n beresbirag pip wheel ".[offline]" --wheel-dir ./dist/offline-wheels
```

Copy that directory and install with `pip install --no-index --find-links
./dist/offline-wheels "beresbirag[offline]"` in an already provisioned compatible
Python environment. Transfer the Python/Conda and Ollama installers separately.
Ollama weights are stored separately from `.model_cache/`; copy its complete
model store if `ollama pull` cannot run on the target. See the official
[Ollama model storage instructions](https://docs.ollama.com/faq#where-are-models-stored).

## Target machine

1. Install Python/Conda, the application with its `offline` extra, and
   [Ollama for Windows](https://docs.ollama.com/windows). While still connected,
   run `ollama pull qwen2.5:1.5b`; use `ollama pull qwen2.5:0.5b` as the smaller
   fallback. Alternatively transfer the prepared Ollama model store.
2. Download `qdrant-x86_64-pc-windows-msvc.zip` from the
   [official Qdrant releases](https://github.com/qdrant/qdrant/releases), matching
   the build server version. Extract and run `./qdrant.exe` from its directory.
   The Windows asset was verified in release v1.20.0. Use the native binary;
   do not install Docker Desktop for this machine. Keep services on loopback
   for this single-user deployment.
3. Copy `.model_cache/` to the application root and place the snapshot and its
   sidecar under `snapshots/`. If `FLASHRANK_CACHE_DIR` is set, unset it or point
   it at the copied model cache; it overrides the shared cache for the reranker.
4. Create `.env` with the following configuration (replace duplicate settings
   instead of appending competing values):

```dotenv
RESOURCE_PROFILE=low
OFFLINE_MODE=true
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_MODE=remote
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=bsb_rag
EMBEDDING_PROVIDER=fastembed
FASTEMBED_DENSE_MODEL=BAAI/bge-small-en-v1.5
FASTEMBED_SPARSE_MODEL=Qdrant/bm25
MODEL_CACHE_DIR=./.model_cache
```

5. Import and check the installation:

```powershell
conda run -n beresbirag python ./scripts/qdrant_snapshot.py import --file ./snapshots/bsb_rag.snapshot
conda run -n beresbirag beresbirag doctor
conda run -n beresbirag beresbirag query "What does John 3:16 teach about love?" --provider ollama
conda run -n beresbirag beresbirag query "John 3:16" --provider ollama
conda run -n beresbirag beresbirag retrieve "Genesis 1:1"
```

Snapshot import restores the configured collection with snapshot priority,
so use the intended target collection. Import rejects mismatched dense model,
dimension, embedding provider, sparse model or collection before upload, then
validates the restored schema. The upload/download use the `api-key` header
when `QDRANT_API_KEY` is set. Local embedded Qdrant mode does not provide this
server snapshot interface; `QDRANT_MODE=remote` means the loopback native server.
See [Qdrant snapshot recovery](https://qdrant.tech/documentation/database-tutorials/create-snapshot/).

Disconnect external networking and repeat the commands to validate runtime
offline operation on the actual target. `doctor` exits 1 for failures and shows
RAM, CPU cores, AVX2, Qdrant schema/point count, local model files, Ollama tags,
and offline/provider consistency. It does not download or load neural models.
Missing caches must be prepared online and transferred again. Missing FlashRank
falls back to score-based reranking without downloading in offline mode.

## Settings and behavior

| Setting | Standard | Low profile |
| --- | --- | --- |
| `LLM_NUM_CTX` | 4096 | 3072 |
| `LLM_NUM_PREDICT` | 512 | 384 |
| `LLM_NUM_THREAD` | automatic | 2 |
| `CONTEXT_TOKEN_BUDGET` | unlimited | 1600 |
| `CONTEXT_CHUNK_TOKEN_CAP` | unlimited | 400 |
| `RETRIEVAL_TOP_K` | 6 | 3 |
| `RETRIEVAL_INITIAL_K` | 25 | 10 |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-12-v2 | ms-marco-TinyBERT-L-2-v2 |
| `FAST_PATH_ENABLED` | false | true |

Explicit environment values override profile defaults. Both low profile and
offline mode force FastEmbed; offline mode blocks cloud LLM generation and
sets `HF_HUB_OFFLINE` if unset. FastEmbed also receives `local_files_only=True`.
Configure local Ollama and Qdrant URLs for truly disconnected operation.

Ollama uses temperature 0, a 30-minute keep-alive and `reasoning=False`.
Ollama or low mode selects a compact grounding prompt. No chat model is created
until generation or explicit access to `pipeline.llm` is needed. Bare references
and concordance-count questions can be answered without an API key or Ollama;
interpretive questions use the LLM. Standard CLI cloud queries retain their
credential preflight when fast paths are disabled. The answer panel reports
`llm`, `fast_path`, or `insufficient_context`.

Context compression drops irrelevant footnotes and reference lists and preserves
whole lines. At least one chunk is retained; an indivisible first line can exceed
a very small configured cap. Fast quotations use the original retrieved verse
lines before compression. Ollama additionally checks the complete formatted
prompt, including source headers, question, chat-template margin and output
reserve; it removes trailing chunks/lines or rejects a prompt that cannot fit.
Token counts use `ceil(characters / 4)`, a corpus estimate rather than the exact
model tokenizer. Validate actual token usage, especially for non-English queries
or custom prompts, when benchmarking. No latency or tokenizer guarantee is made.

The reusable API is `prepare(query, top_k=None)`, `stream(prepared)` /
`astream(prepared)`, and `finalize(prepared, answer)`. `ask` and `aask` combine
these steps. CLI/API omitted `top_k` values use the active profile.

## TASK-11 benchmark candidates

Benchmark `qwen2.5:0.5b`, `qwen2.5:1.5b`, `llama3.2:1b`, `gemma3:1b`, and
`qwen3:1.7b` on the actual target. Record cold/warm latency, RAM/paging, prompt
tokens, citation accuracy and unsupported-answer behavior. The 4 GB target,
real Ollama inference and snapshot round trip require hardware/service testing;
the automated acceptance suite uses fakes and never downloads models.

## Running the UI on a 4 GB machine

When running the Streamlit interface on a resource-constrained 4 GB device:
- Use one browser tab and close other applications to conserve memory.
- Prefer **Retrieve only (no LLM)** mode for instant scripture reference lookups, and switch to LLM mode when summary synthesis is required.
- The first load takes longer due to local model warm-up (FastEmbed, BM25, and FlashRank caches).
- Keep the server host at 127.0.0.1 (local loopback only) for security and resource efficiency.
