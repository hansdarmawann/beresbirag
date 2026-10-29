# syntax=docker/dockerfile:1
# Pinned from python:3.11-slim-bookworm
FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

LABEL org.opencontainers.image.title="beresbirag" \
      org.opencontainers.image.description="Berean Standard Bible (BSB) Hybrid RAG System" \
      org.opencontainers.image.authors="Hans Darmawan <work.hansdarmawan001@gmail.com>"

WORKDIR /app

# Create non-root user app (uid 10001)
RUN groupadd -g 10001 app && \
    useradd -u 10001 -g app -d /app -s /sbin/nologin -M app

# Install pinned dependencies from lockfile first for layer caching
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy application source, scripts, and corpus manifest
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/corpus_manifest.json ./data/corpus_manifest.json

RUN pip install --no-cache-dir --no-deps .

# Prefetch FastEmbed dense/sparse and FlashRank reranker models into /app/.model_cache at build time
RUN MODEL_CACHE_DIR=/app/.model_cache OFFLINE_MODE=false python scripts/prefetch_models.py && \
    chown -R app:app /app

ENV PYTHONUNBUFFERED="1" \
    MODEL_CACHE_DIR="/app/.model_cache" \
    HF_HUB_OFFLINE="1" \
    QDRANT_MODE="remote" \
    QDRANT_URL="http://qdrant:6333"

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/livez'); "

CMD ["uvicorn", "beresbirag.api:app", "--host", "0.0.0.0", "--port", "8000"]
