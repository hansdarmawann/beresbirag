"""Services, dependency injection, model options, and feedback persistence for Streamlit UI."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import streamlit as st

from beresbirag.config import Settings, get_settings
from beresbirag.llm.factory import DEFAULT_MODELS, FOREIGN_MODEL_PREFIXES
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever
from beresbirag.vectorstore.qdrant_manager import QdrantManager

# Optional module-level override for tests (fallback is FEEDBACK_DIR env var / Settings.feedback_dir)
FEEDBACK_DIR: str | Path | None = None


class ServiceFactory(Protocol):
    """Factory protocol for producing retrievers and pipelines."""

    def retriever(self) -> BSBHybridRetriever: ...

    def pipeline(
        self,
        provider: str,
        model: str,
        effort: str | None = None,
    ) -> BSBRAGPipeline: ...


_custom_service_factory: ServiceFactory | None = None


def set_service_factory(factory: ServiceFactory | None) -> None:
    """Set custom service factory (used for dependency injection and tests)."""
    global _custom_service_factory
    _custom_service_factory = factory


def get_service_factory() -> ServiceFactory:
    """Get active service factory or default Streamlit-cached factory."""
    if _custom_service_factory is not None:
        return _custom_service_factory
    return _DefaultCachedServiceFactory()


def resolve_feedback_dir(override: str | Path | None = None) -> Path:
    """Resolve the active feedback directory from explicit override, module override, env, or settings."""
    if override is not None:
        return Path(override)
    if FEEDBACK_DIR is not None:
        return Path(FEEDBACK_DIR)
    env_dir = os.environ.get("FEEDBACK_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip())
    return Path(get_settings().feedback_dir)


def append_feedback(
    turn: dict[str, Any],
    rating: Literal["up", "down", "👍", "👎"],
    comment: str = "",
    feedback_dir: str | Path | None = None,
) -> Path:
    """Append one feedback entry as a JSON line to ``{FEEDBACK_DIR}/feedback.jsonl``."""
    target_dir = resolve_feedback_dir(feedback_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / "feedback.jsonl"

    norm_rating = "down" if rating in ("down", "👎") else "up"
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "query": turn.get("query", ""),
        "answer": turn.get("answer", ""),
        "answer_mode": turn.get("answer_mode", "llm"),
        "provider": turn.get("provider", ""),
        "model": turn.get("model", ""),
        "citation_report": turn.get("citation_report"),
        "timings": turn.get("timings", {}),
        "rating": norm_rating,
        "comment": comment,
    }
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


@st.cache_resource(show_spinner=False)
def _get_cached_retriever(
    collection_name: str,
    qdrant_url: str,
    qdrant_mode: str,
    embedding_model: str,
    reranker_model: str,
    resource_profile: str,
) -> BSBHybridRetriever:
    """Instantiate and cache BSBHybridRetriever once per configuration."""
    settings = get_settings()
    manager = QdrantManager(settings=settings)
    return BSBHybridRetriever(qdrant_manager=manager, settings=settings)


@st.cache_resource(show_spinner=False)
def _get_cached_pipeline(
    provider: str,
    model: str,
    effort: str | None,
    collection_name: str,
    qdrant_url: str,
    qdrant_mode: str,
    embedding_model: str,
    reranker_model: str,
    resource_profile: str,
) -> BSBRAGPipeline:
    """Instantiate and cache BSBRAGPipeline once per (provider, model, effort, config)."""
    settings = get_settings()
    retriever = _get_cached_retriever(
        collection_name,
        qdrant_url,
        qdrant_mode,
        embedding_model,
        reranker_model,
        resource_profile,
    )
    return BSBRAGPipeline(
        retriever=retriever,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        model_name=model,
        effort=effort,  # type: ignore[arg-type]
    )


class _DefaultCachedServiceFactory:
    """Default implementation utilizing Streamlit cache_resource."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _cache_key(self) -> tuple[str, str, str, str, str, str]:
        s = self.settings
        return (
            s.qdrant_collection_name,
            s.qdrant_url,
            s.qdrant_mode,
            s.fastembed_dense_model,
            s.reranker_model,
            s.resource_profile,
        )

    def retriever(self) -> BSBHybridRetriever:
        return _get_cached_retriever(*self._cache_key())

    def pipeline(
        self,
        provider: str,
        model: str,
        effort: str | None = None,
    ) -> BSBRAGPipeline:
        return _get_cached_pipeline(
            provider,
            model,
            effort,
            *self._cache_key(),
        )


def fetch_installed_ollama_models(settings: Settings | None = None) -> list[str]:
    """Fetch installed Ollama model tags via Ollama HTTP API."""
    cfg = settings or get_settings()
    try:
        url = f"{cfg.ollama_base_url.rstrip('/')}/api/tags"
        with httpx.Client(timeout=2.0, trust_env=False) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m["name"] for m in data.get("models", []) if isinstance(m, dict) and "name" in m
                ]
                return models
    except Exception:
        pass
    return []


def model_options(
    provider: str,
    settings: Settings,
    installed_ollama_models: list[str] | None = None,
) -> list[str]:
    """Return deduplicated list of valid model choices for the given provider."""
    prov = provider.lower()
    foreign_prefixes = FOREIGN_MODEL_PREFIXES.get(prov, ())

    candidates: list[str] = []

    if prov == "openai":
        candidates.append(DEFAULT_MODELS["openai"])
        if settings.llm_provider == "openai" and settings.llm_model:
            candidates.append(settings.llm_model)
    elif prov == "anthropic":
        candidates.append(DEFAULT_MODELS["anthropic"])
        if settings.llm_provider == "anthropic" and settings.llm_model:
            candidates.append(settings.llm_model)
    elif prov == "ollama":
        candidates.append(DEFAULT_MODELS["ollama"])
        candidates.extend(
            [
                "qwen2.5:0.5b",
                "qwen2.5:1.5b",
                "llama3.2:1b",
                "gemma3:1b",
                "qwen3:1.7b",
            ]
        )
        if installed_ollama_models:
            candidates.extend(installed_ollama_models)
        if settings.llm_provider == "ollama" and settings.llm_model:
            candidates.append(settings.llm_model)

    # Filter out models that start with foreign prefixes and deduplicate preserving order
    result: list[str] = []
    for item in candidates:
        name = item.strip()
        if not name:
            continue
        if any(name.lower().startswith(prefix) for prefix in foreign_prefixes):
            continue
        if name not in result:
            result.append(name)

    return result


def provider_options(settings: Settings) -> list[str]:
    """Return available LLM providers based on offline mode and active provider."""
    if settings.offline_mode:
        return ["ollama"]

    all_providers = ["openai", "anthropic", "ollama"]
    active = settings.llm_provider.lower()
    if active in all_providers:
        # Put active first, followed by remaining
        return [active] + [p for p in all_providers if p != active]
    return all_providers
