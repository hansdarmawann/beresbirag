"""FastAPI application exposing BSB hybrid retrieval, grounded QA, and SSE streaming endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from beresbirag.config import Settings, get_settings
from beresbirag.llm.factory import (
    DEFAULT_MODELS,
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
    resolve_llm_target,
)
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.logging_config import configure_logging, request_id_var
from beresbirag.models.schemas import RAGResponse, RetrievalResult
from beresbirag.vectorstore.qdrant_manager import CollectionConfigError, QdrantManager

logger = logging.getLogger(__name__)

# Known configuration errors and their HTTP status; anything else maps to 500.
CONFIG_ERROR_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (MissingAPIKeyError, 503),
    (ProviderUnavailableError, 503),
    (CollectionConfigError, 503),
    (ProviderModelMismatchError, 422),
)


class TokenBucketRateLimiter:
    """Thread-safe in-memory token bucket rate limiter keyed by caller identity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, int], tuple[float, float]] = {}

    def check(self, identity: str, rate_per_minute: int) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        if rate_per_minute <= 0:
            return True, 0
        capacity = float(rate_per_minute)
        refill_per_sec = capacity / 60.0
        now = time.monotonic()
        key = (identity, rate_per_minute)

        with self._lock:
            tokens, last_ts = self._buckets.get(key, (capacity, now))
            elapsed = max(0.0, now - last_ts)
            tokens = min(capacity, tokens + elapsed * refill_per_sec)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = (tokens, now)
                return True, 0
            self._buckets[key] = (tokens, now)
            needed = 1.0 - tokens
            retry_after = max(1, int(math.ceil(needed / refill_per_sec)))
            return False, retry_after

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def build_retriever(settings: Settings) -> Any:
    """Construct the shared BSBHybridRetriever instance (monkeypatchable in tests)."""
    from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever

    return BSBHybridRetriever(settings=settings)


def build_pipeline(
    retriever: Any,
    settings: Settings,
    provider: Literal["openai", "anthropic", "ollama"] | None = None,
    model_name: str | None = None,
    effort: Literal["minimal", "low", "medium", "high"] | None = None,
) -> BSBRAGPipeline:
    """Construct a BSBRAGPipeline instance (monkeypatchable in tests)."""
    return BSBRAGPipeline(
        retriever=retriever,
        settings=settings,
        provider=provider,
        model_name=model_name,
        effort=effort,
    )


def _init_app_state(
    target_app: FastAPI, settings: Settings, *, build_shared_retriever: bool = True
) -> None:
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    valid_keys = _parse_api_keys(settings)
    if not valid_keys:
        logger.warning(
            "API_KEYS is empty; POST endpoints are running without API key authentication."
        )
    if build_shared_retriever:
        target_app.state.retriever = build_retriever(settings)
    elif not hasattr(target_app.state, "retriever"):
        target_app.state.retriever = None
    target_app.state.pipelines = {}
    target_app.state.pipelines_lock = threading.Lock()
    target_app.state.semaphore_limit = settings.max_concurrent_requests
    target_app.state.semaphore = threading.BoundedSemaphore(settings.max_concurrent_requests)
    target_app.state.rate_limiter = TokenBucketRateLimiter()


def _ensure_state_initialized(target_app: FastAPI, settings: Settings) -> None:
    if not hasattr(target_app.state, "pipelines_lock"):
        _init_app_state(target_app, settings, build_shared_retriever=False)
    if getattr(target_app.state, "semaphore_limit", None) != settings.max_concurrent_requests:
        target_app.state.semaphore_limit = settings.max_concurrent_requests
        target_app.state.semaphore = threading.BoundedSemaphore(settings.max_concurrent_requests)


@asynccontextmanager
async def lifespan(target_app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared retriever, pipeline cache, concurrency semaphore, and logging."""
    settings = get_settings()
    override_get_settings = target_app.dependency_overrides.get(get_settings)
    if override_get_settings is not None:
        settings = override_get_settings()
    has_retriever_override = get_api_retriever in target_app.dependency_overrides
    _init_app_state(
        target_app,
        settings,
        build_shared_retriever=not has_retriever_override,
    )
    try:
        yield
    finally:
        retriever = getattr(target_app.state, "retriever", None)
        manager = getattr(retriever, "qdrant_manager", None)
        client = getattr(manager, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass


app = FastAPI(
    title="Berean Standard Bible RAG API",
    description="Grounded Scripture retrieval and small-LLM QA service powered by LangChain and Qdrant.",
    version="0.1.0",
    lifespan=lifespan,
)


def _parse_api_keys(settings: Settings) -> list[str]:
    return [k.strip() for k in settings.api_keys.split(",") if k.strip()]


def _allowed_model_pairs(settings: Settings) -> set[str]:
    if settings.allowed_models and settings.allowed_models.strip():
        return {item.strip().lower() for item in settings.allowed_models.split(",") if item.strip()}
    pairs = {f"{prov}:{mod}".lower() for prov, mod in DEFAULT_MODELS.items()}
    if settings.llm_model and settings.llm_model.strip():
        pairs.add(f"{settings.llm_provider}:{settings.llm_model.strip()}".lower())
    return pairs


def _validate_query_length(query: str, settings: Settings) -> None:
    if len(query) > settings.max_query_chars:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Query length ({len(query)} chars) exceeds MAX_QUERY_CHARS "
                f"({settings.max_query_chars})."
            ),
        )


def _validate_allowed_model(
    provider: str | None,
    model: str | None,
    settings: Settings,
) -> tuple[str, str]:
    try:
        resolved_provider, resolved_model = resolve_llm_target(provider, model, settings)
    except ProviderModelMismatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pair = f"{resolved_provider}:{resolved_model}".lower()
    allowed = _allowed_model_pairs(settings)
    if pair not in allowed:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{resolved_provider}:{resolved_model}' is not permitted by ALLOWED_MODELS."
            ),
        )
    return resolved_provider, resolved_model


def enforce_auth_and_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Enforce X-API-Key authentication (on POST) and per-caller token bucket rate limiting."""
    _ensure_state_initialized(request.app, settings)
    valid_keys = _parse_api_keys(settings)
    provided_key = request.headers.get("X-API-Key", "")

    matched_key: str | None = None
    if valid_keys:
        for candidate in valid_keys:
            if provided_key and secrets.compare_digest(provided_key, candidate):
                matched_key = candidate
                break
        if matched_key is None and request.method.upper() == "POST":
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    identity = (
        f"key:{matched_key}"
        if matched_key is not None
        else f"ip:{request.client.host if request.client else 'unknown'}"
    )
    limiter: TokenBucketRateLimiter = request.app.state.rate_limiter
    allowed, retry_after = limiter.check(identity, settings.rate_limit_per_minute)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


@contextmanager
def _acquire_concurrency_slot(target_app: FastAPI, settings: Settings) -> Iterator[None]:
    _ensure_state_initialized(target_app, settings)
    sem: threading.BoundedSemaphore = target_app.state.semaphore
    acquired = sem.acquire(timeout=settings.queue_timeout_s)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy, retry later")
    try:
        yield
    finally:
        sem.release()


def _log_internal_error(exc: Exception) -> JSONResponse:
    error_id = str(uuid.uuid4())
    logger.exception(
        "Unhandled internal server error",
        extra={"error_id": error_id},
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error", "error_id": error_id},
    )


def _handle_known_or_internal_error(exc: Exception) -> JSONResponse:
    for exc_type, status_code in CONFIG_ERROR_STATUS:
        if isinstance(exc, exc_type):
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
    return _log_internal_error(exc)


@app.exception_handler(MissingAPIKeyError)
async def _missing_key_handler(request: Request, exc: MissingAPIKeyError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ProviderUnavailableError)
async def _provider_unavailable_handler(
    request: Request, exc: ProviderUnavailableError
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(CollectionConfigError)
async def _collection_config_handler(request: Request, exc: CollectionConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ProviderModelMismatchError)
async def _provider_mismatch_handler(
    request: Request, exc: ProviderModelMismatchError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return _handle_known_or_internal_error(exc)


@app.middleware("http")
async def request_id_and_logging_middleware(
    request: Request, call_next: Callable[[Request], Any]
) -> Response:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        response = _handle_known_or_internal_error(exc)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "HTTP request completed",
        extra={
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    request_id_var.reset(token)
    return response


class RetrieveRequest(BaseModel):
    """Payload for POST /retrieve."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Scripture reference, topic, or search query")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Maximum number of chunks to return"
    )


class QueryRequest(BaseModel):
    """Payload for POST /query and POST /query/stream."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Bible question or topic inquiry")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Maximum context chunks to retrieve"
    )
    provider: Literal["openai", "anthropic", "ollama"] | None = Field(
        default=None, description="LLM provider: 'openai', 'anthropic', or 'ollama'"
    )
    model: str | None = Field(default=None, description="Specific model name override")
    effort: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort override"
    )


def get_api_retriever(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Any:
    """Dependency provider returning the shared BSB hybrid retriever from app.state."""
    _ensure_state_initialized(request.app, settings)
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        retriever = build_retriever(settings)
        request.app.state.retriever = retriever
    return retriever


def get_pipeline_factory(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Any, Depends(get_api_retriever)],
) -> Callable[..., BSBRAGPipeline]:
    """Dependency provider returning a cached factory for BSBRAGPipeline."""
    _ensure_state_initialized(request.app, settings)

    def _factory(
        provider: Literal["openai", "anthropic", "ollama"] | None = None,
        model_name: str | None = None,
        effort: Literal["minimal", "low", "medium", "high"] | None = None,
    ) -> BSBRAGPipeline:
        resolved_provider, resolved_model = resolve_llm_target(provider, model_name, settings)
        resolved_effort = effort or settings.llm_effort
        cache_key = (resolved_provider, resolved_model, resolved_effort)
        pipelines: dict[tuple[str, str, str], BSBRAGPipeline] = request.app.state.pipelines
        lock: threading.Lock = request.app.state.pipelines_lock
        with lock:
            cached = pipelines.get(cache_key)
            if cached is None or cached.retriever is not retriever:
                cached = build_pipeline(
                    retriever=retriever,
                    settings=settings,
                    provider=resolved_provider,  # type: ignore[arg-type]
                    model_name=resolved_model,
                    effort=resolved_effort,  # type: ignore[arg-type]
                )
                pipelines[cache_key] = cached
            return cached

    return _factory


def _extract_token_usage(pipeline: Any, response: RAGResponse) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None) or getattr(pipeline, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    return None, None


def _log_answered_query(
    response: RAGResponse,
    *,
    top_k: int | None,
    settings: Settings,
    pipeline: Any = None,
) -> None:
    tokens_in, tokens_out = _extract_token_usage(pipeline, response)
    timings = response.timings or {}
    logger.info(
        "Answered RAG query",
        extra={
            "request_id": request_id_var.get(),
            "intent": response.intent.value
            if hasattr(response.intent, "value")
            else str(response.intent),
            "answer_mode": response.answer_mode.value
            if hasattr(response.answer_mode, "value")
            else str(response.answer_mode),
            "provider": response.provider,
            "model": response.model,
            "top_k": top_k if top_k is not None else settings.retrieval_top_k,
            "retrieval_ms": timings.get("retrieval_ms", 0.0),
            "generation_ms": timings.get("generation_ms", 0.0),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },
    )


@app.get("/livez", response_model=dict[str, str])
def livez_endpoint() -> dict[str, str]:
    """Liveness probe with zero external dependencies."""
    return {"status": "alive"}


@app.get("/health", response_model=dict[str, str])
def health_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Backward-compatible liveness alias reporting service status and target collection."""
    return {"status": "ok", "collection": settings.qdrant_collection_name}


def _resolve_readiness_manager(request: Request, settings: Settings) -> Any:
    if (
        hasattr(request.app.state, "qdrant_manager")
        and request.app.state.qdrant_manager is not None
    ):
        return request.app.state.qdrant_manager
    override_retriever_fn = request.app.dependency_overrides.get(get_api_retriever)
    retriever = (
        override_retriever_fn()
        if override_retriever_fn is not None
        else getattr(request.app.state, "retriever", None)
    )
    if retriever is not None:
        if getattr(retriever, "qdrant_manager", None) is not None:
            return retriever.qdrant_manager
        if hasattr(retriever, "collection_exists"):
            return retriever
    return QdrantManager(settings=settings)


@app.get("/readyz")
def readyz_endpoint(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse:
    """Readiness probe verifying Qdrant reachability, collection population, config, and index metadata."""
    checks: dict[str, dict[str, Any]] = {}
    failing: list[str] = []

    manager = _resolve_readiness_manager(request, settings)
    exists = False
    try:
        exists = bool(manager.collection_exists())
        checks["qdrant_reachable"] = {"status": "ok", "detail": "reachable"}
    except Exception as exc:
        checks["qdrant_reachable"] = {"status": "fail", "detail": str(exc)}
        failing.append("qdrant_reachable")

    if "qdrant_reachable" not in failing:
        if not exists:
            checks["collection_populated"] = {
                "status": "fail",
                "detail": f"Collection '{settings.qdrant_collection_name}' does not exist",
            }
            failing.append("collection_populated")
        else:
            try:
                pts = int(manager.count_points())
                if pts > 0:
                    checks["collection_populated"] = {"status": "ok", "points": pts}
                else:
                    checks["collection_populated"] = {
                        "status": "fail",
                        "detail": "Collection has 0 points",
                    }
                    failing.append("collection_populated")
            except Exception as exc:
                checks["collection_populated"] = {"status": "fail", "detail": str(exc)}
                failing.append("collection_populated")

        if not exists:
            checks["collection_config"] = {
                "status": "fail",
                "detail": "Collection missing",
            }
            failing.append("collection_config")
        else:
            try:
                manager.validate_collection()
                checks["collection_config"] = {"status": "ok"}
            except Exception as exc:
                checks["collection_config"] = {"status": "fail", "detail": str(exc)}
                failing.append("collection_config")

        if not exists:
            checks["index_metadata"] = {
                "status": "fail",
                "detail": "Collection missing",
            }
            failing.append("index_metadata")
        else:
            try:
                meta = (
                    manager.read_index_metadata()
                    if hasattr(manager, "read_index_metadata")
                    else None
                )
                if isinstance(meta, dict):
                    provider = settings.resolved_embedding_provider
                    expected_dense = (
                        settings.openai_embedding_model
                        if provider == "openai"
                        else settings.fastembed_dense_model
                    )
                    expected_sparse = settings.fastembed_sparse_model
                    actual_dense = meta.get("dense_model")
                    actual_sparse = meta.get("sparse_model")
                    if (actual_dense and actual_dense != expected_dense) or (
                        actual_sparse and actual_sparse != expected_sparse
                    ):
                        checks["index_metadata"] = {
                            "status": "fail",
                            "detail": "Index metadata model mismatch",
                        }
                        failing.append("index_metadata")
                    else:
                        checks["index_metadata"] = {"status": "ok", "metadata": meta}
                else:
                    checks["index_metadata"] = {"status": "ok", "detail": "validated"}
            except Exception as exc:
                checks["index_metadata"] = {"status": "fail", "detail": str(exc)}
                failing.append("index_metadata")

    checks["llm_provider"] = {
        "status": "info",
        "provider": settings.llm_provider,
        "offline_mode": settings.offline_mode,
    }

    if failing:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": checks,
                "failing_checks": failing,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "checks": checks,
            "failing_checks": [],
        },
    )


@app.post(
    "/retrieve",
    response_model=list[RetrievalResult],
    dependencies=[Depends(enforce_auth_and_rate_limit)],
)
def retrieve_endpoint(
    payload: RetrieveRequest,
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Any, Depends(get_api_retriever)],
) -> list[RetrievalResult] | JSONResponse:
    """Retrieve ranked context chunks for a Bible query or reference."""
    _validate_query_length(payload.query, settings)
    with _acquire_concurrency_slot(http_request.app, settings):
        try:
            _plan, results = retriever.retrieve(payload.query, top_k=payload.top_k)
            return results
        except HTTPException:
            raise
        except Exception as exc:
            return _handle_known_or_internal_error(exc)


@app.post(
    "/query",
    response_model=RAGResponse,
    dependencies=[Depends(enforce_auth_and_rate_limit)],
)
def query_endpoint(
    payload: QueryRequest,
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline_factory: Annotated[Callable[..., BSBRAGPipeline], Depends(get_pipeline_factory)],
) -> RAGResponse | JSONResponse:
    """Answer a Bible inquiry with grounded context citations and verified small-LLM output."""
    _validate_query_length(payload.query, settings)
    _validate_allowed_model(payload.provider, payload.model, settings)
    with _acquire_concurrency_slot(http_request.app, settings):
        try:
            pipeline = pipeline_factory(
                provider=payload.provider,
                model_name=payload.model,
                effort=payload.effort,
            )
            response = pipeline.ask(payload.query, top_k=payload.top_k)
            _log_answered_query(
                response,
                top_k=payload.top_k,
                settings=settings,
                pipeline=pipeline,
            )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            return _handle_known_or_internal_error(exc)


def _format_sse(event: str, data: Any) -> str:
    serialized = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {serialized}\n\n"


@app.post(
    "/query/stream",
    dependencies=[Depends(enforce_auth_and_rate_limit)],
)
async def query_stream_endpoint(
    payload: QueryRequest,
    http_request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline_factory: Annotated[Callable[..., BSBRAGPipeline], Depends(get_pipeline_factory)],
) -> Response:
    """Stream grounded RAG execution via Server-Sent Events (meta -> token -> done)."""
    _validate_query_length(payload.query, settings)
    _validate_allowed_model(payload.provider, payload.model, settings)

    _ensure_state_initialized(http_request.app, settings)
    sem: threading.BoundedSemaphore = http_request.app.state.semaphore
    acquired = await asyncio.to_thread(sem.acquire, True, settings.queue_timeout_s)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy, retry later")

    try:
        pipeline = pipeline_factory(
            provider=payload.provider,
            model_name=payload.model,
            effort=payload.effort,
        )
        prepared = await asyncio.to_thread(pipeline.prepare, payload.query, payload.top_k)
    except HTTPException:
        sem.release()
        raise
    except Exception as exc:
        sem.release()
        return _handle_known_or_internal_error(exc)

    async def _event_stream() -> AsyncIterator[str]:
        try:
            intent_val = (
                prepared.plan.intent.value
                if prepared.plan is not None and hasattr(prepared.plan.intent, "value")
                else str(getattr(prepared.plan, "intent", "hybrid_general"))
            )
            mode_val = (
                prepared.answer_mode.value
                if hasattr(prepared.answer_mode, "value")
                else str(prepared.answer_mode)
            )
            retrieved_summary = [
                {
                    "rank": r.rank,
                    "canonical_ref": r.chunk.metadata.canonical_ref or r.chunk.chunk_id,
                    "source_type": r.chunk.metadata.source_type.value
                    if hasattr(r.chunk.metadata.source_type, "value")
                    else str(r.chunk.metadata.source_type),
                }
                for r in prepared.results
            ]
            yield _format_sse(
                "meta",
                {
                    "intent": intent_val,
                    "answer_mode": mode_val,
                    "retrieved": retrieved_summary,
                },
            )

            collected: list[str] = []
            async for delta in pipeline.astream(prepared):
                text_delta = str(delta)
                collected.append(text_delta)
                yield _format_sse("token", {"token": text_delta})

            final_response = pipeline.finalize(prepared, "".join(collected))
            _log_answered_query(
                final_response,
                top_k=payload.top_k,
                settings=settings,
                pipeline=pipeline,
            )
            yield _format_sse("done", final_response.model_dump(mode="json"))
        except Exception as exc:
            error_id = str(uuid.uuid4())
            logger.exception(
                "Error during SSE stream generation",
                extra={"error_id": error_id},
                exc_info=exc,
            )
            yield _format_sse("error", {"error_id": error_id})
        finally:
            sem.release()

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
