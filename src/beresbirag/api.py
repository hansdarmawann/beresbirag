"""FastAPI application exposing BSB hybrid retrieval and grounded QA endpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from beresbirag.config import Settings, get_settings
from beresbirag.llm.factory import MissingAPIKeyError, ProviderModelMismatchError
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.models.schemas import RAGResponse, RetrievalResult
from beresbirag.vectorstore.qdrant_manager import CollectionConfigError

app = FastAPI(
    title="Berean Standard Bible RAG API",
    description="Grounded Scripture retrieval and small-LLM QA service powered by LangChain and Qdrant.",
    version="0.1.0",
)

# Known configuration errors and their HTTP status; anything else maps to 500.
CONFIG_ERROR_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (MissingAPIKeyError, 503),
    (ProviderModelMismatchError, 422),
    (CollectionConfigError, 503),
)


def _to_http_exception(exc: Exception) -> HTTPException:
    for exc_type, status_code in CONFIG_ERROR_STATUS:
        if isinstance(exc, exc_type):
            return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


class RetrieveRequest(BaseModel):
    """Payload for POST /retrieve."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., min_length=1, description="Scripture reference, topic, or search query"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Maximum number of chunks to return"
    )


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., min_length=1, description="Bible question or topic inquiry"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Maximum context chunks to retrieve"
    )
    provider: Literal["openai", "anthropic", "ollama"] | None = Field(
        default=None, description="LLM provider: 'openai', 'anthropic', or 'ollama'"
    )
    model: str | None = Field(
        default=None, description="Specific model name override"
    )
    effort: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort override"
    )


def get_api_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Any:
    """Dependency provider for the BSB hybrid retriever."""
    try:
        from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever

        return BSBHybridRetriever(settings=settings)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"BSB Retriever service unavailable: {exc}",
        ) from exc


def get_pipeline_factory(
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[Any, Depends(get_api_retriever)],
) -> Callable[..., BSBRAGPipeline]:
    """Dependency provider returning a callable factory for BSBRAGPipeline."""

    def _factory(
        provider: Literal["openai", "anthropic", "ollama"] | None = None,
        model_name: str | None = None,
        effort: Literal["minimal", "low", "medium", "high"] | None = None,
    ) -> BSBRAGPipeline:
        return BSBRAGPipeline(
            retriever=retriever,
            settings=settings,
            provider=provider,
            model_name=model_name,
            effort=effort,
        )

    return _factory


@app.get("/health", response_model=dict[str, str])
def health_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """Health check endpoint reporting service status and target Qdrant collection."""
    return {"status": "ok", "collection": settings.qdrant_collection_name}


@app.post("/retrieve", response_model=list[RetrievalResult])
def retrieve_endpoint(
    request: RetrieveRequest,
    retriever: Annotated[Any, Depends(get_api_retriever)],
) -> list[RetrievalResult]:
    """Retrieve ranked context chunks for a Bible query or reference."""
    try:
        _plan, results = retriever.retrieve(request.query, top_k=request.top_k)
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc) from exc


@app.post("/query", response_model=RAGResponse)
def query_endpoint(
    request: QueryRequest,
    pipeline_factory: Annotated[
        Callable[..., BSBRAGPipeline], Depends(get_pipeline_factory)
    ],
) -> RAGResponse:
    """Answer a Bible inquiry with grounded context citations and verified small-LLM output."""
    try:
        pipeline = pipeline_factory(
            provider=request.provider,
            model_name=request.model,
            effort=request.effort,
        )
        return pipeline.ask(request.query, top_k=request.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc) from exc
