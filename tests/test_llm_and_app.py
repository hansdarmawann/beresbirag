"""Acceptance tests for small-LLM factory, LCEL RAG chain, FastAPI, and CLI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from beresbirag.api import (
    app as fastapi_app,
)
from beresbirag.api import (
    get_api_retriever,
    get_pipeline_factory,
)
from beresbirag.cli import (
    app as typer_app,
)
from beresbirag.cli import (
    set_custom_pipeline_factory,
    set_custom_retriever,
)
from beresbirag.config import Settings
from beresbirag.llm.factory import create_chat_model, is_reasoning_model
from beresbirag.llm.prompts import BSB_RAG_PROMPT, SYSTEM_PROMPT
from beresbirag.llm.rag_chain import (
    BSBRAGPipeline,
    build_citations,
    format_context_blocks,
)
from beresbirag.models.schemas import (
    ChunkDocument,
    ChunkMetadata,
    QueryIntent,
    RetrievalResult,
    SourceType,
)


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Provide realistic sample chunks across Scripture, Outlines, and Topical sources."""
    chunk1 = ChunkDocument(
        chunk_id="scripture:GEN:1:1:0",
        text="In the beginning God created the heavens and the earth.",
        metadata=ChunkMetadata(
            chunk_id="scripture:GEN:1:1:0",
            source_type=SourceType.SCRIPTURE,
            book_code="GEN",
            book_name="Genesis",
            chapter=1,
            verse_start=1,
            verse_end=1,
            canonical_ref="Genesis 1:1",
            section_heading="The Creation",
        ),
    )
    chunk2 = ChunkDocument(
        chunk_id="outline:GEN:0",
        text="Genesis Outline: The Creation of the Heavens and the Earth (1:1 - 2:3)",
        metadata=ChunkMetadata(
            chunk_id="outline:GEN:0",
            source_type=SourceType.BOOK_OUTLINE,
            book_code="GEN",
            book_name="Genesis",
            canonical_ref="BSB Book Outline: Genesis",
            outline_path="Genesis > Creation",
        ),
    )
    return [
        RetrievalResult(chunk=chunk1, score=0.95, rerank_score=0.98, rank=1),
        RetrievalResult(chunk=chunk2, score=0.85, rerank_score=0.89, rank=2),
    ]


class MockPlan:
    def __init__(self, intent: QueryIntent = QueryIntent.EXACT_PASSAGE) -> None:
        self.intent = intent


class MockRetriever:
    """Deterministic in-memory retriever for tests."""

    def __init__(
        self,
        results: list[RetrievalResult],
        intent: QueryIntent = QueryIntent.EXACT_PASSAGE,
    ) -> None:
        self._results = results
        self._intent = intent

    def retrieve(
        self, query: str, top_k: int = 6
    ) -> tuple[MockPlan, list[RetrievalResult]]:
        return MockPlan(intent=self._intent), self._results[:top_k]


# ==============================================================================
# 1. LLM Factory Tests
# ==============================================================================


def test_is_reasoning_model() -> None:
    assert not is_reasoning_model("gpt-4o-mini")
    assert not is_reasoning_model("claude-3-5-haiku-latest")
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("gpt-6.1-sol")
    assert is_reasoning_model("gpt-6-astra")
    assert is_reasoning_model("o4-mini")
    assert is_reasoning_model("o1-preview")


def test_create_chat_model_openai_default() -> None:
    settings = Settings(openai_api_key="test-key-openai")
    model = create_chat_model(provider="openai", settings=settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0.0


def test_create_chat_model_openai_reasoning() -> None:
    settings = Settings(openai_api_key="test-key-openai")
    model = create_chat_model(
        provider="openai",
        model_name="o4-mini",
        effort="minimal",
        settings=settings,
    )
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "o4-mini"
    assert getattr(model, "reasoning_effort", None) == "minimal"

    model_gpt5 = create_chat_model(
        provider="openai",
        model_name="gpt-5-mini",
        effort="low",
        settings=settings,
    )
    assert isinstance(model_gpt5, ChatOpenAI)
    assert model_gpt5.model_name == "gpt-5-mini"
    assert getattr(model_gpt5, "reasoning_effort", None) == "low"


def test_create_chat_model_anthropic() -> None:
    settings = Settings(anthropic_api_key="test-key-anthropic")
    model = create_chat_model(
        provider="anthropic", effort="minimal", settings=settings
    )
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-3-5-haiku-latest"
    assert model.temperature == 0.0
    assert model.max_tokens == 1024

    model_high = create_chat_model(
        provider="anthropic", effort="high", settings=settings
    )
    assert isinstance(model_high, ChatAnthropic)
    assert model_high.max_tokens == 4096


def test_create_chat_model_invalid_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_chat_model(provider="unsupported_provider")  # type: ignore[arg-type]


# ==============================================================================
# 2. Prompts, Formatting & RAG Pipeline Tests
# ==============================================================================


def test_prompt_template_structure() -> None:
    assert "Berean Standard Bible" in SYSTEM_PROMPT
    assert "Strict Grounding & Citation Rules" in SYSTEM_PROMPT
    rendered = BSB_RAG_PROMPT.format(
        context="Sample context", question="What was created?"
    )
    assert "Sample context" in rendered
    assert "What was created?" in rendered


def test_format_context_blocks(sample_results: list[RetrievalResult]) -> None:
    formatted = format_context_blocks(sample_results)
    assert "=== SOURCE [1] | Reference: Genesis 1:1 | Type: scripture ===" in formatted
    assert "In the beginning God created" in formatted
    assert "=== SOURCE [2] | Reference: BSB Book Outline: Genesis | Type: book_outline ===" in formatted

    empty_formatted = format_context_blocks([])
    assert "No relevant context found" in empty_formatted


def test_build_citations(sample_results: list[RetrievalResult]) -> None:
    citations = build_citations(sample_results)
    assert len(citations) == 2
    assert citations[0].canonical_ref == "Genesis 1:1"
    assert citations[0].source_type == SourceType.SCRIPTURE
    assert citations[0].score == 0.98
    assert citations[1].canonical_ref == "BSB Book Outline: Genesis"
    assert citations[1].outline_path == "Genesis > Creation"


def test_rag_pipeline_ask(sample_results: list[RetrievalResult]) -> None:
    fake_answer = "In the beginning God created the heavens and the earth [Genesis 1:1]."
    fake_llm = FakeListChatModel(responses=[fake_answer])
    retriever = MockRetriever(sample_results, intent=QueryIntent.EXACT_PASSAGE)

    pipeline = BSBRAGPipeline(
        retriever=retriever,
        llm=fake_llm,
        provider="openai",
        model_name="gpt-4o-mini",
        effort="minimal",
    )

    response = pipeline.ask("What does Genesis 1:1 say?", top_k=2)
    assert response.query == "What does Genesis 1:1 say?"
    assert response.answer == fake_answer
    assert response.intent == QueryIntent.EXACT_PASSAGE
    assert response.provider == "openai"
    assert len(response.citations) == 2
    assert response.citations[0].canonical_ref == "Genesis 1:1"
    assert len(response.retrieved_chunks) == 2


def test_rag_pipeline_aask(sample_results: list[RetrievalResult]) -> None:
    import asyncio

    fake_answer = "God created the universe [Genesis 1:1]."
    fake_llm = FakeListChatModel(responses=[fake_answer])
    retriever = MockRetriever(sample_results)

    pipeline = BSBRAGPipeline(
        retriever=retriever,
        llm=fake_llm,
        provider="anthropic",
        model_name="claude-3-5-haiku-latest",
    )

    response = asyncio.run(pipeline.aask("Tell me about creation", top_k=1))
    assert response.answer == fake_answer
    assert len(response.citations) == 1
    assert response.citations[0].canonical_ref == "Genesis 1:1"


# ==============================================================================
# 3. FastAPI Endpoints Tests
# ==============================================================================


def test_fastapi_endpoints(sample_results: list[RetrievalResult]) -> None:
    retriever = MockRetriever(sample_results, intent=QueryIntent.EXACT_PASSAGE)
    fake_answer = "Creation began in Genesis 1:1 [Genesis 1:1]."
    fake_llm = FakeListChatModel(responses=[fake_answer])

    def mock_pipeline_factory(**kwargs: object) -> BSBRAGPipeline:
        return BSBRAGPipeline(retriever=retriever, llm=fake_llm)

    fastapi_app.dependency_overrides[get_api_retriever] = lambda: retriever
    fastapi_app.dependency_overrides[get_pipeline_factory] = (
        lambda: mock_pipeline_factory
    )

    client = TestClient(fastapi_app)

    # 1. GET /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok", "collection": "bsb_rag"}

    # 2. POST /retrieve
    res_retrieve = client.post(
        "/retrieve", json={"query": "Genesis 1:1", "top_k": 2}
    )
    assert res_retrieve.status_code == 200
    data_retrieve = res_retrieve.json()
    assert len(data_retrieve) == 2
    assert (
        data_retrieve[0]["chunk"]["metadata"]["canonical_ref"] == "Genesis 1:1"
    )

    # 3. POST /query
    res_query = client.post(
        "/query",
        json={
            "query": "What happened in the beginning?",
            "top_k": 2,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "effort": "minimal",
        },
    )
    assert res_query.status_code == 200
    data_query = res_query.json()
    assert data_query["query"] == "What happened in the beginning?"
    assert data_query["answer"] == fake_answer
    assert len(data_query["citations"]) == 2
    assert data_query["citations"][0]["canonical_ref"] == "Genesis 1:1"

    fastapi_app.dependency_overrides.clear()


# ==============================================================================
# 4. Typer CLI Tests
# ==============================================================================


def test_typer_cli_retrieve(sample_results: list[RetrievalResult]) -> None:
    retriever = MockRetriever(sample_results)
    set_custom_retriever(retriever)

    runner = CliRunner()
    result = runner.invoke(typer_app, ["retrieve", "Genesis 1:1", "--top-k", "2"])

    assert result.exit_code == 0
    assert "Genesis 1:1" in result.output
    assert "Retrieved BSB Chunks" in result.output
    assert "The Creation" in result.output

    set_custom_retriever(None)


def test_typer_cli_query(sample_results: list[RetrievalResult]) -> None:
    retriever = MockRetriever(sample_results)
    fake_answer = "Scripture states God created the heavens [Genesis 1:1]."
    fake_llm = FakeListChatModel(responses=[fake_answer])

    def mock_factory(**kwargs: object) -> BSBRAGPipeline:
        return BSBRAGPipeline(retriever=retriever, llm=fake_llm)

    set_custom_retriever(retriever)
    set_custom_pipeline_factory(mock_factory)

    runner = CliRunner()
    result = runner.invoke(
        typer_app,
        [
            "query",
            "What does the Bible say about creation?",
            "--provider",
            "openai",
            "--effort",
            "minimal",
            "--top-k",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert fake_answer in result.output
    assert "Citations & Grounding Sources" in result.output
    assert "Genesis 1:1" in result.output

    set_custom_retriever(None)
    set_custom_pipeline_factory(None)
