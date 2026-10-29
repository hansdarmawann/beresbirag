"""Unit and AppTest tests for Streamlit UI (TASK-10)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from beresbirag.cli import app as cli_app
from beresbirag.config import Settings
from beresbirag.llm.factory import DEFAULT_MODELS, MissingAPIKeyError
from beresbirag.llm.rag_chain import PreparedQuery
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    CitationItem,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import QueryPlan
from beresbirag.ui.services import (
    model_options,
    provider_options,
    set_service_factory,
)

APP_PATH = str(
    Path(__file__).resolve().parent.parent / "src" / "beresbirag" / "ui" / "streamlit_app.py"
)


def _create_app_test() -> AppTest:
    """Create AppTest instance with sufficient timeout for cold-start module imports."""
    return AppTest.from_file(APP_PATH, default_timeout=25)


def _make_sample_chunks() -> list[RetrievalResult]:
    c1 = ChunkDocument(
        chunk_id="scripture:JHN:3:16-18:0",
        text="[John 3:16] For God so loved the world that He gave His one and only Son...",
        metadata=ChunkMetadata(
            chunk_id="scripture:JHN:3:16-18:0",
            source_type=SourceType.SCRIPTURE,
            canonical_ref="John 3:16-18",
            book_code="JHN",
            chapter=3,
            verse_start=16,
            verse_end=18,
        ),
    )
    c2 = ChunkDocument(
        chunk_id="topical:grace:0",
        text="Key Scripture Passages for Topic 'grace'...",
        metadata=ChunkMetadata(
            chunk_id="topical:grace:0",
            source_type=SourceType.TOPICAL_INDEX,
            canonical_ref="Topical Index: Grace",
            topic="grace",
        ),
    )
    return [
        RetrievalResult(chunk=c1, score=0.95, rank=1, rerank_score=0.98),
        RetrievalResult(chunk=c2, score=0.80, rank=2, rerank_score=0.85),
    ]


class FakeRetriever:
    def __init__(
        self, results: list[RetrievalResult] | None = None, should_fail: bool = False
    ) -> None:
        self.results = results if results is not None else _make_sample_chunks()
        self.should_fail = should_fail
        self.called_with: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> tuple[QueryPlan, list[RetrievalResult]]:
        if self.should_fail:
            raise ConnectionError("Failed to connect to Qdrant at http://localhost:6333")
        self.called_with.append((query, top_k))
        plan = QueryPlan(
            original_query=query,
            intent=QueryIntent.EXACT_PASSAGE if "3:16" in query else QueryIntent.HYBRID_GENERAL,
        )
        return plan, self.results[:top_k]


class FakePipeline:
    def __init__(
        self,
        answer_mode: AnswerMode = AnswerMode.LLM,
        fast_answer: str | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.answer_mode = answer_mode
        self.fast_answer = fast_answer
        self.stream_chunks = stream_chunks or ["God ", "so ", "loved ", "the ", "world."]
        self.prepare_called = False
        self.stream_called = False
        self.finalize_called = False
        self.retriever = FakeRetriever()

    def prepare(self, query: str, top_k: int | None = None) -> PreparedQuery:
        self.prepare_called = True
        plan, results = self.retriever.retrieve(query, top_k or 5)
        return PreparedQuery(
            query=query,
            plan=plan,
            results=results,
            context="[John 3:16] For God so loved the world...",
            fast_answer=self.fast_answer,
            answer_mode=self.answer_mode,
        )

    def stream(self, prepared: PreparedQuery) -> Iterator[str]:
        self.stream_called = True
        yield from self.stream_chunks

    def finalize(self, prepared: PreparedQuery, answer: str) -> RAGResponse:
        self.finalize_called = True
        citations = [
            CitationItem(
                chunk_id=r.chunk.chunk_id,
                source_type=r.chunk.metadata.source_type,
                canonical_ref=r.chunk.metadata.canonical_ref or r.chunk.chunk_id,
                score=r.rerank_score or r.score,
            )
            for r in prepared.results
        ]
        return RAGResponse(
            query=prepared.query,
            answer=answer,
            intent=prepared.plan.intent if prepared.plan else QueryIntent.HYBRID_GENERAL,
            provider="openai",
            model="gpt-4o-mini",
            citations=citations,
            retrieved_chunks=prepared.results,
            answer_mode=prepared.answer_mode,
        )


class FakeServiceFactory:
    def __init__(
        self,
        retriever: FakeRetriever | None = None,
        pipeline: FakePipeline | None = None,
        exc_on_pipeline: Exception | None = None,
    ) -> None:
        self._retriever = retriever or FakeRetriever()
        self._pipeline = pipeline or FakePipeline()
        self.exc_on_pipeline = exc_on_pipeline
        self.pipeline_called = False

    def retriever(self) -> Any:
        return self._retriever

    def pipeline(self, provider: str, model: str, effort: str | None = None) -> Any:
        self.pipeline_called = True
        if self.exc_on_pipeline is not None:
            raise self.exc_on_pipeline
        return self._pipeline


@pytest.fixture(autouse=True)
def mock_diagnostics_and_clean_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from beresbirag.config import get_settings
    from beresbirag.diagnostics import CheckResult

    get_settings.cache_clear()
    fake_checks = [
        CheckResult("RAM", "pass", "available"),
        CheckResult("CPU cores", "pass", "2 cores"),
        CheckResult("AVX2", "pass", "supported"),
        CheckResult("Qdrant", "pass", "test: 100 points"),
    ]
    monkeypatch.setattr(
        "beresbirag.diagnostics.run_diagnostics",
        lambda settings: fake_checks,
    )
    yield
    set_service_factory(None)
    get_settings.cache_clear()


def test_model_options_provider_safety() -> None:
    s_openai = Settings(llm_provider="openai", llm_model="gpt-4o-mini")
    openai_models = model_options("openai", s_openai)
    assert not any(m.startswith("claude-") for m in openai_models)
    assert DEFAULT_MODELS["openai"] in openai_models

    s_anthropic = Settings(llm_provider="anthropic", llm_model="claude-haiku-5-5")
    anthropic_models = model_options("anthropic", s_anthropic)
    assert not any(m.startswith("gpt-") for m in anthropic_models)
    assert DEFAULT_MODELS["anthropic"] in anthropic_models

    s_ollama = Settings(llm_provider="ollama", llm_model="qwen2.5:1.5b")
    installed = ["qwen2.5:0.5b", "phi4-mini:latest"]
    ollama_models = model_options("ollama", s_ollama, installed)
    assert "phi4-mini:latest" in ollama_models
    assert "qwen2.5:0.5b" in ollama_models
    assert DEFAULT_MODELS["ollama"] in ollama_models
    assert not any(m.startswith("claude-") for m in ollama_models)


def test_provider_options_offline_mode() -> None:
    s_online = Settings(offline_mode=False, llm_provider="openai")
    provs = provider_options(s_online)
    assert provs == ["openai", "anthropic", "ollama"]

    s_offline = Settings(offline_mode=True, llm_provider="ollama")
    assert provider_options(s_offline) == ["ollama"]


def test_ui_retrieve_only_mode() -> None:
    fake_retriever = FakeRetriever()
    fake_factory = FakeServiceFactory(retriever=fake_retriever)
    set_service_factory(fake_factory)

    app = _create_app_test()
    app.run()

    # Select Retrieve only
    app.radio[0].set_value("Retrieve only (no LLM)")
    # Enter question
    app.text_area[0].input("John 3:16")
    # Click Ask
    app.button[4].click()
    app.run()

    assert not app.exception
    assert fake_retriever.called_with
    assert not fake_factory.pipeline_called

    # Check expanders count equals retrieved chunks count
    assert len(app.expander) >= 2
    # Verify intent badge rendered
    assert any("exact_passage" in m.value for m in app.markdown)


def test_ui_fast_path_mode() -> None:
    fake_pipe = FakePipeline(
        answer_mode=AnswerMode.FAST_PATH,
        fast_answer="[John 3:16] For God so loved the world...",
    )
    fake_factory = FakeServiceFactory(pipeline=fake_pipe)
    set_service_factory(fake_factory)

    app = _create_app_test()
    app.run()

    app.radio[0].set_value("Answer with LLM")
    app.text_area[0].input("John 3:16")
    app.button[4].click()
    app.run()

    assert not app.exception
    assert fake_pipe.prepare_called
    # Fast path: stream should NOT be called
    assert not fake_pipe.stream_called
    # Verify caption
    assert any("without an LLM" in c.value for c in app.caption)


def test_ui_llm_streaming_mode() -> None:
    fake_pipe = FakePipeline(
        answer_mode=AnswerMode.LLM,
        stream_chunks=["Streamed ", "biblical ", "answer."],
    )
    fake_factory = FakeServiceFactory(pipeline=fake_pipe)
    set_service_factory(fake_factory)

    app = _create_app_test()
    app.run()

    app.radio[0].set_value("Answer with LLM")
    app.text_area[0].input("What is grace?")
    app.button[4].click()
    app.run()

    assert not app.exception
    assert fake_pipe.prepare_called
    assert fake_pipe.stream_called
    assert fake_pipe.finalize_called
    # Check that citations dataframe is rendered
    assert len(app.dataframe) >= 1


def test_ui_missing_api_key_error() -> None:
    fake_factory = FakeServiceFactory(
        exc_on_pipeline=MissingAPIKeyError("openai API key missing: set OPENAI_API_KEY in .env")
    )
    set_service_factory(fake_factory)

    app = _create_app_test()
    app.run()

    app.radio[0].set_value("Answer with LLM")
    app.text_area[0].input("Tell me about faith")
    app.button[4].click()
    app.run()

    assert len(app.error) >= 1
    assert "OPENAI_API_KEY" in app.error[0].value


def test_ui_connection_error_hints_doctor() -> None:
    fake_retriever = FakeRetriever(should_fail=True)
    fake_factory = FakeServiceFactory(retriever=fake_retriever)
    set_service_factory(fake_factory)

    app = _create_app_test()
    app.run()

    app.radio[0].set_value("Retrieve only (no LLM)")
    app.text_area[0].input("Genesis 1:1")
    app.button[4].click()
    app.run()

    assert len(app.error) >= 1
    assert "beresbirag doctor" in app.error[0].value


def test_ui_low_resource_profile_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOURCE_PROFILE", "low")
    set_service_factory(FakeServiceFactory())

    app = _create_app_test()
    app.run()

    # Radio should default to Retrieve only (no LLM)
    assert app.radio[0].value == "Retrieve only (no LLM)"
    # Top K slider should default to low profile retrieval_top_k (3)
    assert app.slider[0].value == 3


def test_cli_ui_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    result = runner.invoke(cli_app, ["ui", "--port", "8600"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    assert "--server.port" in called_cmd
    assert "8600" in called_cmd
    assert "--browser.gatherUsageStats" in called_cmd
    assert "false" in called_cmd


def test_cli_ui_launcher_streamlit_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    with patch.dict(sys.modules, {"streamlit": None}):
        result = runner.invoke(cli_app, ["ui"])
        assert result.exit_code == 1
        assert "Streamlit is not installed" in result.output
        assert "pip install" in result.output
