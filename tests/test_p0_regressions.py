"""Regression tests for TASK-08 P0 correctness fixes (no real LLM, remote Qdrant, or network)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from qdrant_client import QdrantClient, models
from typer.testing import CliRunner

import beresbirag.cli as cli_mod
import beresbirag.config as config_mod
import beresbirag.ingestion.pipeline as pipeline_mod
from beresbirag.api import app as fastapi_app
from beresbirag.api import get_api_retriever, get_pipeline_factory
from beresbirag.config import Settings
from beresbirag.ingestion.pipeline import IngestionPipeline
from beresbirag.llm import (
    DEFAULT_MODELS,
    INSUFFICIENT_CONTEXT_ANSWER,
    BSBRAGPipeline,
    MissingAPIKeyError,
    ProviderModelMismatchError,
    create_chat_model,
    resolve_llm_target,
)
from beresbirag.llm.prompts import SYSTEM_PROMPT
from beresbirag.models import AnswerMode
from beresbirag.models.schemas import (
    ChunkDocument,
    ChunkMetadata,
    QueryIntent,
    RetrievalResult,
    SourceType,
)
from beresbirag.vectorstore import CollectionConfigError
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantManager,
)

LLM_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_EFFORT",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "EMBEDDING_PROVIDER",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell environment from leaking into Settings."""
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _sample_result() -> RetrievalResult:
    chunk = ChunkDocument(
        chunk_id="scripture:GEN:1:1:0",
        text="In the beginning God created the heavens and the earth.",
        metadata=ChunkMetadata(
            chunk_id="scripture:GEN:1:1:0",
            source_type=SourceType.SCRIPTURE,
            book_code="GEN",
            canonical_ref="Genesis 1:1",
            verse_refs=["GEN 1:1"],
        ),
    )
    return RetrievalResult(chunk=chunk, score=0.9, rank=1)


class FakeRetriever:
    """Deterministic retriever returning a fixed (plan, results) pair."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int = 6) -> tuple[Any, list[RetrievalResult]]:
        return SimpleNamespace(intent=QueryIntent.HYBRID_GENERAL), self._results[:top_k]


class SpyChatModel(SimpleChatModel):
    """Chat model that records every invocation."""

    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "spy-chat"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        self.calls += 1
        return "Spy answer [Genesis 1:1]."


def _manager(
    client: QdrantClient, dense_size: int, collection: str = "test_p0_rag"
) -> QdrantManager:
    return QdrantManager(
        settings=_settings(QDRANT_MODE="memory", QDRANT_COLLECTION_NAME=collection),
        client=client,
        dense_embeddings=DeterministicHashEmbeddings(size=dense_size),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=dense_size,
    )


# ==============================================================================
# 1. Provider / model resolution
# ==============================================================================


def test_pipeline_anthropic_uses_anthropic_default_model() -> None:
    settings = _settings(ANTHROPIC_API_KEY="test-key-anthropic")
    assert settings.llm_provider == "openai"
    assert settings.llm_model is None

    pipeline = BSBRAGPipeline(retriever=FakeRetriever([]), provider="anthropic", settings=settings)

    assert isinstance(pipeline.llm, ChatAnthropic)
    assert pipeline.llm.model == "claude-haiku-5-5"
    assert (pipeline.provider, pipeline.model_name) == ("anthropic", "claude-haiku-5-5")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "claude-haiku-5-5"),
        ("anthropic", "gpt-4o-mini"),
        ("anthropic", "o4-mini"),
        ("anthropic", "chatgpt-4o-latest"),
    ],
)
def test_resolve_llm_target_rejects_foreign_model(provider: str, model: str) -> None:
    with pytest.raises(ProviderModelMismatchError, match=model):
        resolve_llm_target(provider, model, _settings())


def test_resolve_llm_target_uses_llm_model_for_matching_provider() -> None:
    settings = _settings(LLM_PROVIDER="anthropic", LLM_MODEL="claude-haiku-5-5")
    assert resolve_llm_target(None, None, settings) == ("anthropic", "claude-haiku-5-5")


def test_resolve_llm_target_ignores_llm_model_for_other_provider() -> None:
    settings = _settings(LLM_PROVIDER="openai", LLM_MODEL="gpt-4.1-mini")
    assert resolve_llm_target("anthropic", None, settings) == (
        "anthropic",
        DEFAULT_MODELS["anthropic"],
    )
    assert resolve_llm_target("openai", None, settings) == ("openai", "gpt-4.1-mini")
    # An explicit model always wins.
    assert resolve_llm_target(None, "gpt-5-mini", settings) == ("openai", "gpt-5-mini")


def test_resolve_llm_target_empty_llm_model_falls_back_to_default() -> None:
    settings = _settings(LLM_MODEL="")
    assert resolve_llm_target(None, None, settings) == ("openai", DEFAULT_MODELS["openai"])


def test_resolve_llm_target_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        resolve_llm_target("mistral", None, _settings())


# ==============================================================================
# 2. Fail fast on missing API keys
# ==============================================================================


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
def test_create_chat_model_missing_key_raises(provider: str, env_var: str) -> None:
    with pytest.raises(MissingAPIKeyError, match=env_var):
        create_chat_model(provider=provider, settings=_settings())  # type: ignore[arg-type]


def test_create_chat_model_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-anthropic")
    model = create_chat_model(provider="anthropic", settings=settings)
    assert isinstance(model, ChatAnthropic)
    assert model.anthropic_api_key.get_secret_value() == "env-key-anthropic"


# ==============================================================================
# 3. Empty context never calls the LLM
# ==============================================================================


def test_insufficient_context_answer_matches_system_prompt() -> None:
    assert INSUFFICIENT_CONTEXT_ANSWER in SYSTEM_PROMPT


def test_empty_context_skips_llm_sync_and_async() -> None:
    spy = SpyChatModel()
    pipeline = BSBRAGPipeline(retriever=FakeRetriever([]), llm=spy, settings=_settings())

    for response in (
        pipeline.ask("Who was Melchizedek's mother?"),
        asyncio.run(pipeline.aask("Who was Melchizedek's mother?")),
    ):
        assert response.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT
        assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
        assert response.citations == []
        assert response.retrieved_chunks == []

    assert spy.calls == 0


def test_non_empty_context_calls_llm() -> None:
    spy = SpyChatModel()
    pipeline = BSBRAGPipeline(
        retriever=FakeRetriever([_sample_result()]), llm=spy, settings=_settings()
    )
    response = pipeline.ask("What does Genesis 1:1 say?")
    assert response.answer_mode == AnswerMode.LLM
    assert spy.calls == 1


# ==============================================================================
# 4. BM25 IDF modifier & 5. collection validation
# ==============================================================================


def test_sparse_vector_uses_idf_modifier() -> None:
    manager = _manager(QdrantClient(location=":memory:"), dense_size=64)
    manager.ensure_collection(recreate=True)
    info = manager.client.get_collection(manager.collection_name)
    assert info.config.params.sparse_vectors is not None
    assert info.config.params.sparse_vectors[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


def test_dense_dimension_mismatch_raises_collection_config_error() -> None:
    client = QdrantClient(location=":memory:")
    _manager(client, dense_size=384).ensure_collection(recreate=True)

    mismatched = _manager(client, dense_size=1536)
    with pytest.raises(CollectionConfigError, match="--recreate") as excinfo:
        mismatched.hybrid_search("In the beginning", k=3)
    assert "384" in str(excinfo.value)
    assert "1536" in str(excinfo.value)

    flt = mismatched.build_filter(verse_refs=["GEN 1:1"])
    assert flt is not None
    with pytest.raises(CollectionConfigError, match="--recreate"):
        _manager(client, dense_size=1536).scroll_by_filter(flt)


def test_legacy_collection_without_idf_raises() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="test_p0_rag",
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=64, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    with pytest.raises(CollectionConfigError, match="idf") as excinfo:
        _manager(client, dense_size=64).validate_collection()
    assert str(excinfo.value).endswith(
        "Re-run: conda run -n beresbirag python ./scripts/ingest_all.py --recreate"
    )


def test_validate_collection_noop_when_missing_and_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(location=":memory:")
    _manager(client, dense_size=64).validate_collection()  # no collection: no-op

    _manager(client, dense_size=64).ensure_collection(recreate=True)
    reader = _manager(client, dense_size=64)
    calls: list[None] = []
    original = reader.validate_collection

    def counting_validate() -> None:
        calls.append(None)
        original()

    monkeypatch.setattr(reader, "validate_collection", counting_validate)
    reader.hybrid_search("In the beginning", k=3)
    reader.hybrid_search("Let there be light", k=3)
    flt = reader.build_filter(verse_refs=["GEN 1:1"])
    assert flt is not None
    reader.scroll_by_filter(flt)
    assert len(calls) == 1


# ==============================================================================
# 6. Embedding provider default
# ==============================================================================


def test_embedding_default_is_fastembed_even_with_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    settings = _settings()
    assert settings.embedding_provider == "fastembed"
    assert settings.resolved_embedding_provider == "fastembed"
    assert settings.dense_vector_size == 384


def test_auto_embedding_provider_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config_mod, "_auto_embedding_warning_emitted", False)
    settings = _settings(EMBEDDING_PROVIDER="auto", OPENAI_API_KEY="sk-test-123")
    with caplog.at_level(logging.WARNING, logger="beresbirag.config"):
        assert settings.resolved_embedding_provider == "openai"
        assert settings.resolved_embedding_provider == "openai"
    warnings = [r for r in caplog.records if "EMBEDDING_PROVIDER=auto" in r.getMessage()]
    assert len(warnings) == 1


# ==============================================================================
# 7. Ingestion honors injected settings
# ==============================================================================


@pytest.fixture
def synthetic_corpus_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "usj": tmp_path / "synthetic_usj.zip",
        "pdf": tmp_path / "synthetic_outlines.pdf",
        "topical": tmp_path / "synthetic_topical.xlsx",
        "concordance": tmp_path / "synthetic_concordance.xlsx",
    }
    for path in paths.values():
        path.write_bytes(b"")
    return paths


@pytest.fixture
def captured_parser_paths(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    fake_index = SimpleNamespace(to_chunk_documents=lambda: [])

    def fake_from_pdf(pdf_path: Any = None) -> Any:
        captured["pdf"] = pdf_path
        return fake_index

    def fake_usj(*, zip_path: Any = None, **_: Any) -> Iterator[ChunkDocument]:
        captured["usj"] = zip_path
        return iter([])

    def fake_topical(*, xlsx_path: Any = None, **_: Any) -> Iterator[ChunkDocument]:
        captured["topical"] = xlsx_path
        return iter([])

    def fake_concordance(*, xlsx_path: Any = None, **_: Any) -> Iterator[ChunkDocument]:
        captured["concordance"] = xlsx_path
        return iter([])

    monkeypatch.setattr(pipeline_mod.BookOutlineIndex, "from_pdf", fake_from_pdf)
    monkeypatch.setattr(pipeline_mod, "iter_usj_scripture_chunks", fake_usj)
    monkeypatch.setattr(pipeline_mod, "iter_topical_chunks", fake_topical)
    monkeypatch.setattr(pipeline_mod, "iter_concordance_chunks", fake_concordance)
    return captured


def test_ingestion_uses_injected_settings_paths(
    synthetic_corpus_paths: dict[str, Path], captured_parser_paths: dict[str, Any]
) -> None:
    settings = _settings(
        QDRANT_MODE="memory",
        BSB_USJ_ZIP_PATH=str(synthetic_corpus_paths["usj"]),
        BSB_OUTLINES_PDF_PATH=str(synthetic_corpus_paths["pdf"]),
        BSB_TOPICAL_XLSX_PATH=str(synthetic_corpus_paths["topical"]),
        BSB_CONCORDANCE_XLSX_PATH=str(synthetic_corpus_paths["concordance"]),
    )
    pipeline = IngestionPipeline(settings=settings)
    assert pipeline.collect_chunks() == []

    for key, path in synthetic_corpus_paths.items():
        assert Path(captured_parser_paths[key]) == path.resolve(), key


def test_ingestion_explicit_path_overrides_settings(
    synthetic_corpus_paths: dict[str, Path], captured_parser_paths: dict[str, Any]
) -> None:
    settings = _settings(
        QDRANT_MODE="memory", BSB_OUTLINES_PDF_PATH=str(synthetic_corpus_paths["pdf"])
    )
    explicit = synthetic_corpus_paths["usj"].with_name("explicit_outlines.pdf")
    pipeline = IngestionPipeline(settings=settings, outlines_pdf_path=explicit)
    pipeline.load_outline_index()
    assert captured_parser_paths["pdf"] == explicit


# ==============================================================================
# 8. CLI & API error surfaces
# ==============================================================================


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Inject settings and an empty-context retriever into the CLI, recording pipelines."""
    env: dict[str, Any] = {"pipelines": []}

    class RecordingPipeline(BSBRAGPipeline):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            env["pipelines"].append(self)

    def use_settings(settings: Settings) -> None:
        monkeypatch.setattr(cli_mod, "get_settings", lambda: settings)

    env["use_settings"] = use_settings
    monkeypatch.setattr(cli_mod, "BSBRAGPipeline", RecordingPipeline)
    cli_mod.set_custom_pipeline_factory(None)
    cli_mod.set_custom_retriever(FakeRetriever([]))
    try:
        yield env
    finally:
        cli_mod.set_custom_retriever(None)


def test_cli_query_defaults_to_settings_provider(cli_env: dict[str, Any]) -> None:
    cli_env["use_settings"](
        _settings(LLM_PROVIDER="anthropic", LLM_EFFORT="low", ANTHROPIC_API_KEY="test-key")
    )
    result = CliRunner().invoke(cli_mod.app, ["query", "Who was Melchizedek's mother?"])

    assert result.exit_code == 0, result.output
    (pipeline,) = cli_env["pipelines"]
    assert (pipeline.provider, pipeline.model_name, pipeline.effort) == (
        "anthropic",
        "claude-haiku-5-5",
        "low",
    )
    assert isinstance(pipeline.llm, ChatAnthropic)
    assert "anthropic / claude-haiku-5-5" in result.output


def test_cli_query_missing_key_exits_2(cli_env: dict[str, Any]) -> None:
    cli_env["use_settings"](_settings(LLM_PROVIDER="openai"))
    result = CliRunner().invoke(cli_mod.app, ["query", "What does Genesis 1:1 say?"])
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_cli_query_provider_model_mismatch_exits_2(cli_env: dict[str, Any]) -> None:
    cli_env["use_settings"](_settings(OPENAI_API_KEY="test-key"))
    result = CliRunner().invoke(
        cli_mod.app,
        [
            "query",
            "What does Genesis 1:1 say?",
            "--provider",
            "openai",
            "--model",
            "claude-haiku-5-5",
        ],
    )
    assert result.exit_code == 2


def test_cli_retrieve_collection_config_error_exits_2(cli_env: dict[str, Any]) -> None:
    class BrokenRetriever:
        def retrieve(self, query: str, top_k: int = 6) -> Any:
            raise CollectionConfigError("dimension mismatch. Re-run: ... --recreate")

    cli_mod.set_custom_retriever(BrokenRetriever())
    result = CliRunner().invoke(cli_mod.app, ["retrieve", "Genesis 1:1"])
    assert result.exit_code == 2
    assert "--recreate" in result.output


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    try:
        yield TestClient(fastapi_app)
    finally:
        fastapi_app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("settings_kwargs", "body", "status"),
    [
        ({}, {"provider": "openai"}, 503),
        ({"OPENAI_API_KEY": "test-key"}, {"provider": "openai", "model": "claude-haiku-5-5"}, 422),
    ],
)
def test_api_query_config_errors(
    api_client: TestClient, settings_kwargs: dict[str, str], body: dict[str, str], status: int
) -> None:
    settings = _settings(**settings_kwargs)
    retriever = FakeRetriever([_sample_result()])

    def factory(**kwargs: Any) -> BSBRAGPipeline:
        return BSBRAGPipeline(retriever=retriever, settings=settings, **kwargs)

    fastapi_app.dependency_overrides[get_pipeline_factory] = lambda: factory
    res = api_client.post("/query", json={"query": "What does Genesis 1:1 say?", **body})
    assert res.status_code == status


def test_api_collection_config_error_is_503(api_client: TestClient) -> None:
    class BrokenRetriever:
        def retrieve(self, query: str, top_k: int = 6) -> Any:
            raise CollectionConfigError("dimension mismatch. Re-run: ... --recreate")

    fastapi_app.dependency_overrides[get_api_retriever] = lambda: BrokenRetriever()
    res = api_client.post("/retrieve", json={"query": "Genesis 1:1"})
    assert res.status_code == 503
    assert "--recreate" in res.json()["detail"]
