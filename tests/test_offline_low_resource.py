"""Offline acceptance tests: fake models, transports and hardware probes only."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_ollama import ChatOllama
from typer.testing import CliRunner

import beresbirag.cli as cli
import beresbirag.diagnostics as diagnostics
from beresbirag.api import QueryRequest, RetrieveRequest
from beresbirag.config import Settings
from beresbirag.llm.context_budget import estimate_tokens, fit_context
from beresbirag.llm.factory import (
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
    create_chat_model,
    resolve_llm_target,
)
from beresbirag.llm.fast_path import try_fast_answer
from beresbirag.llm.prompts import (
    BSB_RAG_PROMPT,
    BSB_RAG_PROMPT_COMPACT,
    INSUFFICIENT_CONTEXT_ANSWER,
)
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import route_query
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import create_dense_embeddings, create_sparse_embeddings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name, field in Settings.model_fields.items():
        monkeypatch.delenv(field.alias or name.upper(), raising=False)
    # Record restoration even when a Settings validator creates this variable.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.delenv("FLASHRANK_CACHE_DIR", raising=False)


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def result(text, source=SourceType.SCRIPTURE, **metadata):
    return RetrievalResult(
        chunk=ChunkDocument(
            chunk_id="sample",
            text=text,
            metadata=ChunkMetadata(chunk_id="sample", source_type=source, **metadata),
        )
    )


class Retriever:
    def __init__(self, results):
        self.results = results
        self.top_k = None

    def retrieve(self, query, top_k=None):
        self.top_k = top_k
        return route_query(query), self.results


def test_profile_and_explicit_overrides(monkeypatch):
    cfg = settings(RESOURCE_PROFILE="low")
    assert (cfg.retrieval_top_k, cfg.retrieval_initial_k) == (3, 10)
    assert cfg.reranker_model == "ms-marco-TinyBERT-L-2-v2"
    assert (cfg.context_token_budget, cfg.context_chunk_token_cap) == (1600, 400)
    assert cfg.fast_path_enabled is True
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    cfg = settings(RESOURCE_PROFILE="low", FAST_PATH_ENABLED=False, LLM_NUM_THREAD=4)
    assert cfg.retrieval_top_k == 5
    assert cfg.fast_path_enabled is False
    assert cfg.llm_num_thread == 4


def test_standard_defaults():
    cfg = settings()
    assert cfg.retrieval_top_k == 6
    assert cfg.context_token_budget is None
    assert cfg.context_chunk_token_cap is None
    assert not cfg.fast_path_enabled


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_offline_blocks_cloud_and_forces_local_embeddings(provider, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = settings(OFFLINE_MODE=True, EMBEDDING_PROVIDER="openai", OPENAI_API_KEY="test")
    assert cfg.resolved_embedding_provider == "fastembed"
    with pytest.raises(ProviderUnavailableError, match="offline mode: use --provider ollama"):
        create_chat_model(provider=provider, settings=cfg)


def test_offline_sets_hub_flag(monkeypatch):
    settings(OFFLINE_MODE=True)
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"


def test_ollama_factory_and_resolution():
    cfg = settings(RESOURCE_PROFILE="low")
    model = create_chat_model(provider="ollama", settings=cfg)
    assert isinstance(model, ChatOllama)
    assert (model.num_ctx, model.num_thread, model.num_predict) == (3072, 2, 384)
    assert model.model == "qwen2.5:1.5b"
    assert model.temperature == 0.0 and model.reasoning is False
    assert model.keep_alive == "30m"
    assert resolve_llm_target("ollama", "custom:model", cfg) == ("ollama", "custom:model")
    with pytest.raises(ProviderModelMismatchError):
        create_chat_model(provider="ollama", model_name="claude-test", settings=cfg)


def test_budget_notes_cross_references_and_no_mutation():
    original = result(
        "[Book: Genesis | Cross-Refs: John 1:1]\n[Genesis 1:1] In the beginning.\nFootnotes:\n- Literally a note."
    )
    before = original.model_dump()
    trimmed = fit_context([original], "What does Genesis 1:1 say?", 400, 400)[0]
    assert "Footnotes:" not in trimmed.chunk.text
    assert "Cross-Refs:" not in trimmed.chunk.text
    assert trimmed.chunk.text.startswith("[Book: Genesis]\n")
    assert original.model_dump() == before
    assert trimmed is not original and trimmed.chunk is not original.chunk
    retained = fit_context(
        [original], "What footnotes and parallel references are in Genesis 1:5?", 400, 400
    )
    assert retained[0].chunk.text == original.chunk.text


def test_budget_line_boundaries_total_and_at_least_one():
    chunks = [result("Header\n[John 3:16] Whole verse.\n[John 3:17] Another whole verse.")] * 3
    fitted = fit_context(chunks, "love", 20, 9)
    assert len(fitted) == 2
    assert all(item.chunk.text == "Header\n[John 3:16] Whole verse." for item in fitted)
    assert all(estimate_tokens(item.chunk.text) <= 9 for item in fitted)
    assert len(fit_context(chunks, "love", 1, 1)) == 1
    assert fit_context(chunks, "love", None, None) is chunks
    assert fit_context([], "love", 10, 10) == []


@pytest.mark.parametrize(
    ("source", "extra"),
    [
        (SourceType.TOPICAL_INDEX, "Additional References: x\nAll Verse References: y"),
        (SourceType.CONCORDANCE, "Verse Citations: x"),
    ],
)
def test_budget_drops_reference_lists(source, extra):
    assert (
        fit_context([result("Header\nUseful text\n" + extra, source)], "query", 100, None)[
            0
        ].chunk.text
        == "Header\nUseful text"
    )


def test_fast_passage_deduplicates_and_orders():
    chunks = [result("[John 3:16] For God so loved the world.")] * 2
    answer = try_fast_answer(route_query("John 3:16"), chunks)
    assert answer.startswith("**John 3:16** (BSB)")
    assert answer.count("[John 3:16]") == 1
    chunks = [result("[Genesis 1:3] Third.\n[Genesis 1:1] First.\n[Genesis 1:2] Second.")]
    answer = try_fast_answer(route_query("Genesis 1:1-3"), chunks)
    assert answer.index("First.") < answer.index("Second.") < answer.index("Third.")


@pytest.mark.parametrize(
    "query",
    ["What does John 3:16 teach about love?", "John 3:16 explain", "John 3:16 apa artinya?"],
)
def test_interpretation_is_not_fast(query):
    assert try_fast_answer(route_query(query), [result("[John 3:16] Verse.")]) is None


def test_fast_incomplete_passage_falls_back():
    assert try_fast_answer(route_query("Genesis 1:1-3"), [result("[Genesis 1:1] First.")]) is None


def test_concordance_count():
    chunks = [
        result(
            "Word: love\nBook Distribution: Genesis (3), John (20)",
            SourceType.CONCORDANCE,
            concordance_word="love",
            concordance_occurrences=23,
        )
    ]
    answer = try_fast_answer(route_query("How many times does the word love occur?"), chunks)
    assert '"love" occurs 23 times in the BSB [BSB Concordance: love].' in answer
    assert "Book Distribution: Genesis (3), John (20)" in answer
    assert try_fast_answer(route_query("How many occurrences of faith?"), chunks) is None


@pytest.mark.parametrize(
    ("mode", "fast", "chunks"),
    [
        (AnswerMode.LLM, False, [result("[John 3:16] Verse.")]),
        (AnswerMode.FAST_PATH, True, [result("[John 3:16] Verse.")]),
        (AnswerMode.INSUFFICIENT_CONTEXT, True, []),
    ],
)
def test_streaming_sync_async_and_modes(mode, fast, chunks):
    pipeline = BSBRAGPipeline(
        Retriever(chunks),
        settings=settings(FAST_PATH_ENABLED=fast),
        llm=GenericFakeChatModel(messages=iter(["Answer."] * 5)),
    )
    prepared = pipeline.prepare("John 3:16")
    streamed = "".join(pipeline.stream(prepared))
    assert streamed == pipeline.ask("John 3:16").answer
    response = pipeline.finalize(prepared, streamed)
    assert response.answer_mode == mode
    assert asyncio.run(pipeline.aask("John 3:16")).answer == streamed
    if mode == AnswerMode.INSUFFICIENT_CONTEXT:
        assert response.answer == INSUFFICIENT_CONTEXT_ANSWER and response.citations == []


def test_lazy_model_and_low_profile_top_k():
    retriever = Retriever([result("[John 3:16] Verse.")])
    pipeline = BSBRAGPipeline(
        retriever, provider="openai", settings=settings(RESOURCE_PROFILE="low")
    )
    assert pipeline._llm is None
    assert pipeline.ask("John 3:16").answer_mode == AnswerMode.FAST_PATH
    assert pipeline._llm is None and retriever.top_k == 3
    with pytest.raises(MissingAPIKeyError):
        pipeline.ask("What does John 3:16 teach about love?")
    pipeline = BSBRAGPipeline(Retriever([]), settings=settings())
    assert pipeline.ask("No context").answer_mode == AnswerMode.INSUFFICIENT_CONTEXT
    assert pipeline._llm is None


def test_prompt_selection_and_ollama_complete_budget():
    retriever = Retriever(
        [result("Header\n" + "\n".join("A whole source line." for _ in range(30)))] * 3
    )
    cfg = settings(LLM_PROVIDER="ollama", LLM_NUM_CTX=260, LLM_NUM_PREDICT=50)
    pipeline = BSBRAGPipeline(retriever, settings=cfg)
    assert pipeline.prompt is BSB_RAG_PROMPT_COMPACT
    prepared = pipeline.prepare("Explain")
    assert (
        pipeline._prompt_tokens("Explain", prepared.results) + cfg.llm_num_predict
        <= cfg.llm_num_ctx
    )
    assert BSBRAGPipeline(retriever, settings=settings()).prompt is BSB_RAG_PROMPT
    assert BSBRAGPipeline(retriever, settings=cfg, prompt=BSB_RAG_PROMPT).prompt is BSB_RAG_PROMPT
    with pytest.raises(ValueError, match="exceed LLM_NUM_CTX"):
        pipeline.prepare("long " * 1000)


def test_embedding_load_options(monkeypatch, tmp_path):
    import fastembed

    import beresbirag.vectorstore.embeddings as embeddings

    dense, sparse = Mock(), Mock()
    monkeypatch.setattr(fastembed, "TextEmbedding", dense)
    monkeypatch.setattr(embeddings, "FastEmbedSparse", sparse)
    cfg = settings(OFFLINE_MODE=True, RESOURCE_PROFILE="low", MODEL_CACHE_DIR=str(tmp_path))
    create_dense_embeddings(cfg)._get_model()
    create_sparse_embeddings(cfg)
    for factory in (dense, sparse):
        assert factory.call_args.kwargs["cache_dir"] == str(tmp_path)
        assert factory.call_args.kwargs["threads"] == 2
        assert factory.call_args.kwargs["local_files_only"] is True


def test_missing_flashrank_cache_never_downloads(monkeypatch, tmp_path):
    import flashrank

    factory = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(flashrank, "Ranker", factory)
    reranker = PassageReranker(settings=settings(OFFLINE_MODE=True, MODEL_CACHE_DIR=str(tmp_path)))
    assert reranker.ranker is None
    factory.assert_not_called()


@pytest.mark.parametrize(("names", "status"), [(["qwen2.5:1.5b"], "pass"), ([], "fail")])
def test_ollama_diagnostics_transport(names, status):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": name} for name in names]})
    )
    check = diagnostics.check_ollama(settings(LLM_PROVIDER="ollama"), transport)
    assert check.status == status


def test_ollama_latest_and_unreachable():
    cfg = settings(LLM_PROVIDER="ollama", LLM_MODEL="custom")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": "custom:latest"}]})
    )
    assert diagnostics.check_ollama(cfg, transport).status == "pass"

    def fail(request):
        raise httpx.ConnectError("not running", request=request)

    assert diagnostics.check_ollama(cfg, httpx.MockTransport(fail)).status == "fail"


def test_diagnostics_hardware_and_offline_consistency(monkeypatch):
    monkeypatch.setattr(
        diagnostics.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4 * 1024**3, available=1024**3),
    )
    monkeypatch.setattr(diagnostics.psutil, "cpu_count", lambda **kwargs: 2)
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        diagnostics.ctypes,
        "windll",
        SimpleNamespace(
            kernel32=SimpleNamespace(IsProcessorFeaturePresent=lambda flag: flag == 40)
        ),
        raising=False,
    )
    monkeypatch.setattr(diagnostics, "check_model_cache", lambda cfg: [])
    manager = Mock()
    manager.count_points.return_value = 24
    monkeypatch.setattr(diagnostics, "QdrantManager", lambda **kwargs: manager)
    checks = {
        check.name: check for check in diagnostics.run_diagnostics(settings(OFFLINE_MODE=True))
    }
    assert checks["RAM"].status == "warn"
    assert checks["AVX2"].status == "pass"
    assert checks["Offline consistency"].status == "fail"
    assert checks["Qdrant"].status == "pass"
    manager.validate_collection.assert_called_once()
    manager.client.close.assert_called_once()


def test_doctor_exit_and_cli_defaults(monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings(RESOURCE_PROFILE="low"))
    monkeypatch.setattr(
        cli, "run_diagnostics", lambda cfg: [diagnostics.CheckResult("test", "fail", "missing")]
    )
    outcome = CliRunner().invoke(cli.app, ["doctor"])
    assert outcome.exit_code == 1 and "missing" in outcome.output
    retriever = Retriever([])
    monkeypatch.setattr(cli, "_custom_retriever", retriever)
    assert CliRunner().invoke(cli.app, ["retrieve", "love"]).exit_code == 0
    assert retriever.top_k == 3
    assert QueryRequest(query="love", provider="ollama").top_k is None
    assert RetrieveRequest(query="love").top_k is None


@pytest.fixture
def snapshot_module():
    path = Path(__file__).parents[1] / "scripts" / "qdrant_snapshot.py"
    spec = importlib.util.spec_from_file_location("snapshot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("field", "value"), [("dense_model", "wrong"), ("dense_dim", 1536), ("sparse_model", "wrong")]
)
def test_snapshot_sidecar_rejects_mismatch(snapshot_module, field, value):
    cfg = settings(EMBEDDING_PROVIDER="fastembed")
    sidecar = snapshot_module.embedding_metadata(cfg)
    snapshot_module.validate_sidecar(sidecar, cfg)
    sidecar[field] = value
    with pytest.raises(ValueError, match=field):
        snapshot_module.validate_sidecar(sidecar, cfg)


def test_snapshot_import_mismatch_exits_before_network(snapshot_module, monkeypatch, tmp_path):
    import json

    file = tmp_path / "data.snapshot"
    file.write_bytes(b"snapshot")
    Path(str(file) + ".json").write_text(json.dumps({"dense_model": "wrong"}))
    post = Mock(side_effect=AssertionError("must not contact Qdrant"))
    monkeypatch.setattr(snapshot_module.httpx, "post", post)
    monkeypatch.setattr(snapshot_module, "get_settings", lambda: settings())
    outcome = CliRunner().invoke(snapshot_module.app, ["import", "--file", str(file)])
    assert outcome.exit_code == 1 and "mismatch" in outcome.output
    post.assert_not_called()


def test_snapshot_round_trip_transport(snapshot_module, monkeypatch, tmp_path):
    import json

    cfg = settings(EMBEDDING_PROVIDER="fastembed", QDRANT_API_KEY="test-key")
    monkeypatch.setattr(snapshot_module, "get_settings", lambda: cfg)
    manager = Mock()
    manager.count_points.return_value = 24
    manager.client.create_snapshot.return_value = SimpleNamespace(name="server.snapshot")
    monkeypatch.setattr(snapshot_module, "QdrantManager", lambda **kwargs: manager)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["api-key"] == "test-key"
        if request.method == "GET":
            assert request.url.path.endswith("/snapshots/server.snapshot")
            return httpx.Response(200, content=b"snapshot bytes")
        assert request.url.params["priority"] == "snapshot"
        assert request.url.path.endswith("/snapshots/upload")
        assert b"snapshot bytes" in request.read()
        return httpx.Response(200, json={"result": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(snapshot_module.httpx, "stream", client.stream)
        monkeypatch.setattr(snapshot_module.httpx, "post", client.post)
        file = tmp_path / "collection.snapshot"
        runner = CliRunner()
        exported = runner.invoke(snapshot_module.app, ["export", "--out", str(file)])
        assert exported.exit_code == 0, exported.output
        assert file.read_bytes() == b"snapshot bytes"
        sidecar = json.loads(Path(str(file) + ".json").read_text())
        assert sidecar["points"] == 24 and sidecar["dense_dim"] == 384
        assert sidecar["created_at"]
        imported = runner.invoke(snapshot_module.app, ["import", "--file", str(file)])
        assert imported.exit_code == 0, imported.output
    assert len(requests) == 2
    assert manager.validate_collection.call_count == 2


@pytest.mark.parametrize("query", ["Genesis 1:1, 999", "Genesis 1-3"])
def test_fast_path_does_not_ignore_invalid_or_capped_ranges(query):
    chunks = [result("\n".join(f"[Genesis 1:{verse}] Text." for verse in range(1, 32)))]
    assert try_fast_answer(route_query(query), chunks) is None


def test_offline_injected_model_cannot_bypass_cloud_guard():
    pipeline = BSBRAGPipeline(
        Retriever([result("[John 3:16] Verse.")]),
        settings=settings(OFFLINE_MODE=True),
        llm=GenericFakeChatModel(messages=iter(["should not run"])),
    )
    with pytest.raises(ProviderUnavailableError):
        pipeline.ask("Explain love")


def test_cli_fast_query_needs_no_key_or_ollama(monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings(RESOURCE_PROFILE="low"))
    monkeypatch.setattr(cli, "_custom_retriever", Retriever([result("[John 3:16] Verse.")]))
    monkeypatch.setattr(cli, "_custom_pipeline_factory", None)
    outcome = CliRunner().invoke(cli.app, ["query", "John 3:16", "--provider", "ollama"])
    assert outcome.exit_code == 0, outcome.output
    assert "fast_path" in outcome.output and "Verse." in outcome.output


@pytest.mark.parametrize("failure", ["missing", "empty", "schema", "unreachable"])
def test_qdrant_diagnostics_failures(monkeypatch, failure):
    manager = Mock()
    manager.collection_exists.return_value = failure != "missing"
    manager.count_points.return_value = 0 if failure == "empty" else 24
    if failure == "schema":
        manager.validate_collection.side_effect = ValueError("schema mismatch")
    if failure == "unreachable":
        manager.collection_exists.side_effect = ConnectionError("unreachable")
    monkeypatch.setattr(diagnostics, "QdrantManager", lambda **kwargs: manager)
    monkeypatch.setattr(diagnostics, "check_model_cache", lambda cfg: [])
    checks = diagnostics.run_diagnostics(settings())
    assert next(check for check in checks if check.name == "Qdrant").status == "fail"


def test_cache_diagnostics_checks_files_without_loading_models(monkeypatch, tmp_path):
    import fastembed
    from flashrank.Config import model_file_map

    cfg = settings(MODEL_CACHE_DIR=str(tmp_path))
    dense_dir, sparse_dir = tmp_path / "dense", tmp_path / "sparse"
    for cls, directory in (
        (fastembed.TextEmbedding, dense_dir),
        (fastembed.SparseTextEmbedding, sparse_dir),
    ):
        directory.mkdir()
        (directory / "model.bin").write_bytes(b"test")
        monkeypatch.setattr(
            cls,
            "_get_model_description",
            Mock(return_value=SimpleNamespace(model_file="model.bin", additional_files=[])),
        )

        def resolve_cache(*args, directory=directory, **kwargs):
            assert kwargs["local_files_only"] is True
            return directory

        monkeypatch.setattr(cls, "download_model", resolve_cache)
    rank_dir = tmp_path / cfg.reranker_model
    rank_dir.mkdir()
    for name in (
        model_file_map[cfg.reranker_model],
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        (rank_dir / name).write_bytes(b"test")
    assert all(check.status == "pass" for check in diagnostics.check_model_cache(cfg))
    (dense_dir / "model.bin").unlink()
    assert diagnostics.check_model_cache(cfg)[0].status == "fail"
