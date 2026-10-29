"""End-to-end multi-source BSB ingestion, retrieval benchmark, and RAG integration tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from beresbirag.config import Settings, resolve_data_file
from beresbirag.ingestion import (
    IngestionPipeline,
    IngestionStats,
    parse_book_list,
    parse_source_list,
)
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.models.schemas import QueryIntent, SourceType
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager


def _load_script_module(script_name: str) -> Any:
    """Dynamically load a CLI script from ./scripts/ for direct integration testing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_Path_stem := script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_Path_stem] = module
    spec.loader.exec_module(module)
    return module


class _GroundedEchoChatModel(SimpleChatModel):
    """Deterministic local chat model that cites the first retrieved BSB context block."""

    model_name: str = "gpt-4o-mini-test"

    @property
    def _llm_type(self) -> str:
        return "grounded-echo-chat"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        prompt_text = "\n".join(str(m.content) for m in messages)
        for line in prompt_text.splitlines():
            if line.startswith("[") and "]" in line and not line.startswith("[Source:"):
                return f"According to the Berean Standard Bible: {line}"
        return "Grounded response from Berean Standard Bible context."


def _have_all_raw_bsb_files() -> bool:
    required = (
        "bsb_usj.zip",
        "bsb_book_outlines.pdf",
        "bsb_topical_index.xlsx",
        "bsb_concordance.xlsx",
    )
    return all(resolve_data_file(name).is_file() for name in required)


@pytest.fixture(scope="module")
def e2e_indexed_stack() -> tuple[QdrantManager, IngestionStats, BSBHybridRetriever]:
    """Index a real multi-source BSB slice (GEN, JHN, REV + Outlines + Topics + Concordance) into memory."""
    if not _have_all_raw_bsb_files():
        pytest.skip("Raw BSB corpus files are unavailable in this environment")

    settings = Settings(
        QDRANT_MODE="memory",
        QDRANT_COLLECTION_NAME="test_e2e_bsb_rag",
        RERANKER_ENABLED=False,
    )
    manager = QdrantManager(
        settings=settings,
        dense_embeddings=DeterministicHashEmbeddings(size=384),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=384,
    )
    pipeline = IngestionPipeline(qdrant_manager=manager, settings=settings)
    stats = pipeline.run(
        recreate=True,
        sources="scripture,book_outline,topical_index,concordance",
        books="GEN,JHN,REV",
        max_topics=15,
        max_concordance=15,
        batch_size=64,
        show_progress=True,
    )
    reranker = PassageReranker(settings=settings, enabled=False)
    retriever = BSBHybridRetriever(
        qdrant_manager=manager,
        reranker=reranker,
        settings=settings,
    )
    return manager, stats, retriever


def test_parse_source_and_book_lists() -> None:
    assert parse_source_list("scripture,book_outline") == [
        SourceType.SCRIPTURE,
        SourceType.BOOK_OUTLINE,
    ]
    assert parse_source_list([SourceType.TOPICAL_INDEX, "concordance"]) == [
        SourceType.TOPICAL_INDEX,
        SourceType.CONCORDANCE,
    ]
    assert len(parse_source_list(None)) == 4
    assert parse_book_list("gen, jhn , rev") == {"GEN", "JHN", "REV"}
    assert parse_book_list(None) is None


def test_e2e_ingestion_counts_and_sources(
    e2e_indexed_stack: tuple[QdrantManager, IngestionStats, BSBHybridRetriever],
) -> None:
    manager, stats, _ = e2e_indexed_stack
    assert stats.book_outline_count >= 66
    assert stats.scripture_count > 100
    assert stats.topical_index_count >= 15
    assert stats.concordance_count == 15
    assert stats.total_upserted == (
        stats.book_outline_count
        + stats.scripture_count
        + stats.topical_index_count
        + stats.concordance_count
    )
    assert stats.collection_points == stats.total_upserted
    assert manager.count_points() == stats.total_upserted


def test_e2e_retrieval_benchmark_recall_and_mrr(
    e2e_indexed_stack: tuple[QdrantManager, IngestionStats, BSBHybridRetriever],
) -> None:
    _, _, retriever = e2e_indexed_stack
    eval_mod = _load_script_module("evaluate_retrieval.py")
    report = eval_mod.evaluate_benchmark(retriever, cases=eval_mod.BENCHMARK_QUERIES, top_k=5)

    assert len(report.results) == 15
    assert report.recall_at_5 >= 0.90, (
        f"Expected Recall@5 >= 0.90, got {report.recall_at_5:.2%} "
        f"(misses: {[r.case.case_id for r in report.results if not r.hit_at_5]})"
    )
    assert report.recall_at_1 >= 0.85
    assert report.mrr >= 0.85


def test_e2e_rag_pipeline_across_intents(
    e2e_indexed_stack: tuple[QdrantManager, IngestionStats, BSBHybridRetriever],
) -> None:
    _, _, retriever = e2e_indexed_stack
    rag = BSBRAGPipeline(
        retriever=retriever,
        llm=_GroundedEchoChatModel(),
        provider="openai",
        model_name="gpt-4o-mini",
        effort="minimal",
    )

    # 1. Exact Scripture & Footnote query
    resp_scripture = rag.ask(
        "What does Genesis 1:1-3 say and what footnote is on verse 3?", top_k=4
    )
    assert resp_scripture.intent == QueryIntent.EXACT_PASSAGE
    assert resp_scripture.citations
    assert resp_scripture.citations[0].source_type == SourceType.SCRIPTURE
    assert "Genesis 1:1" in resp_scripture.answer
    assert any("2 Corinthians 4:6" in r.chunk.text for r in resp_scripture.retrieved_chunks)

    # 2. Book Outline query
    resp_outline = rag.ask("What is the Genesis outline?", top_k=4)
    assert resp_outline.intent == QueryIntent.OUTLINE
    assert any(
        c.source_type == SourceType.BOOK_OUTLINE and c.book_code == "GEN"
        for c in resp_outline.citations
    )

    # 3. Topical Index query
    resp_topic = rag.ask("Topic: 666", top_k=4)
    assert resp_topic.intent == QueryIntent.TOPICAL
    assert any(c.source_type == SourceType.TOPICAL_INDEX for c in resp_topic.citations)

    # 4. Concordance query
    resp_conc = rag.ask("Occurrences of 12 in the concordance", top_k=4)
    assert resp_conc.intent == QueryIntent.CONCORDANCE
    assert any(c.source_type == SourceType.CONCORDANCE for c in resp_conc.citations)
