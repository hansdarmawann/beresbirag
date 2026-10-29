"""Acceptance tests for TASK-13 repository hygiene, caching, and formatting."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import beresbirag.config
from beresbirag.config import Settings, resolve_data_file
from beresbirag.llm import rag_chain
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, RetrievalResult, SourceType
from beresbirag.retrieval import BSBHybridRetriever, format_context_blocks
from beresbirag.vectorstore.qdrant_manager import QdrantManager


def _make_dummy_result(
    chunk_id: str, text: str, ref: str, source_type: SourceType
) -> RetrievalResult:
    chunk = ChunkDocument(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            source_type=source_type,
            canonical_ref=ref,
            book_code="JHN",
            chapter=1,
            verse_refs=[ref],
        ),
    )
    return RetrievalResult(chunk=chunk, score=0.95, rank=1)


def test_context_formatter_dedup() -> None:
    """1. BSBHybridRetriever.format_context_for_llm(results) == format_context_blocks(results)."""
    results = [
        _make_dummy_result(
            "chunk-1", "In the beginning was the Word...", "John 1:1", SourceType.SCRIPTURE
        ),
        _make_dummy_result("chunk-2", "The Prologue of John", "John 1", SourceType.BOOK_OUTLINE),
    ]

    retriever_formatted = BSBHybridRetriever.format_context_for_llm(results)
    direct_formatted = format_context_blocks(results)
    chain_formatted = rag_chain.format_context_blocks(results)

    assert retriever_formatted == direct_formatted
    assert direct_formatted == chain_formatted
    assert "=== SOURCE [1] | Reference: John 1:1 | Type: scripture ===" in direct_formatted
    assert "=== SOURCE [2] | Reference: John 1 | Type: book_outline ===" in direct_formatted

    # Empty results behavior
    assert BSBHybridRetriever.format_context_for_llm([]) == ""
    assert format_context_blocks([]) == ""
    assert (
        rag_chain.format_context_blocks([])
        == "No relevant context found in the Berean Standard Bible corpus."
    )


def test_vector_store_caching() -> None:
    """2. 5 consecutive hybrid_search calls invoke collection_exists at most once, and recreate re-checks."""
    settings = Settings(qdrant_mode="memory", embedding_provider="fastembed")
    qm = QdrantManager(settings=settings)

    # Initial ensure collection
    qm.ensure_collection(recreate=False)

    call_count = 0
    orig_collection_exists = qm.client.collection_exists

    def counting_spy(collection_name: str) -> bool:
        nonlocal call_count
        call_count += 1
        return bool(orig_collection_exists(collection_name))

    qm.client.collection_exists = counting_spy

    for _ in range(5):
        qm.hybrid_search("grace and truth", k=2)

    assert call_count <= 1

    count_after_5 = call_count
    qm.ensure_collection(recreate=True)
    count_after_recreate = call_count

    qm.hybrid_search("grace and truth", k=2)
    assert call_count > count_after_recreate


def test_bsb_data_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3. BSB_DATA_DIR resolution for relative/absolute paths; inspect config has no .worktrees."""
    monkeypatch.setenv("BSB_DATA_DIR", str(tmp_path))
    settings = Settings(bsb_data_dir=str(tmp_path))

    assert resolve_data_file("bsb_usj.zip") == tmp_path / "bsb_usj.zip"
    assert settings.usj_zip_file == tmp_path / "bsb_usj.zip"

    # Absolute path remains unchanged
    abs_path = (tmp_path / "sub" / "custom.pdf").resolve()
    assert resolve_data_file(abs_path) == abs_path

    # Verify no .worktrees or find_repo_root in config.py
    source = inspect.getsource(beresbirag.config)
    assert ".worktrees" not in source
    assert "find_repo_root" not in source
