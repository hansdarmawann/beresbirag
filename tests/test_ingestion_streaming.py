"""Acceptance tests for streaming ingestion, index metadata, and corpus verification (TASK-12)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from beresbirag.config import Settings
from beresbirag.ingestion.pipeline import IngestionPipeline
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import CollectionConfigError, QdrantManager


def _make_chunk(idx: int) -> ChunkDocument:
    cid = f"scripture:GEN:1:{idx}:0"
    return ChunkDocument(
        chunk_id=cid,
        text=f"[Genesis 1:{idx}] Sample verse text {idx}.",
        metadata=ChunkMetadata(
            chunk_id=cid,
            source_type=SourceType.SCRIPTURE,
            book_code="GEN",
            book_name="Genesis",
            chapter=1,
            verse_start=idx,
            verse_end=idx,
            canonical_ref=f"Genesis 1:{idx}",
            verse_refs=[f"GEN 1:{idx}"],
        ),
    )


def _memory_manager(
    *,
    dense_model: str = "BAAI/bge-small-en-v1.5",
    sparse_model: str = "Qdrant/bm25",
    client: Any = None,
) -> QdrantManager:
    settings = Settings(
        qdrant_mode="memory",
        qdrant_collection_name="bsb_stream_test",
        embedding_provider="fastembed",
        fastembed_dense_model=dense_model,
        fastembed_sparse_model=sparse_model,
    )
    return QdrantManager(
        settings=settings,
        client=client,
        dense_embeddings=DeterministicHashEmbeddings(size=384),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=384,
    )


@pytest.mark.parametrize("show_progress", [False, True])
def test_streaming_ingestion_buffers_at_most_batch_size(show_progress: bool) -> None:
    total_chunks = 15
    batch_size = 4
    yielded_count = 0
    upserted_count = 0
    max_in_flight = 0

    manager = _memory_manager()
    orig_upsert = manager.upsert_chunks

    def instrumented_upsert(chunks: Iterable[ChunkDocument], batch_size: int = 64) -> int:
        nonlocal upserted_count, max_in_flight
        batch_list = list(chunks)
        in_flight = yielded_count - upserted_count
        max_in_flight = max(max_in_flight, in_flight, len(batch_list))
        count = orig_upsert(batch_list, batch_size=batch_size)
        upserted_count += count
        return count

    manager.upsert_chunks = instrumented_upsert  # type: ignore[method-assign]

    pipeline = IngestionPipeline(qdrant_manager=manager, settings=manager.settings)

    def instrumented_generator(**kwargs: Any) -> Iterator[ChunkDocument]:
        nonlocal yielded_count, max_in_flight
        for i in range(1, total_chunks + 1):
            yielded_count += 1
            in_flight = yielded_count - upserted_count
            max_in_flight = max(max_in_flight, in_flight)
            yield _make_chunk(i)

    pipeline.iter_scripture_chunks = instrumented_generator  # type: ignore[method-assign]

    stats = pipeline.run(
        recreate=True,
        sources=[SourceType.SCRIPTURE],
        batch_size=batch_size,
        show_progress=show_progress,
    )

    assert stats.total_upserted == total_chunks
    assert max_in_flight <= batch_size


def test_meta_collection_written_and_validated_on_model_mismatch() -> None:
    manager = _memory_manager(dense_model="BAAI/bge-small-en-v1.5")
    pipeline = IngestionPipeline(qdrant_manager=manager, settings=manager.settings)
    pipeline.iter_scripture_chunks = lambda **kw: iter([_make_chunk(1), _make_chunk(2)])  # type: ignore[method-assign]

    manifest_hashes = {
        "bsb_usj.zip": "062785c0621e0976e7bf3d5a0c32d3d38c5e9d8f71215217c5d33be03bdfaede",
        "bsb_book_outlines.pdf": "a3a648d85ce85bf0f7d2404769ebfcb73e6e3e3dd880eb8122b2d4d0318ff0e0",
    }
    pipeline.run(
        recreate=True,
        sources=[SourceType.SCRIPTURE],
        batch_size=2,
        show_progress=False,
        corpus_manifest=manifest_hashes,
    )

    meta = manager.read_index_metadata()
    assert meta is not None
    assert meta["embedding_provider"] == "fastembed"
    assert meta["dense_model"] == "BAAI/bge-small-en-v1.5"
    assert meta["dense_dim"] == 384
    assert meta["sparse_model"] == "Qdrant/bm25"
    assert meta["sparse_modifier"] == "idf"
    assert meta["corpus_manifest"] == manifest_hashes
    assert meta["chunking"] == {"max_verses_per_chunk": 10, "verse_overlap": 2}

    # Matching model passes validation
    manager.validate_collection()

    # Mismatched dense_model sharing the same client raises CollectionConfigError
    mismatched_manager = _memory_manager(
        dense_model="BAAI/bge-small-en-v1.5-other",
        client=manager.client,
    )
    with pytest.raises(CollectionConfigError, match="dense_model"):
        mismatched_manager.validate_collection()


def test_verify_corpus_exits_1_on_sha256_mismatch(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_corpus.py"
    spec = importlib.util.spec_from_file_location("verify_corpus_test", script_path)
    assert spec is not None and spec.loader is not None
    verify_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_mod)

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    sample_file = data_dir / "bsb_usj.zip"
    sample_file.write_bytes(b"actual content")

    manifest_file = tmp_path / "corpus_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "bsb_usj.zip",
                        "bytes": len(b"actual content"),
                        "sha256": "0" * 64,
                        "source_url": "TODO: official BSB download URL",
                        "notes": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        verify_mod.app,
        ["--manifest", str(manifest_file), "--data-dir", str(data_dir)],
    )
    assert result.exit_code == 1
    assert "mismatch" in result.output.lower()
