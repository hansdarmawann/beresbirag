"""Unit and integration tests for TASK-04: QdrantManager, hybrid RRF search, and payload filtering."""

from __future__ import annotations

from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from beresbirag.config import Settings
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
    FastEmbedDenseEmbeddings,
    create_dense_embeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager


def _sample_chunks() -> list[ChunkDocument]:
    return [
        ChunkDocument(
            chunk_id="scripture:GEN:1:1-5:0",
            text=(
                "[Book: Genesis | Passage: Genesis 1:1-5 | Heading: The Creation / The First Day]\n"
                "[Genesis 1:1] In the beginning God created the heavens and the earth.\n"
                "[Genesis 1:3] And God said, 'Let there be light,' and there was light."
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:GEN:1:1-5:0",
                source_type=SourceType.SCRIPTURE,
                book_code="GEN",
                book_name="Genesis",
                testament="OT",
                canonical_order=1,
                chapter=1,
                verse_start=1,
                verse_end=5,
                canonical_ref="Genesis 1:1-5",
                verse_refs=["GEN 1:1", "GEN 1:2", "GEN 1:3", "GEN 1:4", "GEN 1:5"],
                section_heading="The Creation / The First Day",
                outline_path="1. God and the World > a. The Creation",
            ),
        ),
        ChunkDocument(
            chunk_id="scripture:JHN:1:1-5:0",
            text=(
                "[Book: John | Passage: John 1:1-5 | Heading: The Beginning]\n"
                "[John 1:1] In the beginning was the Word, and the Word was with God, and the Word was God."
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:JHN:1:1-5:0",
                source_type=SourceType.SCRIPTURE,
                book_code="JHN",
                book_name="John",
                testament="NT",
                canonical_order=43,
                chapter=1,
                verse_start=1,
                verse_end=5,
                canonical_ref="John 1:1-5",
                verse_refs=["JHN 1:1", "JHN 1:2", "JHN 1:3", "JHN 1:4", "JHN 1:5"],
                section_heading="The Beginning",
            ),
        ),
        ChunkDocument(
            chunk_id="outline:GEN:overview",
            text=(
                "[Source: BSB Book Outline | Book: Genesis (GEN)]\n"
                "1. God and the World (1:1-2:25)\n"
                "   a. The Creation (1:1-2:3)\n"
                "2. Early History of Mankind (3:1-11:32)"
            ),
            metadata=ChunkMetadata(
                chunk_id="outline:GEN:overview",
                source_type=SourceType.BOOK_OUTLINE,
                book_code="GEN",
                book_name="Genesis",
                testament="OT",
                canonical_order=1,
                canonical_ref="Genesis Outline",
            ),
        ),
        ChunkDocument(
            chunk_id="topical:666:0",
            text=(
                "[Source: BSB Topical Index | Topic: 666 | Total References: 8]\n"
                "- [Revelation 13:16-18] Here is a call for wisdom: calculate the number of the beast, 666.\n"
                "- [2 Chronicles 9:13] The weight of gold that came to Solomon each year was 666 talents."
            ),
            metadata=ChunkMetadata(
                chunk_id="topical:666:0",
                source_type=SourceType.TOPICAL_INDEX,
                canonical_ref="Topical Index: 666",
                topic="666",
                topic_display="666",
                verse_refs=["REV 13:16", "REV 13:17", "REV 13:18", "2CH 9:13"],
            ),
        ),
        ChunkDocument(
            chunk_id="concordance:12:0",
            text=(
                "[Source: BSB Concordance | Headword: 12 | Total Occurrences: 26]\n"
                "Book Distribution: 1Ch (6), Gen (3), Mat (4)\n"
                "- [1Ch 25:9] sons, and his brothers— 12 in all;"
            ),
            metadata=ChunkMetadata(
                chunk_id="concordance:12:0",
                source_type=SourceType.CONCORDANCE,
                canonical_ref="Concordance: 12 (26 Occurrences)",
                concordance_word="12",
                concordance_occurrences=26,
                verse_refs=["1CH 25:9", "1CH 25:10"],
            ),
        ),
    ]


def test_embedding_factory_dual_provider() -> None:
    s_fast = Settings(EMBEDDING_PROVIDER="auto", OPENAI_API_KEY="")
    dense_fast = create_dense_embeddings(s_fast)
    assert isinstance(dense_fast, FastEmbedDenseEmbeddings)

    s_openai = Settings(EMBEDDING_PROVIDER="auto", OPENAI_API_KEY="sk-fake-key-for-unit-test")
    dense_openai = create_dense_embeddings(s_openai)
    assert isinstance(dense_openai, OpenAIEmbeddings)


def test_qdrant_manager_hybrid_upsert_scroll_and_search() -> None:
    settings = Settings(QDRANT_MODE="memory", QDRANT_COLLECTION_NAME="test_bsb_rag")
    manager = QdrantManager(
        settings=settings,
        dense_embeddings=DeterministicHashEmbeddings(size=128),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=128,
    )
    manager.ensure_collection(recreate=True)
    assert manager.collection_exists()

    chunks = _sample_chunks()
    upserted = manager.upsert_chunks(chunks, batch_size=2)
    assert upserted == 5
    assert manager.count_points() == 5

    # Re-upserting the same chunks should be idempotent (still 5 points)
    manager.upsert_chunks(chunks[:2], batch_size=2)
    assert manager.count_points() == 5

    # 1. Exact verse_refs filter via scroll_by_filter
    verse_filter = manager.build_filter(verse_refs=["GEN 1:3"])
    assert verse_filter is not None
    exact_hits = manager.scroll_by_filter(verse_filter)
    assert len(exact_hits) == 1
    assert exact_hits[0].chunk.chunk_id == "scripture:GEN:1:1-5:0"

    # 2. Topic filter via scroll_by_filter
    topic_filter = manager.build_filter(
        source_types=[SourceType.TOPICAL_INDEX],
        topic="666",
    )
    assert topic_filter is not None
    topic_hits = manager.scroll_by_filter(topic_filter)
    assert len(topic_hits) == 1
    assert topic_hits[0].chunk.metadata.topic == "666"

    # 3. Concordance headword filter
    conc_filter = manager.build_filter(
        source_types=[SourceType.CONCORDANCE],
        concordance_word="12",
    )
    assert conc_filter is not None
    conc_hits = manager.scroll_by_filter(conc_filter)
    assert len(conc_hits) == 1
    assert conc_hits[0].chunk.metadata.concordance_occurrences == 26

    # 4. Hybrid RRF search with book_codes filter
    gen_filter = manager.build_filter(
        source_types=[SourceType.SCRIPTURE],
        book_codes=["GEN"],
        testament="OT",
        chapter=1,
    )
    hybrid_hits = manager.hybrid_search(
        query="In the beginning God created the heavens and the earth light",
        k=3,
        qdrant_filter=gen_filter,
    )
    assert len(hybrid_hits) == 1
    assert hybrid_hits[0].chunk.chunk_id == "scripture:GEN:1:1-5:0"
    assert hybrid_hits[0].rank == 1


def test_docker_and_ci_files_exist() -> None:
    wt_root = Path(__file__).resolve().parents[1]
    assert (wt_root / "Dockerfile").is_file()
    assert (wt_root / ".dockerignore").is_file()
    assert (wt_root / "docker-compose.yml").is_file()
    assert (wt_root / ".github" / "workflows" / "ci.yml").is_file()
