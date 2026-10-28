"""Unit tests for TASK-01 foundation: books registry, Pydantic schemas, and config settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from beresbirag.config import Settings, find_repo_root, resolve_data_file
from beresbirag.models.books import (
    BSB_BOOKS,
    format_canonical_ref,
    get_book_by_code,
    resolve_book,
)
from beresbirag.models.schemas import (
    ChunkDocument,
    ChunkMetadata,
    CitationItem,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
    SourceType,
    VerseRange,
    make_point_uuid,
)


def test_bsb_books_registry_complete() -> None:
    assert len(BSB_BOOKS) == 66
    ot_books = [b for b in BSB_BOOKS if b.testament == "OT"]
    nt_books = [b for b in BSB_BOOKS if b.testament == "NT"]
    assert len(ot_books) == 39
    assert len(nt_books) == 27
    orders = [b.canonical_order for b in BSB_BOOKS]
    assert orders == list(range(1, 67))


@pytest.mark.parametrize(
    ("raw_input", "expected_code", "expected_name"),
    [
        ("GEN", "GEN", "Genesis"),
        ("Gen.", "GEN", "Genesis"),
        ("Genesis", "GEN", "Genesis"),
        ("1Ch", "1CH", "1 Chronicles"),
        ("1 Chr", "1CH", "1 Chronicles"),
        ("2 Chronicles", "2CH", "2 Chronicles"),
        ("Psalm", "PSA", "Psalms"),
        ("Psalms", "PSA", "Psalms"),
        ("Ps", "PSA", "Psalms"),
        ("Song of Songs", "SNG", "Song of Solomon"),
        ("Song of Solomon", "SNG", "Song of Solomon"),
        ("1 Cor", "1CO", "1 Corinthians"),
        ("1co", "1CO", "1 Corinthians"),
        ("Jn", "JHN", "John"),
        ("John", "JHN", "John"),
        ("1 John", "1JN", "1 John"),
        ("Rev", "REV", "Revelation"),
    ],
)
def test_resolve_book_aliases(raw_input: str, expected_code: str, expected_name: str) -> None:
    info = resolve_book(raw_input)
    assert info is not None
    assert info.code == expected_code
    assert info.name == expected_name
    assert get_book_by_code(expected_code) == info


def test_format_canonical_ref() -> None:
    assert format_canonical_ref("GEN", 1, 1) == "Genesis 1:1"
    assert format_canonical_ref("PSA", 23, 1) == "Psalm 23:1"
    assert format_canonical_ref("1CO", 13) == "1 Corinthians 13"


def test_chunk_document_roundtrip_and_deterministic_uuid() -> None:
    meta = ChunkMetadata(
        chunk_id="scripture:GEN:1:1-3:0",
        source_type=SourceType.SCRIPTURE,
        book_code="GEN",
        book_name="Genesis",
        testament="OT",
        canonical_order=1,
        chapter=1,
        verse_start=1,
        verse_end=3,
        canonical_ref="Genesis 1:1-3",
        verse_refs=["GEN 1:1", "GEN 1:2", "GEN 1:3"],
        section_heading="The Creation / The First Day",
        outline_path="1. God and the World > a. The Creation",
        cross_references=["John 1:1-5", "Hebrews 11:1-3"],
        footnotes=["1:3 Cited in 2 Corinthians 4:6"],
    )
    chunk = ChunkDocument(
        chunk_id=meta.chunk_id,
        text="[Genesis 1:1] In the beginning God created the heavens and the earth.",
        metadata=meta,
    )
    assert chunk.point_id == make_point_uuid("scripture:GEN:1:1-3:0")
    lc_doc = chunk.to_langchain_document()
    assert lc_doc.id == chunk.point_id
    assert lc_doc.metadata["book_code"] == "GEN"
    assert lc_doc.metadata["source_type"] == "scripture"

    restored = ChunkDocument.from_langchain_document(lc_doc)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == chunk.text
    assert restored.metadata.book_code == "GEN"
    assert restored.metadata.verse_refs == ["GEN 1:1", "GEN 1:2", "GEN 1:3"]


def test_settings_embedding_provider_resolution() -> None:
    s_auto_no_key = Settings(EMBEDDING_PROVIDER="auto", OPENAI_API_KEY="")
    assert s_auto_no_key.resolved_embedding_provider == "fastembed"
    assert s_auto_no_key.dense_vector_size == 384

    s_auto_with_key = Settings(EMBEDDING_PROVIDER="auto", OPENAI_API_KEY="sk-test-123")
    assert s_auto_with_key.resolved_embedding_provider == "openai"
    assert s_auto_with_key.dense_vector_size == 1536

    s_forced_fastembed = Settings(EMBEDDING_PROVIDER="fastembed", OPENAI_API_KEY="sk-test-123")
    assert s_forced_fastembed.resolved_embedding_provider == "fastembed"
    assert s_forced_fastembed.dense_vector_size == 384


def test_rag_response_and_verse_range_models() -> None:
    vr = VerseRange(
        book_code="JHN",
        book_name="John",
        start_chapter=3,
        start_verse=16,
        end_chapter=3,
        end_verse=16,
        canonical_ref="John 3:16",
        expanded_refs=["JHN 3:16"],
    )
    assert vr.expanded_refs == ["JHN 3:16"]

    meta = ChunkMetadata(chunk_id="scripture:JHN:3:16:0", source_type=SourceType.SCRIPTURE)
    doc = ChunkDocument(chunk_id=meta.chunk_id, text="[John 3:16] For God so loved...", metadata=meta)
    rr = RetrievalResult(chunk=doc, score=0.92, rerank_score=0.98, rank=1)
    cit = CitationItem(
        chunk_id=doc.chunk_id,
        source_type=SourceType.SCRIPTURE,
        canonical_ref="John 3:16",
        book_code="JHN",
        score=0.98,
    )
    resp = RAGResponse(
        query="What does John 3:16 say?",
        answer="John 3:16 states...",
        intent=QueryIntent.EXACT_PASSAGE,
        citations=[cit],
        retrieved_chunks=[rr],
    )
    assert resp.intent == QueryIntent.EXACT_PASSAGE
    assert len(resp.citations) == 1


def test_git_hook_and_repo_root_exist() -> None:
    root = find_repo_root()
    hook_file = root / ".githooks" / "commit-msg"
    assert hook_file.is_file()
    hook_text = hook_file.read_text(encoding="utf-8")
    assert "Hans Darmawan" in hook_text
    assert "work.hansdarmawan001@gmail.com" in hook_text
    assert "Co-authored-by:" in hook_text
    assert isinstance(resolve_data_file("./bsb_usj.zip"), Path)
