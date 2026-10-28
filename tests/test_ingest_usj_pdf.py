"""Integration tests for the USJ scripture chunker and PDF book outline parser."""

from __future__ import annotations

import pytest

from beresbirag.ingestion.pdf_outline_parser import (
    BookOutlineIndex,
    parse_book_outlines,
    parse_verse_range,
)
from beresbirag.ingestion.usj_parser import iter_usj_scripture_chunks
from beresbirag.models.books import BSB_BOOKS
from beresbirag.models.schemas import ChunkDocument, SourceType


@pytest.fixture(scope="module")
def outline_index() -> BookOutlineIndex:
    return BookOutlineIndex.from_pdf()


@pytest.fixture(scope="module")
def scripture_chunks(outline_index: BookOutlineIndex) -> list[ChunkDocument]:
    return list(
        iter_usj_scripture_chunks(outline_index=outline_index, book_filter={"GEN", "JHN", "PHM"})
    )


def test_parse_verse_range_forms() -> None:
    assert parse_verse_range("1:1\u20132:25", False) == (1, 1, 2, 25)
    assert parse_verse_range("\u202f1:1\u2060\u201331\u202f", False) == (1, 1, 1, 31)
    assert parse_verse_range("9:31", False) == (9, 31, 9, 31)
    assert parse_verse_range("1\u201325", True) == (1, 1, 1, 25)


def test_outline_covers_all_66_books(outline_index: BookOutlineIndex) -> None:
    assert set(outline_index.nodes_by_book) == {b.code for b in BSB_BOOKS}
    assert all(outline_index.nodes_by_book[b.code] for b in BSB_BOOKS)


@pytest.mark.parametrize(
    ("book", "chapter", "verse"),
    [("GEN", 1, 1), ("EXO", 20, 1), ("JHN", 1, 1), ("REV", 21, 1)],
)
def test_outline_breadcrumbs(
    outline_index: BookOutlineIndex, book: str, chapter: int, verse: int
) -> None:
    crumb = outline_index.get_outline_breadcrumbs(book, chapter, verse)
    assert crumb
    assert crumb.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."))


def test_genesis_breadcrumb_is_hierarchical(outline_index: BookOutlineIndex) -> None:
    crumb = outline_index.get_outline_breadcrumbs("GEN", 1, 1)
    assert crumb is not None
    assert crumb.startswith("1. God and the World > a. The Creation")


def test_outline_chunk_documents() -> None:
    index, chunks = parse_book_outlines()
    assert chunks
    assert len(index.nodes_by_book) == 66
    assert all(isinstance(c, ChunkDocument) for c in chunks)
    assert all(c.metadata.source_type == SourceType.BOOK_OUTLINE for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {f"outline:{b.code}:overview" for b in BSB_BOOKS} <= set(ids)
    gen = next(c for c in chunks if c.chunk_id == "outline:GEN:overview")
    assert gen.text.startswith("[Source: BSB Book Outline | Book: Genesis (GEN) | Division:")
    assert gen.metadata.canonical_ref and gen.metadata.canonical_ref.startswith("Genesis Outline")
    assert any(c.chunk_id.startswith("outline:GEN:sec:") for c in chunks)


def test_scripture_books_are_canonically_ordered(scripture_chunks: list[ChunkDocument]) -> None:
    orders = [c.metadata.canonical_order for c in scripture_chunks]
    assert orders == sorted(orders)
    assert {c.metadata.book_code for c in scripture_chunks} == {"GEN", "JHN", "PHM"}
    assert all(c.metadata.source_type == SourceType.SCRIPTURE for c in scripture_chunks)


def test_genesis_first_chunk(scripture_chunks: list[ChunkDocument]) -> None:
    chunk = scripture_chunks[0]
    meta = chunk.metadata
    assert (meta.book_code, meta.chapter, meta.verse_start) == ("GEN", 1, 1)
    assert meta.verse_end is not None and meta.verse_end >= 5
    assert {"GEN 1:1", "GEN 1:2", "GEN 1:3"} <= set(meta.verse_refs)
    assert "[Genesis 1:5]" in chunk.text
    assert "And there was evening, and there was morning" in chunk.text
    assert chunk.text.startswith("[Book: Genesis | Passage: Genesis 1:1-")
    assert meta.section_heading and meta.section_heading.startswith("The Creation")
    assert "The First Day" in meta.section_heading


def test_genesis_footnotes_and_cross_refs(scripture_chunks: list[ChunkDocument]) -> None:
    chunk = scripture_chunks[0]
    meta = chunk.metadata
    for note in ("1:3 Cited in 2 Corinthians 4:6", "1:5 Literally day one"):
        assert note in chunk.text
        assert any(note in f for f in meta.footnotes)
    for ref in ("John 1:1\u20135", "Hebrews 11:1\u20133"):
        assert ref in chunk.text
        assert ref in meta.cross_references


def test_outline_path_populated_only_with_index(scripture_chunks: list[ChunkDocument]) -> None:
    assert scripture_chunks[0].metadata.outline_path
    assert "Outline:" in scripture_chunks[0].text
    plain = next(iter_usj_scripture_chunks(book_filter={"PHM"}))
    assert plain.metadata.outline_path is None


def test_sliding_window_and_unique_ids(scripture_chunks: list[ChunkDocument]) -> None:
    ids = [c.chunk_id for c in scripture_chunks]
    assert len(ids) == len(set(ids))
    assert all(len(c.metadata.verse_refs) <= 10 for c in scripture_chunks)
    small = list(iter_usj_scripture_chunks(book_filter={"PHM"}, max_verses_per_chunk=4, verse_overlap=1))
    assert all(len(c.metadata.verse_refs) <= 4 for c in small)
    assert any(c.chunk_id.endswith(":1") for c in small)


def test_philemon_single_chapter_book(scripture_chunks: list[ChunkDocument]) -> None:
    phm = [c for c in scripture_chunks if c.metadata.book_code == "PHM"]
    verses = {r for c in phm for r in c.metadata.verse_refs}
    assert verses == {f"PHM 1:{v}" for v in range(1, 26)}


def test_john_prologue(scripture_chunks: list[ChunkDocument]) -> None:
    jhn = next(c for c in scripture_chunks if c.metadata.book_code == "JHN")
    assert "JHN 1:1" in jhn.metadata.verse_refs
    assert "In the beginning was the Word" in jhn.text


def test_invalid_window_arguments() -> None:
    with pytest.raises(ValueError):
        list(iter_usj_scripture_chunks(book_filter={"PHM"}, max_verses_per_chunk=3, verse_overlap=3))
