"""Integration tests for the USJ scripture chunker and PDF book outline parser."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from beresbirag.config import resolve_data_file
from beresbirag.ingestion.pdf_outline_parser import (
    BookOutlineIndex,
    parse_book_outlines,
    parse_verse_range,
)
from beresbirag.ingestion.usj_parser import iter_usj_scripture_chunks
from beresbirag.models.books import BSB_BOOKS
from beresbirag.models.schemas import ChunkDocument, SourceType


def build_synthetic_outline_pdf(tmp_path: Path) -> Path:
    """Create a 3-page synthetic outlines PDF conforming to the parser's page offset."""
    doc = fitz.open()
    doc.new_page()  # Page 0 (cover)
    doc.new_page()  # Page 1 (intro)
    page2 = doc.new_page()  # Page 2 (FIRST_OUTLINE_PAGE_INDEX)
    lines = [
        "Genesis",
        "Outline",
        "1. God and the World ( 1:1–2:25 )",
        "a. The Creation ( 1:1–2:3 )",
        "i. The Six Days of Creation ( 1:1–31 )",
        "1. The First Day: Light ( 1:1–5 )",
    ]
    page2.insert_text((50, 72), "\n".join(lines))
    pdf_path = tmp_path / "synthetic_outlines.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def build_synthetic_usj_zip(tmp_path: Path) -> Path:
    """Create a minimal USJ 3.1 ZIP archive containing Genesis and Philemon."""
    gen_doc = {
        "type": "USJ",
        "version": "3.1",
        "content": [
            {"type": "chapter", "marker": "c", "number": "1"},
            {"type": "para", "marker": "s1", "content": ["The Creation"]},
            {
                "type": "para",
                "marker": "r",
                "content": [{"type": "ref", "loc": "JHN 1:1-5", "content": ["John 1:1–5"]}],
            },
            {
                "type": "para",
                "marker": "p",
                "content": [
                    {"type": "verse", "marker": "v", "number": "1"},
                    "In the beginning God created the heavens and the earth.",
                    {"type": "verse", "marker": "v", "number": "2"},
                    "Now the earth was formless and void.",
                ],
            },
            {
                "type": "para",
                "marker": "pmo",
                "content": [
                    {"type": "verse", "marker": "v", "number": "3"},
                    "And God said, 'Let there be light,' and there was light.",
                    {"type": "note", "marker": "f", "content": ["1:3 Cited in 2 Corinthians 4:6"]},
                ],
            },
        ],
    }
    phm_verses = []
    for v in range(1, 15):
        phm_verses.extend([{"type": "verse", "marker": "v", "number": str(v)}, f"Verse {v} text."])
    phm_doc = {
        "type": "USJ",
        "version": "3.1",
        "content": [
            {"type": "chapter", "marker": "c", "number": "1"},
            {"type": "para", "marker": "p", "content": phm_verses},
        ],
    }

    zip_path = tmp_path / "synthetic_usj.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("bsb_usj/GEN.usj", json.dumps(gen_doc))
        zf.writestr("bsb_usj/PHM.usj", json.dumps(phm_doc))
    return zip_path


def test_synthetic_outline_parsing_and_breadcrumbs(tmp_path: Path) -> None:
    pdf_path = build_synthetic_outline_pdf(tmp_path)
    index, chunks = parse_book_outlines(pdf_path)
    assert "GEN" in index.nodes_by_book
    assert len(index.nodes_by_book["GEN"]) == 4

    crumb = index.get_outline_breadcrumbs("GEN", 1, 1)
    assert crumb is not None
    assert (
        crumb
        == "1. God and the World > a. The Creation > i. The Six Days of Creation > 1. The First Day: Light"
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "outline:GEN:overview"
    assert chunk.metadata.source_type == SourceType.BOOK_OUTLINE
    assert chunk.metadata.book_code == "GEN"
    assert "[Source: BSB Book Outline | Book: Genesis (GEN)" in chunk.text


def test_synthetic_usj_chunking_with_outline(tmp_path: Path) -> None:
    pdf_path = build_synthetic_outline_pdf(tmp_path)
    outline_index = BookOutlineIndex.from_pdf(pdf_path)

    zip_path = build_synthetic_usj_zip(tmp_path)
    chunks = list(
        iter_usj_scripture_chunks(
            zip_path=zip_path,
            outline_index=outline_index,
            book_filter={"GEN"},
        )
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "scripture:GEN:1:1-3:0"
    assert chunk.metadata.book_code == "GEN"
    assert chunk.metadata.chapter == 1
    assert chunk.metadata.verse_start == 1
    assert chunk.metadata.verse_end == 3
    assert chunk.metadata.verse_refs == ["GEN 1:1", "GEN 1:2", "GEN 1:3"]
    assert chunk.metadata.section_heading == "The Creation"
    assert chunk.metadata.cross_references == ["John 1:1–5"]
    assert any("Cited in 2 Corinthians 4:6" in note for note in chunk.metadata.footnotes)
    assert chunk.metadata.outline_path is not None
    assert "1. God and the World" in chunk.metadata.outline_path
    assert "[Genesis 1:1] In the beginning" in chunk.text
    assert "Footnotes:" in chunk.text


def test_synthetic_usj_sliding_window_overlap(tmp_path: Path) -> None:
    zip_path = build_synthetic_usj_zip(tmp_path)
    chunks = list(
        iter_usj_scripture_chunks(
            zip_path=zip_path,
            book_filter={"PHM"},
            max_verses_per_chunk=5,
            verse_overlap=2,
        )
    )
    assert len(chunks) > 1
    assert all(len(c.metadata.verse_refs) <= 5 for c in chunks)
    assert chunks[0].metadata.verse_refs == [f"PHM 1:{v}" for v in range(1, 6)]
    assert chunks[1].metadata.verse_refs[0] == "PHM 1:4"
    assert chunks[0].chunk_id.endswith(":0")
    assert chunks[1].chunk_id.endswith(":1")


@pytest.fixture(scope="module")
def outline_index() -> BookOutlineIndex:
    if not resolve_data_file("bsb_book_outlines.pdf").is_file():
        pytest.skip("Original BSB Book Outlines PDF is unavailable")
    return BookOutlineIndex.from_pdf()


@pytest.fixture(scope="module")
def scripture_chunks(outline_index: BookOutlineIndex) -> list[ChunkDocument]:
    if not resolve_data_file("bsb_usj.zip").is_file():
        pytest.skip("Original BSB USJ archive is unavailable")
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
    if not resolve_data_file("bsb_book_outlines.pdf").is_file():
        pytest.skip("Original BSB Book Outlines PDF is unavailable")
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
    small = list(
        iter_usj_scripture_chunks(book_filter={"PHM"}, max_verses_per_chunk=4, verse_overlap=1)
    )
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
        list(
            iter_usj_scripture_chunks(book_filter={"PHM"}, max_verses_per_chunk=3, verse_overlap=3)
        )
