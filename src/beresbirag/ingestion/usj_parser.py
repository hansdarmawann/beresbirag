"""Parse the BSB USJ 3.1 archive into pericope-based, sliding-window scripture chunks."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beresbirag.config import get_settings, resolve_data_file
from beresbirag.ingestion.pdf_outline_parser import BookOutlineIndex
from beresbirag.models.books import BookInfo, get_book_by_code
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType

# Paragraph markers that carry no scripture text or headings.
_IGNORED_MARKERS = {
    "id", "ide", "usfm", "h", "toc1", "toc2", "toc3", "toca1", "toca2", "toca3",
    "mt", "mt1", "mt2", "mt3", "mt4", "rem", "sts", "cl", "cp", "cd", "sp",
}  # fmt: skip
_SUBHEADING_MARKERS = {"s2", "s3", "s4", "ms", "ms1", "ms2", "mr", "d"}
_VERSE_NUM_RE = re.compile(r"\d+")


@dataclass(slots=True)
class _Verse:
    number: int
    parts: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    subheadings: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


@dataclass(slots=True)
class _Pericope:
    chapter: int
    heading: str | None
    cross_refs: list[str] = field(default_factory=list)
    verses: list[_Verse] = field(default_factory=list)


def _flatten(node: Any) -> str:
    """Flatten a USJ node (or string) to plain text, skipping footnote notes."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") in {"note", "verse", "chapter"}:
            return ""
        return "".join(_flatten(child) for child in node.get("content", []))
    if isinstance(node, list):
        return "".join(_flatten(child) for child in node)
    return ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_refs(node: Any) -> list[str]:
    """Return display text of every nested ``ref`` node, falling back to plain text."""
    refs: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("type") == "ref":
                text = _clean(_flatten(item))
                if text:
                    refs.append(text)
                return
            for child in item.get("content", []):
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(node)
    if refs:
        return refs
    plain = _clean(_flatten(node)).strip("()")
    return [p.strip() for p in plain.split(";") if p.strip()]


class _BookWalker:
    """Walks a USJ document and groups verses into pericopes."""

    def __init__(self) -> None:
        self.chapter = 0
        self.s1: str | None = None
        self.subs: list[str] = []
        self.pericopes: list[_Pericope] = []
        self.current: _Pericope | None = None
        self.verse: _Verse | None = None
        self.verse_index: dict[tuple[int, int], _Verse] = {}

    # -- pericope management
    def _new_pericope(self) -> _Pericope:
        self.current = _Pericope(chapter=self.chapter, heading=self.s1)
        self.pericopes.append(self.current)
        return self.current

    def _ensure_pericope(self) -> _Pericope:
        if self.current is None:
            return self._new_pericope()
        return self.current

    def _start_section(self, text: str) -> None:
        cur = self.current
        if cur is not None and cur.verses:
            self.current = None
        self.s1 = text
        self.subs = []
        if self.current is None:
            self._new_pericope()
        else:
            self.current.heading = text

    # -- node handlers
    def walk(self, content: list[Any]) -> list[_Pericope]:
        for node in content:
            if not isinstance(node, dict):
                continue
            kind = node.get("type")
            if kind == "chapter":
                self._on_chapter(node)
            elif kind == "para":
                self._on_para(node)
        return [p for p in self.pericopes if p.verses]

    def _on_chapter(self, node: dict[str, Any]) -> None:
        match = _VERSE_NUM_RE.search(str(node.get("number", "")))
        if not match:
            return
        self.chapter = int(match.group())
        self.subs = []
        self.current = None
        self.verse = None

    def _on_para(self, node: dict[str, Any]) -> None:
        marker = str(node.get("marker", ""))
        content = node.get("content", [])
        if marker in _IGNORED_MARKERS:
            return
        if marker == "s1":
            self._start_section(_clean(_flatten(content)))
        elif marker in _SUBHEADING_MARKERS:
            text = _clean(_flatten(content))
            if text:
                self._ensure_pericope()
                self.subs.append(text)
        elif marker == "r":
            self._ensure_pericope().cross_refs.extend(_extract_refs(content))
        else:
            if self.verse is not None and self.verse.parts:
                self.verse.parts.append(" ")  # verse continues in a new paragraph block
            for child in content:
                self._inline(child)

    def _inline(self, node: Any) -> None:
        if isinstance(node, str):
            if self.verse is not None:
                self.verse.parts.append(node)
            return
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind == "verse":
            self._on_verse(node)
        elif kind == "note":
            if self.verse is not None:
                text = _clean(_flatten(node.get("content", [])))
                if text:
                    self.verse.footnotes.append(text)
        else:
            for child in node.get("content", []):
                self._inline(child)

    def _on_verse(self, node: dict[str, Any]) -> None:
        match = _VERSE_NUM_RE.search(str(node.get("number", "")))
        if not match:
            return
        number = int(match.group())
        key = (self.chapter, number)
        existing = self.verse_index.get(key)
        if existing is not None:
            self.verse = existing
            return
        pericope = self._ensure_pericope()
        verse = _Verse(number=number, subheadings=tuple(dict.fromkeys(self.subs)))
        pericope.verses.append(verse)
        self.verse_index[key] = verse
        self.verse = verse


def _windows(verses: list[_Verse], size: int, overlap: int) -> list[list[_Verse]]:
    if len(verses) <= size:
        return [verses]
    step = size - overlap
    windows: list[list[_Verse]] = []
    start = 0
    while True:
        window = verses[start : start + size]
        windows.append(window)
        if start + size >= len(verses):
            break
        start += step
    return windows


def _build_chunk(
    book: BookInfo,
    pericope: _Pericope,
    verses: list[_Verse],
    sub_idx: int,
    outline_index: BookOutlineIndex | None,
) -> ChunkDocument:
    chapter = pericope.chapter
    v_start, v_end = verses[0].number, verses[-1].number
    span = f"{v_start}-{v_end}" if v_end != v_start else f"{v_start}"
    canonical_ref = f"{book.name} {chapter}:{span}"

    headings: list[str] = []
    if pericope.heading:
        headings.append(pericope.heading)
    for verse in verses:
        for sub in verse.subheadings:
            if sub not in headings:
                headings.append(sub)
    heading = " / ".join(headings) or None

    outline_path = (
        outline_index.get_outline_breadcrumbs(book.code, chapter, v_start)
        if outline_index
        else None
    )
    cross_refs = list(pericope.cross_refs)
    footnotes = [
        f"[{book.name} {chapter}:{v.number}] {note}" for v in verses for note in v.footnotes
    ]

    header_parts = [
        f"Book: {book.name}",
        f"Passage: {canonical_ref}",
        f"Testament: {book.testament}",
    ]
    if outline_path:
        header_parts.append(f"Outline: {outline_path}")
    if heading:
        header_parts.append(f"Heading: {heading}")
    if cross_refs:
        header_parts.append(f"Cross-Refs: {'; '.join(cross_refs)}")
    lines = [f"[{' | '.join(header_parts)}]"]
    lines.extend(f"[{book.name} {chapter}:{v.number}] {v.text}" for v in verses)
    if footnotes:
        lines.append("Footnotes:")
        lines.extend(f"- {note}" for note in footnotes)

    chunk_id = f"scripture:{book.code}:{chapter}:{v_start}-{v_end}:{sub_idx}"
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        source_type=SourceType.SCRIPTURE,
        book_code=book.code,
        book_name=book.name,
        testament=book.testament,
        canonical_order=book.canonical_order,
        chapter=chapter,
        verse_start=v_start,
        verse_end=v_end,
        canonical_ref=canonical_ref,
        verse_refs=[f"{book.code} {chapter}:{v.number}" for v in verses],
        section_heading=heading,
        outline_path=outline_path,
        cross_references=cross_refs,
        footnotes=footnotes,
    )
    return ChunkDocument(chunk_id=chunk_id, text="\n".join(lines), metadata=metadata)


def parse_usj_book(
    doc: dict[str, Any],
    book: BookInfo,
    outline_index: BookOutlineIndex | None = None,
    max_verses_per_chunk: int = 10,
    verse_overlap: int = 2,
) -> Iterator[ChunkDocument]:
    """Yield chunks for one decoded USJ document."""
    pericopes = _BookWalker().walk(doc.get("content", []))
    for pericope in pericopes:
        for sub_idx, window in enumerate(
            _windows(pericope.verses, max_verses_per_chunk, verse_overlap)
        ):
            yield _build_chunk(book, pericope, window, sub_idx, outline_index)


def iter_usj_scripture_chunks(
    zip_path: str | Path | None = None,
    outline_index: BookOutlineIndex | None = None,
    book_filter: set[str] | None = None,
    max_verses_per_chunk: int = 10,
    verse_overlap: int = 2,
) -> Iterator[ChunkDocument]:
    """Stream scripture chunks for every (or filtered) book in canonical order."""
    if max_verses_per_chunk < 1:
        raise ValueError("max_verses_per_chunk must be >= 1")
    if not 0 <= verse_overlap < max_verses_per_chunk:
        raise ValueError("verse_overlap must be >= 0 and < max_verses_per_chunk")

    path = get_settings().usj_zip_file if zip_path is None else resolve_data_file(zip_path)
    wanted = {code.strip().upper() for code in book_filter} if book_filter else None

    with zipfile.ZipFile(path) as archive:
        entries: list[tuple[int, BookInfo, str]] = []
        for name in archive.namelist():
            if not name.endswith(".usj"):
                continue
            code = Path(name).stem.upper()
            try:
                book = get_book_by_code(code)
            except KeyError:
                continue
            if wanted is not None and code not in wanted:
                continue
            entries.append((book.canonical_order, book, name))

        for _, book, name in sorted(entries, key=lambda e: e[0]):
            doc = json.loads(archive.read(name).decode("utf-8-sig"))
            yield from parse_usj_book(doc, book, outline_index, max_verses_per_chunk, verse_overlap)
