"""Parse the BSB book outlines PDF into hierarchical outline nodes and chunk documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from beresbirag.config import get_settings, resolve_data_file
from beresbirag.models.books import BSB_BOOKS, BookInfo, get_book_by_code, resolve_book
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; `fitz` is the legacy alias.
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

FIRST_OUTLINE_PAGE_INDEX = 2
LARGE_BOOK_NODE_THRESHOLD = 25
MAX_VERSE = 999

_NODE_RE = re.compile(r"^(?P<marker>\d+|[a-z]+)\.\s+(?P<title>.*)\(\s*(?P<range>[^()]*)\)\s*$")
_NODE_START_RE = re.compile(r"^(?:\d+|[a-z]+)\.\s+\S")
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


@dataclass(slots=True)
class OutlineNode:
    """A single entry of a book outline."""

    book_code: str
    book_name: str
    level: int
    marker: str
    title: str
    raw_range: str
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    breadcrumb: str = ""

    def contains(self, chapter: int, verse: int) -> bool:
        return (
            (self.start_chapter, self.start_verse)
            <= (chapter, verse)
            <= (
                self.end_chapter,
                self.end_verse,
            )
        )

    def contains_node(self, other: OutlineNode) -> bool:
        return self.contains(other.start_chapter, other.start_verse) and self.contains(
            other.end_chapter, other.end_verse
        )

    @property
    def label(self) -> str:
        return f"{self.marker} {self.title}"


def _roman_value(marker: str) -> int | None:
    if not marker or any(ch not in _ROMAN for ch in marker):
        return None
    total = 0
    for idx, ch in enumerate(marker):
        value = _ROMAN[ch]
        if idx + 1 < len(marker) and value < _ROMAN[marker[idx + 1]]:
            total -= value
        else:
            total += value
    return total


def _letter_value(marker: str) -> int | None:
    return ord(marker) - ord("a") + 1 if len(marker) == 1 and marker.isalpha() else None


def parse_verse_range(raw: str, single_chapter: bool) -> tuple[int, int, int, int] | None:
    """Parse '1:1–2:25', '1:1–31', '9:31', or single-chapter '1–25' into 4 ints."""
    cleaned = re.sub(r"[^\d:]+", "-", raw).strip("-")
    if not cleaned:
        return None
    parts = cleaned.split("-")
    try:
        if single_chapter and ":" not in cleaned:
            start = int(parts[0])
            end = int(parts[-1])
            return 1, start, 1, end
        first = parts[0].split(":")
        if len(first) == 1:  # chapter-only form, e.g. '3' or '3–5'
            start_ch = int(first[0])
            end_ch = int(parts[-1].split(":")[0])
            return start_ch, 1, end_ch, MAX_VERSE
        start_ch, start_v = int(first[0]), int(first[1])
        if len(parts) == 1:
            return start_ch, start_v, start_ch, start_v
        last = parts[-1].split(":")
        if len(last) == 2:
            return start_ch, start_v, int(last[0]), int(last[1])
        return start_ch, start_v, start_ch, int(last[0])
    except (ValueError, IndexError):
        return None


class _LevelTracker:
    """Disambiguates outline marker levels (1./a./i./1.) using sequence and verse ranges."""

    def __init__(self) -> None:
        self.top = 0
        self.letter = 0
        self.roman = 0
        self.nested = 0
        self.last_non_top: OutlineNode | None = None
        self.last_level2: OutlineNode | None = None

    def resolve(self, marker: str, span: tuple[int, int, int, int]) -> int:
        probe = OutlineNode("", "", 0, marker, "", "", *span)
        candidates: list[int] = []
        if marker.isdigit():
            number = int(marker)
            if number == self.top + 1:
                candidates.append(1)
            if self.last_non_top is not None and number == self.nested + 1:
                candidates.append(4)
            if not candidates:
                candidates.append(4 if self.last_non_top and number > 1 else 1)
        else:
            if _letter_value(marker) == self.letter + 1:
                candidates.append(2)
            if _roman_value(marker) == self.roman + 1 and self.last_level2 is not None:
                candidates.append(3)
            if not candidates:
                candidates.append(3 if _roman_value(marker) and self.last_level2 else 2)
        if len(candidates) == 1:
            return candidates[0]
        if self.last_non_top is not None and self.last_non_top.contains_node(probe):
            return candidates[-1]
        return candidates[0]

    def update(self, level: int, marker: str, node: OutlineNode) -> None:
        if level == 1:
            self.top = int(marker) if marker.isdigit() else self.top + 1
            self.letter = self.roman = self.nested = 0
            self.last_non_top = None
            self.last_level2 = None
        elif level == 2:
            self.letter = _letter_value(marker) or self.letter + 1
            self.roman = self.nested = 0
            self.last_non_top = node
            self.last_level2 = node
        elif level == 3:
            self.roman = _roman_value(marker) or self.roman + 1
            self.nested = 0
            self.last_non_top = node
        else:
            self.nested = int(marker) if marker.isdigit() else self.nested + 1


class BookOutlineIndex:
    """Hierarchical outline index for all BSB books."""

    def __init__(self, nodes_by_book: dict[str, list[OutlineNode]] | None = None) -> None:
        self.nodes_by_book: dict[str, list[OutlineNode]] = nodes_by_book or {}

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_pdf(cls, pdf_path: str | Path | None = None) -> BookOutlineIndex:
        path = get_settings().outlines_pdf_file if pdf_path is None else resolve_data_file(pdf_path)
        with fitz.open(str(path)) as doc:
            lines: list[str] = []
            for page_idx in range(FIRST_OUTLINE_PAGE_INDEX, len(doc)):
                lines.extend(doc[page_idx].get_text().splitlines())
        return cls(cls._parse_lines(lines))

    @staticmethod
    def _split_books(lines: list[str]) -> list[tuple[BookInfo, list[str]]]:
        cleaned = [ln.replace("\u202f", " ").replace("\u2060", "").strip() for ln in lines]
        headers: list[int] = [
            i
            for i in range(len(cleaned) - 1)
            if cleaned[i] and cleaned[i + 1].lower() == "outline" and "(" not in cleaned[i]
        ]
        books: list[tuple[BookInfo, list[str]]] = []
        for pos, start in enumerate(headers):
            end = headers[pos + 1] if pos + 1 < len(headers) else len(cleaned)
            book = resolve_book(cleaned[start])
            if book is None and len(headers) == len(BSB_BOOKS):
                book = BSB_BOOKS[pos]
            if book is None:
                continue
            books.append((book, cleaned[start + 2 : end]))
        return books

    @classmethod
    def _parse_lines(cls, lines: list[str]) -> dict[str, list[OutlineNode]]:
        result: dict[str, list[OutlineNode]] = {}
        for book, body in cls._split_books(lines):
            result[book.code] = cls._parse_book(book, body)
        return result

    @staticmethod
    def _join_entries(body: list[str]) -> list[str]:
        """Join wrapped outline entries so each entry ends with a closing parenthesis."""
        entries: list[str] = []
        pending: str | None = None
        for line in body:
            if not line:
                continue
            if pending is not None:
                if _NODE_START_RE.match(line) and ")" in pending:
                    entries.append(pending)
                    pending = line
                else:
                    pending = f"{pending} {line}"
            elif _NODE_START_RE.match(line):
                pending = line
            else:
                continue
            if pending.rstrip().endswith(")"):
                entries.append(pending)
                pending = None
        if pending is not None:
            entries.append(pending)
        return entries

    @classmethod
    def _parse_book(cls, book: BookInfo, body: list[str]) -> list[OutlineNode]:
        single_chapter = book.code in {"OBA", "PHM", "2JN", "3JN", "JUD"}
        tracker = _LevelTracker()
        stack: list[OutlineNode] = []
        nodes: list[OutlineNode] = []
        for entry in cls._join_entries(body):
            match = _NODE_RE.match(re.sub(r"\s+", " ", entry))
            if not match:
                continue
            span = parse_verse_range(match["range"], single_chapter)
            if span is None:
                continue
            marker = match["marker"]
            level = tracker.resolve(marker, span)
            raw_range = re.sub(r"[^\d:]+", "\u2013", match["range"]).strip("\u2013")
            node = OutlineNode(
                book_code=book.code,
                book_name=book.name,
                level=level,
                marker=f"{marker}.",
                title=match["title"].strip(),
                raw_range=raw_range,
                start_chapter=span[0],
                start_verse=span[1],
                end_chapter=span[2],
                end_verse=span[3],
            )
            tracker.update(level, marker, node)
            del stack[level - 1 :]
            stack.append(node)
            node.breadcrumb = " > ".join(n.label for n in stack)
            nodes.append(node)
        return nodes

    # ------------------------------------------------------------------ lookup
    def books(self) -> list[str]:
        return sorted(self.nodes_by_book, key=lambda c: get_book_by_code(c).canonical_order)

    def get_outline_breadcrumbs(self, book_code: str, chapter: int, verse: int) -> str | None:
        """Return the breadcrumb of the deepest (then narrowest) node containing the verse."""
        best: OutlineNode | None = None
        for node in self.nodes_by_book.get(book_code.upper(), []):
            if not node.contains(chapter, verse):
                continue
            if best is None or node.level > best.level:
                best = node
            elif node.level == best.level and _span_size(node) < _span_size(best):
                best = node
        return best.breadcrumb if best else None

    # ------------------------------------------------------------------ chunks
    def to_chunk_documents(self) -> list[ChunkDocument]:
        chunks: list[ChunkDocument] = []
        for code in self.books():
            nodes = self.nodes_by_book[code]
            if not nodes:
                continue
            book = get_book_by_code(code)
            first, last = nodes[0], max(nodes, key=lambda n: (n.end_chapter, n.end_verse))
            overview_range = f"{first.start_chapter}:{first.start_verse}\u2013{last.end_chapter}:{last.end_verse}"
            chunks.append(
                self._make_chunk(
                    book,
                    f"outline:{code}:overview",
                    "Complete Outline",
                    overview_range,
                    nodes,
                    first,
                    last,
                )
            )
            if len(nodes) > LARGE_BOOK_NODE_THRESHOLD:
                for idx, group in enumerate(_top_level_groups(nodes)):
                    head = group[0]
                    chunks.append(
                        self._make_chunk(
                            book,
                            f"outline:{code}:sec:{idx}",
                            head.label,
                            head.raw_range,
                            group,
                            head,
                            head,
                        )
                    )
        return chunks

    @staticmethod
    def _make_chunk(
        book: BookInfo,
        chunk_id: str,
        division: str,
        raw_range: str,
        nodes: list[OutlineNode],
        first: OutlineNode,
        last: OutlineNode,
    ) -> ChunkDocument:
        base_level = min(n.level for n in nodes)
        body = "\n".join(
            f"{'  ' * (n.level - base_level)}{n.marker} {n.title} ({n.raw_range})" for n in nodes
        )
        header = (
            f"[Source: BSB Book Outline | Book: {book.name} ({book.code}) | "
            f"Division: {division} ({raw_range})]"
        )
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source_type=SourceType.BOOK_OUTLINE,
            book_code=book.code,
            book_name=book.name,
            testament=book.testament,
            canonical_order=book.canonical_order,
            chapter=first.start_chapter,
            verse_start=first.start_verse,
            verse_end=last.end_verse,
            canonical_ref=f"{book.name} Outline ({raw_range})",
        )
        return ChunkDocument(chunk_id=chunk_id, text=f"{header}\n{body}", metadata=metadata)


def _span_size(node: OutlineNode) -> int:
    return (node.end_chapter - node.start_chapter) * 1000 + (node.end_verse - node.start_verse)


def _top_level_groups(nodes: list[OutlineNode]) -> list[list[OutlineNode]]:
    groups: list[list[OutlineNode]] = []
    for node in nodes:
        if node.level == 1 or not groups:
            groups.append([node])
        else:
            groups[-1].append(node)
    return groups


def parse_book_outlines(
    pdf_path: str | Path | None = None,
) -> tuple[BookOutlineIndex, list[ChunkDocument]]:
    """Parse the outlines PDF and return the index together with its chunk documents."""
    index = BookOutlineIndex.from_pdf(pdf_path)
    return index, index.to_chunk_documents()
