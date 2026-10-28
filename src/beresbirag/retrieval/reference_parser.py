"""Deterministic, zero-LLM Bible reference extraction from natural-language queries."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from beresbirag.models.books import BSB_BOOKS, BookInfo, resolve_book
from beresbirag.models.schemas import VerseRange

MAX_EXPANDED_REFS = 50

# Verses per chapter for every BSB book (derived from bsb_usj.zip, canonical order).
_BSB_VERSE_COUNTS_RAW: dict[str, str] = {
    "GEN": "31 25 24 26 32 22 24 22 29 32 32 20 18 24 21 16 27 33 38 18 34 24 20 67 34 35 46 22 35 43 55 32 20 31 29 43 36 30 23 23 57 38 34 34 28 34 31 22 33 26",
    "EXO": "22 25 22 31 23 30 25 32 35 29 10 51 22 31 27 36 16 27 25 26 36 31 33 18 40 37 21 43 46 38 18 35 23 35 35 38 29 31 43 38",
    "LEV": "17 16 17 35 19 30 38 36 24 20 47 8 59 57 33 34 16 30 37 27 24 33 44 23 55 46 34",
    "NUM": "54 34 51 49 31 27 89 26 23 36 35 16 33 45 41 50 13 32 22 29 35 41 30 25 18 65 23 31 40 16 54 42 56 29 34 13",
    "DEU": "46 37 29 49 33 25 26 20 29 22 32 32 18 29 23 22 20 22 21 20 23 30 25 22 19 19 26 68 29 20 30 52 29 12",
    "JOS": "18 24 17 24 15 27 26 35 27 43 23 24 33 15 63 10 18 28 51 9 45 34 16 33",
    "JDG": "36 23 31 24 31 40 25 35 57 18 40 15 25 20 20 31 13 31 30 48 25",
    "RUT": "22 23 18 22",
    "1SA": "28 36 21 22 12 21 17 22 27 27 15 25 23 52 35 23 58 30 24 42 15 23 29 22 44 25 12 25 11 31 13",
    "2SA": "27 32 39 12 25 23 29 18 13 19 27 31 39 33 37 23 29 33 43 26 22 51 39 25",
    "1KI": "53 46 28 34 18 38 51 66 28 29 43 33 34 31 34 34 24 46 21 43 29 53",
    "2KI": "18 25 27 44 27 33 20 29 37 36 21 21 25 29 38 20 41 37 37 21 26 20 37 20 30",
    "1CH": "54 55 24 43 26 81 40 40 44 14 47 40 14 17 29 43 27 17 19 8 30 19 32 31 31 32 34 21 30",
    "2CH": "17 18 17 22 14 42 22 18 31 19 23 16 22 15 19 14 19 34 11 37 20 12 21 27 28 23 9 27 36 27 21 33 25 33 27 23",
    "EZR": "11 70 13 24 17 22 28 36 15 44",
    "NEH": "11 20 32 23 19 19 73 18 38 39 36 47 31",
    "EST": "22 23 15 17 14 14 10 17 32 3",
    "JOB": "22 13 26 21 27 30 21 22 35 22 20 25 28 22 35 22 16 21 29 29 34 30 17 25 6 14 23 28 25 31 40 22 33 37 16 33 24 41 30 24 34 17",
    "PSA": "6 12 8 8 12 10 17 9 20 18 7 8 6 7 5 11 15 50 14 9 13 31 6 10 22 12 14 9 11 12 24 11 22 22 28 12 40 22 13 17 13 11 5 26 17 11 9 14 20 23 19 9 6 7 23 13 11 11 17 12 8 12 11 10 13 20 7 35 36 5 24 20 28 23 10 12 20 72 13 19 16 8 18 12 13 17 7 18 52 17 16 15 5 23 11 13 12 9 9 5 8 28 22 35 45 48 43 13 31 7 10 10 9 8 18 19 2 29 176 7 8 9 4 8 5 6 5 6 8 8 3 18 3 3 21 26 9 8 24 13 10 7 12 15 21 10 20 14 9 6",
    "PRO": "33 22 35 27 23 35 27 36 18 32 31 28 25 35 33 33 28 24 29 30 31 29 35 34 28 28 27 28 27 33 31",
    "ECC": "18 26 22 16 20 12 29 17 18 20 10 14",
    "SNG": "17 17 11 16 16 13 13 14",
    "ISA": "31 22 26 6 30 13 25 22 21 34 16 6 22 32 9 14 14 7 25 6 17 25 18 23 12 21 13 29 24 33 9 20 24 17 10 22 38 22 8 31 29 25 28 28 25 13 15 22 26 11 23 15 12 17 13 12 21 14 21 22 11 12 19 12 25 24",
    "JER": "19 37 25 31 31 30 34 22 26 25 23 17 27 22 21 21 27 23 15 18 14 30 40 10 38 24 22 17 32 24 40 44 26 22 19 32 21 28 18 16 18 22 13 30 5 28 7 47 39 46 64 34",
    "LAM": "22 22 66 22 22",
    "EZK": "28 10 27 17 17 14 27 18 11 22 25 28 23 23 8 63 24 32 14 49 32 31 49 27 17 21 36 26 21 26 18 32 33 31 15 38 28 23 29 49 26 20 27 31 25 24 23 35",
    "DAN": "21 49 30 37 31 28 28 27 27 21 45 13",
    "HOS": "11 23 5 19 15 11 16 14 17 15 12 14 16 9",
    "JOL": "20 32 21",
    "AMO": "15 16 15 13 27 14 17 14 15",
    "OBA": "21",
    "JON": "17 10 10 11",
    "MIC": "16 13 12 13 15 16 20",
    "NAM": "15 13 19",
    "HAB": "17 20 19",
    "ZEP": "18 15 20",
    "HAG": "15 23",
    "ZEC": "21 13 10 14 11 15 14 23 17 12 17 14 9 21",
    "MAL": "14 17 18 6",
    "MAT": "25 23 17 25 48 34 29 34 38 42 30 50 58 36 39 28 27 35 30 34 46 46 39 51 46 75 66 20",
    "MRK": "45 28 35 41 43 56 37 38 50 52 33 44 37 72 47 20",
    "LUK": "80 52 38 44 39 49 50 56 62 42 54 59 35 35 32 31 37 43 48 47 38 71 56 53",
    "JHN": "51 25 36 54 47 71 53 59 41 42 57 50 38 31 27 33 26 40 42 31 25",
    "ACT": "26 47 26 37 42 15 60 40 43 48 30 25 52 28 41 40 34 28 41 38 40 30 35 27 27 32 44 31",
    "ROM": "32 29 31 25 21 23 25 39 33 21 36 21 14 23 33 27",
    "1CO": "31 16 23 21 13 20 40 13 27 33 34 31 13 40 58 24",
    "2CO": "24 17 18 18 21 18 16 24 15 18 33 21 14",
    "GAL": "24 21 29 31 26 18",
    "EPH": "23 22 21 32 33 24",
    "PHP": "30 30 21 23",
    "COL": "29 23 25 18",
    "1TH": "10 20 13 18 28",
    "2TH": "12 17 18",
    "1TI": "20 15 16 16 25 21",
    "2TI": "18 26 17 22",
    "TIT": "16 15 15",
    "PHM": "25",
    "HEB": "14 18 19 16 14 20 28 13 28 39 40 29 25",
    "JAS": "27 26 18 17 20",
    "1PE": "25 25 22 19 14",
    "2PE": "21 22 18",
    "1JN": "10 29 24 21 21",
    "2JN": "13",
    "3JN": "14",
    "JUD": "25",
    "REV": "20 29 22 11 14 17 17 13 21 11 19 17 18 20 8 21 18 24 21 15 27 21",
}

BSB_VERSE_COUNTS: dict[str, tuple[int, ...]] = {
    code: tuple(int(n) for n in raw.split()) for code, raw in _BSB_VERSE_COUNTS_RAW.items()
}

# Aliases that are ordinary English words: in lowercase they only count as a book
# when followed by an explicit chapter:verse reference.
_COMMON_WORD_ALIASES: frozenset[str] = frozenset(
    {
        "act", "acts", "am", "can", "col", "dan", "ex", "hag", "he", "is", "jam",
        "job", "joe", "jon", "lam", "mar", "mark", "num", "numbers", "judges",
        "pro", "re", "ruth", "song", "songs", "tit",
    }
)  # fmt: skip

_ORDINAL_WORDS: dict[str, str] = {
    "first": "1", "1st": "1", "second": "2", "2nd": "2", "third": "3", "3rd": "3",
}  # fmt: skip
_ORDINAL_PREFIX: dict[str, str] = {
    "1": r"(?:1st|first|1)",
    "2": r"(?:2nd|second|2)",
    "3": r"(?:3rd|third|3)",
}
_DASH = r"(?:\s*[-\u2010\u2011\u2012\u2013\u2014\u2015]\s*|\s+(?:to|through|thru)\s+)"
_VERSE_SEP = r"(?::\s*|\.(?=\d))"


def _normalize(raw: str) -> str:
    cleaned = re.sub(r"[.\-_]+", " ", raw.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _alias_to_pattern(alias: str) -> str:
    tokens = alias.split(" ")
    parts: list[str] = []
    for idx, token in enumerate(tokens):
        if idx == 0 and token in _ORDINAL_PREFIX and len(tokens) > 1:
            parts.append(_ORDINAL_PREFIX[token] + r"\s*")
            continue
        parts.append(re.escape(token))
        if idx < len(tokens) - 1:
            parts.append(r"\s+")
    return "".join(parts)


def _build_book_alternation() -> str:
    aliases: set[str] = set()
    for book in BSB_BOOKS:
        for candidate in (book.code, book.name, book.short_abbr, *book.aliases):
            norm = _normalize(candidate)
            if norm:
                aliases.add(norm)
    # Longest aliases first so "1 corinthians" wins over "1 co" and "co".
    ordered = sorted(aliases, key=lambda a: (-len(a), a))
    return "|".join(_alias_to_pattern(a) for a in ordered)


_BOOK_ALT = _build_book_alternation()

_BOOK_RE = re.compile(rf"(?<![A-Za-z0-9])(?P<book>{_BOOK_ALT})(?![A-Za-z])\.?", re.IGNORECASE)

_REF_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<book>{_BOOK_ALT})\.?\s*"
    r"(?P<c1>\d{1,3})(?!\d)"
    rf"(?:{_VERSE_SEP}(?P<v1>\d{{1,3}})(?!\d)(?:[a-c](?![A-Za-z]))?)?"
    rf"(?:{_DASH}(?P<x>\d{{1,3}})(?!\d)(?:{_VERSE_SEP}(?P<y>\d{{1,3}})(?!\d))?(?:[a-c](?![A-Za-z]))?)?"
    r"(?![\d:])",
    re.IGNORECASE,
)

_CONTINUATION_RE = re.compile(
    r"\s*[,;]\s*(?:and\s+)?"
    r"(?P<c1>\d{1,3})(?!\d)"
    rf"(?:{_VERSE_SEP}(?P<v1>\d{{1,3}})(?!\d))?"
    rf"(?:{_DASH}(?P<x>\d{{1,3}})(?!\d)(?:{_VERSE_SEP}(?P<y>\d{{1,3}})(?!\d))?)?"
    r"(?![\d:A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _RawRef:
    start_chapter: int
    start_verse: int | None
    end_chapter: int
    end_verse: int | None


def chapter_count(book_code: str) -> int:
    """Return the number of chapters of a BSB book (0 if unknown)."""
    return len(BSB_VERSE_COUNTS.get(book_code.upper(), ()))


def verse_count(book_code: str, chapter: int) -> int:
    """Return the number of verses in a BSB chapter (0 if the chapter does not exist)."""
    counts = BSB_VERSE_COUNTS.get(book_code.upper(), ())
    if 1 <= chapter <= len(counts):
        return counts[chapter - 1]
    return 0


def _resolve_book_text(book_text: str) -> BookInfo | None:
    norm = _normalize(book_text)
    first, _, rest = norm.partition(" ")
    if first in _ORDINAL_WORDS and rest:
        norm = f"{_ORDINAL_WORDS[first]} {rest}"
    else:
        match = re.match(r"^(1st|2nd|3rd)(\S+)$", norm)
        if match:
            norm = f"{_ORDINAL_WORDS[match.group(1)]} {match.group(2)}"
    return resolve_book(norm)


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _is_capitalized(book_text: str) -> bool:
    first = book_text.strip()[:1]
    return first.isupper() or first.isdigit()


def _accept_book_text(book_text: str, *, has_verse: bool) -> bool:
    """Reject ambiguous lowercase English words that collide with book aliases."""
    letters = _letters(book_text)
    norm = _normalize(book_text)
    if len(letters) <= 2:
        return _is_capitalized(book_text)
    if has_verse or _is_capitalized(book_text):
        return True
    return len(letters) >= 4 and norm not in _COMMON_WORD_ALIASES


def _display_name(book: BookInfo) -> str:
    return "Psalm" if book.code == "PSA" else book.name


def _format_canonical(book: BookInfo, raw: _RawRef) -> str:
    name = _display_name(book)
    sc, sv, ec, ev = raw.start_chapter, raw.start_verse, raw.end_chapter, raw.end_verse
    if sv is None:
        return f"{name} {sc}" if sc == ec else f"{name} {sc}-{ec}"
    if sc == ec:
        return f"{name} {sc}:{sv}" if sv == ev else f"{name} {sc}:{sv}-{ev}"
    return f"{name} {sc}:{sv}-{ec}:{ev}"


def _iter_verse_keys(book_code: str, raw: _RawRef) -> Iterator[str]:
    for chapter in range(raw.start_chapter, raw.end_chapter + 1):
        first = raw.start_verse if (chapter == raw.start_chapter and raw.start_verse) else 1
        last = verse_count(book_code, chapter)
        if chapter == raw.end_chapter and raw.end_verse:
            last = min(last, raw.end_verse)
        for verse in range(first, last + 1):
            yield f"{book_code} {chapter}:{verse}"


def expand_verse_refs(book_code: str, raw: _RawRef, cap: int = MAX_EXPANDED_REFS) -> list[str]:
    """Expand a validated reference into normalized USJ verse keys (capped)."""
    keys: list[str] = []
    for key in _iter_verse_keys(book_code, raw):
        keys.append(key)
        if len(keys) >= cap:
            break
    return keys


def _normalize_raw(
    book: BookInfo,
    c1: int,
    v1: int | None,
    x: int | None,
    y: int | None,
) -> _RawRef | None:
    """Interpret captured numbers as chapter/verse bounds and validate against BSB versification."""
    n_chapters = chapter_count(book.code)
    if n_chapters == 0:
        return None

    # Single-chapter books ("Jude 3", "Philemon 4-6") cite verses directly.
    if n_chapters == 1 and v1 is None and y is None and not (c1 == 1 and x is None):
        c1, v1 = 1, c1

    if v1 is None and y is None:
        sc, sv, ec, ev = c1, None, (x if x is not None else c1), None
    elif v1 is None:  # "Genesis 1-2:3"
        sc, sv, ec, ev = c1, 1, (x if x is not None else c1), y
    elif x is None:  # "John 3:16"
        sc, sv, ec, ev = c1, v1, c1, v1
    elif y is None:  # "Genesis 1:1-5"
        sc, sv, ec, ev = c1, v1, c1, x
    else:  # "Genesis 1:1-2:3"
        sc, sv, ec, ev = c1, v1, x, y

    if not 1 <= sc <= n_chapters:
        return None
    if sv is not None and not 1 <= sv <= verse_count(book.code, sc):
        return None

    ec = min(ec, n_chapters)
    if ec < sc:
        ec, ev = sc, sv
    if sv is not None:
        if ev is None:
            ev = verse_count(book.code, ec)
        ev = min(ev, verse_count(book.code, ec))
        if ec == sc and ev < sv:
            ev = sv
    return _RawRef(sc, sv, ec, ev)


def _make_range(book: BookInfo, raw: _RawRef) -> VerseRange:
    return VerseRange(
        book_code=book.code,
        book_name=book.name,
        start_chapter=raw.start_chapter,
        start_verse=raw.start_verse,
        end_chapter=raw.end_chapter,
        end_verse=raw.end_verse,
        canonical_ref=_format_canonical(book, raw),
        expanded_refs=expand_verse_refs(book.code, raw),
    )


def _to_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def extract_bible_references(text: str) -> list[VerseRange]:
    """Extract explicit Bible references (verse, range, cross-chapter, whole chapter) from text."""
    if not text:
        return []

    results: list[VerseRange] = []
    seen: set[str] = set()

    def _add(book: BookInfo, raw: _RawRef | None) -> None:
        if raw is None:
            return
        vr = _make_range(book, raw)
        key = f"{vr.book_code}|{vr.canonical_ref}"
        if key not in seen:
            seen.add(key)
            results.append(vr)

    pos = 0
    while True:
        match = _REF_RE.search(text, pos)
        if match is None:
            break
        pos = match.end()
        book_text = match.group("book")
        has_verse = match.group("v1") is not None or match.group("y") is not None
        if not _accept_book_text(book_text, has_verse=has_verse):
            # Retry just after the rejected book token so overlapping matches are not lost.
            pos = match.start("book") + 1
            continue
        book = _resolve_book_text(book_text)
        if book is None:
            pos = match.start("book") + 1
            continue

        raw = _normalize_raw(
            book,
            int(match.group("c1")),
            _to_int(match.group("v1")),
            _to_int(match.group("x")),
            _to_int(match.group("y")),
        )
        _add(book, raw)
        if raw is None:
            continue

        # Continuations sharing the same book: "John 3:16, 18" or "Gen 1:1; 2:4-7".
        ctx_chapter = raw.end_chapter
        ctx_has_verse = raw.start_verse is not None
        while True:
            cont = _CONTINUATION_RE.match(text, pos)
            if cont is None:
                break
            sep_end = cont.start("c1")
            if _BOOK_RE.match(text, sep_end):
                break
            c1 = int(cont.group("c1"))
            v1 = _to_int(cont.group("v1"))
            x = _to_int(cont.group("x"))
            y = _to_int(cont.group("y"))
            if v1 is None and ctx_has_verse and y is None:
                # Bare numbers after a verse reference are verses in the current chapter.
                cont_raw = _normalize_raw(book, ctx_chapter, c1, x, None)
            else:
                cont_raw = _normalize_raw(book, c1, v1, x, y)
            if cont_raw is None:
                break
            _add(book, cont_raw)
            ctx_chapter = cont_raw.end_chapter
            ctx_has_verse = cont_raw.start_verse is not None
            pos = cont.end()

    return results


def find_book_mentions(text: str) -> list[BookInfo]:
    """Detect Bible book names mentioned anywhere in the text (with or without chapter numbers)."""
    if not text:
        return []
    found: list[BookInfo] = []
    for match in _BOOK_RE.finditer(text):
        book_text = match.group("book")
        letters = _letters(book_text)
        norm = _normalize(book_text)
        if len(letters) < 3:
            continue
        if not _is_capitalized(book_text) and (len(letters) < 4 or norm in _COMMON_WORD_ALIASES):
            continue
        book = _resolve_book_text(book_text)
        if book is not None and book not in found:
            found.append(book)
    return found
