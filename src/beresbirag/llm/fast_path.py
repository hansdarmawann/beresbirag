"""Deterministic verbatim passage and concordance-count answers."""

from __future__ import annotations

import re

from beresbirag.models.books import resolve_book
from beresbirag.models.schemas import QueryIntent, RetrievalResult, SourceType
from beresbirag.retrieval.query_router import QueryPlan
from beresbirag.retrieval.reference_parser import (
    _CONTINUATION_RE,
    _REF_RE,
    extract_bible_references,
    verse_count,
)

_FILLERS = set(
    "read show quote display give me what does do say says text of the verse verses passage please".split()
)
_VERSE_LINE = re.compile(r"^\[(.+?) (\d+):(\d+)\]\s+.+$")


def _bare_reference(query: str) -> bool:
    # Reuse the parser's aliases and range syntax, but remove only valid matches.
    remaining = []
    position = 0
    for match in _REF_RE.finditer(query):
        if match.start() < position or not extract_bible_references(match.group()):
            continue
        end = match.end()
        parsed = extract_bible_references(match.group())
        while continuation := _CONTINUATION_RE.match(query, end):
            extended = extract_bible_references(query[match.start() : continuation.end()])
            if len(extended) <= len(parsed):
                break
            end = continuation.end()
            parsed = extended
        remaining.append(query[position : match.start()])
        position = end
    remaining.append(query[position:])
    return all(word.lower() in _FILLERS for word in re.findall(r"\w+", " ".join(remaining)))


def try_fast_answer(plan: QueryPlan, results: list[RetrievalResult]) -> str | None:
    if plan.intent == QueryIntent.EXACT_PASSAGE and _bare_reference(plan.original_query):
        for vr in plan.verse_ranges:
            count = sum(
                (
                    vr.end_verse
                    if chapter == vr.end_chapter and vr.end_verse is not None
                    else verse_count(vr.book_code, chapter)
                )
                - (
                    vr.start_verse
                    if chapter == vr.start_chapter and vr.start_verse is not None
                    else 1
                )
                + 1
                for chapter in range(vr.start_chapter, vr.end_chapter + 1)
            )
            if count > len(vr.expanded_refs):
                return (
                    None  # The parser caps expansion; do not present a truncated range as complete.
                )
        requested = set(plan.expanded_refs)
        verses: dict[str, tuple[tuple[int, int, int], str]] = {}
        for result in results:
            if result.chunk.metadata.source_type != SourceType.SCRIPTURE:
                continue
            for line in result.chunk.text.splitlines():
                match = _VERSE_LINE.match(line.strip())
                if not match:
                    continue
                book = resolve_book(match[1])
                if book is None:
                    continue
                chapter, verse = int(match[2]), int(match[3])
                key = f"{book.code} {chapter}:{verse}"
                if key in requested:
                    verses.setdefault(key, ((book.canonical_order, chapter, verse), line.strip()))
        # Do not label a partially retrieved passage as a complete quotation.
        if not requested or requested - verses.keys():
            return None
        header = "; ".join(vr.canonical_ref for vr in plan.verse_ranges)
        return f"**{header}** (BSB)\n\n" + "\n\n".join(line for _, line in sorted(verses.values()))
    if plan.intent == QueryIntent.CONCORDANCE and re.search(
        r"how many (?:times|occurrences)", plan.original_query, re.I
    ):
        for result in results:
            meta = result.chunk.metadata
            if (
                meta.source_type == SourceType.CONCORDANCE
                and plan.concordance_word
                and meta.concordance_word == plan.concordance_word
                and meta.concordance_occurrences is not None
            ):
                word = plan.concordance_word
                distribution = next(
                    (
                        line
                        for line in result.chunk.text.splitlines()
                        if line.startswith("Book Distribution:")
                    ),
                    "",
                )
                return (
                    f'"{word}" occurs {meta.concordance_occurrences} times in the BSB '
                    f"[BSB Concordance: {word}].\n\n{distribution}"
                ).strip()
    return None
