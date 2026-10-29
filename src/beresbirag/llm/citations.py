"""Deterministic validation of bracket citations in RAG answers against the retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from beresbirag.models.books import resolve_book
from beresbirag.models.schemas import CitationReport, RetrievalResult, SourceType
from beresbirag.retrieval.hybrid_retriever import _topic_variants
from beresbirag.retrieval.reference_parser import extract_bible_references, find_book_mentions

CitationStatus = Literal["valid", "partial", "unsupported", "unparsed"]
LabelKind = Literal["topic", "concordance", "outline"]

_BRACKET_RE = re.compile(r"\[([^\[\]]{1,120})\]")
_NUMERIC_RE = re.compile(r"^(?:source\s*)?(\d{1,3})$", re.IGNORECASE)
_LABEL_RE = re.compile(
    r"^(?:BSB\s+)?(?P<kind>Topical\s+Index|Concordance|Book\s+Outline)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)
# Outline chunk canonical_ref format, e.g. "Romans Outline (1:1-16:27)".
_OUTLINE_REF_RE = re.compile(r"^(?:BSB\s+)?(?P<value>.+?)\s+Outline\b.*$", re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
_BSB_PREFIX_RE = re.compile(r"^BSB\s+", re.IGNORECASE)

_LABEL_KINDS: dict[str, LabelKind] = {
    "topical index": "topic",
    "concordance": "concordance",
    "book outline": "outline",
}


def extract_citations(answer: str) -> list[str]:
    """Return the contents of every [bracket] in the answer, deduplicated in order."""
    seen: dict[str, None] = {}
    for match in _BRACKET_RE.finditer(answer or ""):
        content = match.group(1).strip()
        if content:
            seen.setdefault(content, None)
    return list(seen)


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", _BSB_PREFIX_RE.sub("", text.strip())).lower()


def parse_label(content: str) -> tuple[LabelKind, str] | None:
    """Parse '[BSB Topical Index: X]'-style labels into (kind, normalized value).

    Outline values are normalized to the USJ book code; '(Part 2)' / '(26 Occurrences)'
    suffixes are dropped so labels copied from chunk headers still match.
    """
    match = _LABEL_RE.match(content.strip())
    if match:
        kind = _LABEL_KINDS[re.sub(r"\s+", " ", match["kind"].lower())]
        value = match["value"]
    else:
        outline = _OUTLINE_REF_RE.match(content.strip())
        if outline is None or resolve_book(outline["value"]) is None:
            return None
        kind, value = "outline", outline["value"]
    value = _TRAILING_PAREN_RE.sub("", value).strip().strip("\"'“”").lower()
    if not value:
        return None
    if kind == "outline":
        book = resolve_book(value) or next(iter(find_book_mentions(value.title())), None)
        return ("outline", book.code) if book else None
    return kind, value


def _label_matches(kind: LabelKind, value: str, result: RetrievalResult) -> bool:
    meta = result.chunk.metadata
    if kind == "topic":
        return meta.source_type == SourceType.TOPICAL_INDEX and (
            (meta.topic or "").lower() in _topic_variants(value)
        )
    if kind == "concordance":
        return (
            meta.source_type == SourceType.CONCORDANCE
            and (meta.concordance_word or "").lower() == value
        )
    return meta.source_type == SourceType.BOOK_OUTLINE and meta.book_code == value


def _looks_like_scripture(content: str) -> bool:
    return bool(re.search(r"\d", content)) and bool(find_book_mentions(content))


@dataclass
class _Classified:
    status: CitationStatus
    chunk_ids: list[str] = field(default_factory=list)


def _classify(content: str, results: list[RetrievalResult]) -> _Classified:
    numeric = _NUMERIC_RE.match(content)
    if numeric:
        index = int(numeric.group(1))
        if 1 <= index <= len(results):
            return _Classified("valid", [results[index - 1].chunk.chunk_id])
        return _Classified("unsupported")

    # A citation copied verbatim from a source header is supported by that source.
    normalized = _normalize_label(content)
    exact = [
        r.chunk.chunk_id
        for r in results
        if r.chunk.metadata.canonical_ref
        and _normalize_label(r.chunk.metadata.canonical_ref) == normalized
    ]
    if exact:
        return _Classified("valid", exact)

    label = parse_label(content)
    if label is not None:
        kind, value = label
        supporting = [r.chunk.chunk_id for r in results if _label_matches(kind, value, r)]
        return _Classified("valid" if supporting else "unsupported", supporting)

    refs = extract_bible_references(content)
    # Fabricated segments ("John 3:16; Revelation 99:99") must not hide behind a valid one.
    fabricated = sum(
        1
        for segment in re.split(r"\s*;\s*", content)
        if _looks_like_scripture(segment) and not extract_bible_references(segment)
    )
    if not refs:
        return _Classified(
            "unsupported" if fabricated or _looks_like_scripture(content) else "unparsed"
        )

    retrieved_keys = {key.upper() for r in results for key in r.chunk.metadata.verse_refs}
    statuses: list[CitationStatus] = ["unsupported"] * fabricated
    wanted: set[str] = set()
    for vr in refs:
        keys = {key.upper() for key in vr.expanded_refs}
        wanted |= keys
        found = keys & retrieved_keys
        statuses.append("valid" if found == keys else "partial" if found else "unsupported")
    supporting = [
        r.chunk.chunk_id
        for r in results
        if wanted.intersection(key.upper() for key in r.chunk.metadata.verse_refs)
    ]
    if all(status == "valid" for status in statuses):
        return _Classified("valid", supporting)
    if all(status == "unsupported" for status in statuses):
        return _Classified("unsupported")
    return _Classified("partial", supporting)


def validate_citations(answer: str, results: list[RetrievalResult]) -> CitationReport:
    """Classify each bracket citation in `answer` as valid, partial, unsupported or unparsed."""
    buckets: dict[CitationStatus, list[str]] = {
        "valid": [], "partial": [], "unsupported": [], "unparsed": [],
    }  # fmt: skip
    used: dict[str, None] = {}
    for content in extract_citations(answer):
        classified = _classify(content, results)
        buckets[classified.status].append(content)
        if classified.status in ("valid", "partial"):
            for chunk_id in classified.chunk_ids:
                used.setdefault(chunk_id, None)
    return CitationReport(
        valid=buckets["valid"],
        partial=buckets["partial"],
        unsupported=buckets["unsupported"],
        unparsed=buckets["unparsed"],
        used_chunk_ids=list(used),
        has_citations=bool(buckets["valid"] or buckets["partial"] or buckets["unsupported"]),
    )


def strip_unsupported_citations(answer: str, unsupported: list[str]) -> str:
    """Remove unsupported [citations] from the answer and append a visible warning footer."""
    if not unsupported:
        return answer
    targets = set(unsupported)

    def _drop(match: re.Match[str]) -> str:
        return "" if match.group(1).strip() in targets else match.group(0)

    cleaned = re.sub(r"[ \t]*" + _BRACKET_RE.pattern, _drop, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip()
    listed = ", ".join(f"[{item}]" for item in unsupported)
    return (
        f"{cleaned}\n\n> ⚠️ Removed {len(unsupported)} citation(s) not found in the "
        f"retrieved BSB context: {listed}"
    )
