"""Deterministic (zero-LLM) query intent router for the BSB hybrid retrieval pipeline."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from beresbirag.models.schemas import QueryIntent, SourceType, VerseRange
from beresbirag.retrieval.reference_parser import extract_bible_references, find_book_mentions

ALL_SOURCES: tuple[SourceType, ...] = (
    SourceType.SCRIPTURE,
    SourceType.TOPICAL_INDEX,
    SourceType.BOOK_OUTLINE,
    SourceType.CONCORDANCE,
)

_OUTLINE_RE = re.compile(
    r"\boutlines?\b|\bstructure\b|\bmain\s+divisions?\b|\bsections\s+of\s+the\s+book\s+of\b"
    r"|\bmacro[\s-]+structure\b",
    re.IGNORECASE,
)
_CONCORDANCE_RE = re.compile(
    r"\bconcordance\b|\bhow\s+many\s+occurrences\b|\bhow\s+many\s+times\s+(?:does|is)\s+the\s+word\b"
    r"|\boccurrences?\s+of\b",
    re.IGNORECASE,
)
_TOPICAL_RE = re.compile(
    r"\btopical\s+index\b|\btopics?\b|\bwhat\s+verses\s+are\s+about\b|\bverses\s+on\s+the\s+topic\s+of\b",
    re.IGNORECASE,
)

_QUOTED_RE = re.compile(
    r"\"([^\"]+)\"|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019|(?<![A-Za-z])'([^']+)'(?![A-Za-z])"
)

_WORD = r"[A-Za-z0-9][\w'\u2019-]*"
_CONCORDANCE_WORD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\boccurrences?\s+of\s+(?:the\s+)?(?:word\s+)?(?P<w>{_WORD})", re.IGNORECASE),
    re.compile(rf"\bthe\s+word\s+(?P<w>{_WORD})", re.IGNORECASE),
    re.compile(rf"\bword\s+(?P<w>{_WORD})", re.IGNORECASE),
    re.compile(
        rf"\bconcordance\s+(?:entry\s+|entries\s+|results?\s+)?(?:for|of|on)\s+(?:the\s+)?(?:word\s+)?(?P<w>{_WORD})",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?P<w>{_WORD})\s+(?:in\s+the\s+)?concordance\b", re.IGNORECASE),
)

_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bverses\s+on\s+the\s+topic\s+of\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+verses\s+are\s+about\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(
        r"\btopic(?:al\s+index)?(?:\s+(?:entry|entries|list|references?))?\s*(?:of|for|on|about|:)\s+(?P<k>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bverses\s+(?:are\s+)?(?:about|on|regarding|concerning)\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"\b(?:about|regarding|concerning)\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"(?:^|\bthe\s+)(?P<k>[A-Za-z][\w' -]*?)\s+topic\b", re.IGNORECASE),
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "for", "on", "in", "is", "word", "words", "does",
        "do", "appear", "appears", "occur", "occurs", "times", "many", "how", "bible",
        "bsb", "index", "topical", "topic", "concordance", "entry", "what", "which",
    }
)  # fmt: skip

_TOPIC_TRAILING_RE = re.compile(
    r"\s+(?:in|from|within|according\s+to|of)\s+(?:the\s+)?"
    r"(?:bible|bsb|scriptures?|topical\s+index|berean\s+standard\s+bible)\b.*$",
    re.IGNORECASE,
)
_TOPIC_LEADING_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


class QueryPlan(BaseModel):
    """Deterministic retrieval plan derived from a natural-language query."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    intent: QueryIntent = QueryIntent.HYBRID_GENERAL
    verse_ranges: list[VerseRange] = Field(default_factory=list)
    book_codes: list[str] = Field(default_factory=list)
    target_sources: list[SourceType] = Field(default_factory=lambda: list(ALL_SOURCES))
    concordance_word: str | None = None
    topic_keyword: str | None = None

    @property
    def expanded_refs(self) -> list[str]:
        """All normalized verse keys requested by the plan (deduplicated, in order)."""
        refs: list[str] = []
        seen: set[str] = set()
        for vr in self.verse_ranges:
            for ref in vr.expanded_refs:
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
        return refs

    @property
    def has_exact_targets(self) -> bool:
        """True when the plan carries payload values that can be matched exactly in Qdrant."""
        return bool(self.verse_ranges or self.concordance_word or self.topic_keyword)


def _first_quoted(query: str) -> str | None:
    for match in _QUOTED_RE.finditer(query):
        value = next((g for g in match.groups() if g), None)
        if value and value.strip():
            return value.strip()
    return None


def _clean_token(word: str) -> str | None:
    cleaned = word.strip().strip("'\u2019\"-").lower()
    if not cleaned or cleaned in _STOPWORDS:
        return None
    return cleaned


def extract_concordance_word(query: str) -> str | None:
    """Extract the concordance headword (quoted word or '<occurrences of|word> X' pattern)."""
    quoted = _first_quoted(query)
    if quoted:
        return quoted.lower()
    for pattern in _CONCORDANCE_WORD_PATTERNS:
        for match in pattern.finditer(query):
            word = _clean_token(match.group("w"))
            if word:
                return word
    return None


def _clean_topic(raw: str) -> str | None:
    text = re.split(r"[?!.;,]", raw, maxsplit=1)[0]
    text = _TOPIC_TRAILING_RE.sub("", text)
    text = re.sub(r"\s+(?:say|says|said|teach|teaches)\s*$", "", text, flags=re.IGNORECASE)
    previous = None
    while previous != text:
        previous = text
        text = _TOPIC_LEADING_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not text or text in _STOPWORDS:
        return None
    return text


def extract_topic_keyword(query: str) -> str | None:
    """Extract the topical-index keyword from a topical query."""
    quoted = _first_quoted(query)
    if quoted:
        return quoted.lower()
    for pattern in _TOPIC_PATTERNS:
        match = pattern.search(query)
        if match:
            topic = _clean_topic(match.group("k"))
            if topic:
                return topic
    return None


def route_query(query: str) -> QueryPlan:
    """Classify a query into a retrieval plan using deterministic rules (no LLM calls)."""
    text = (query or "").strip()

    # Rule 1: explicit Bible references -> exact passage lookup.
    verse_ranges = extract_bible_references(text)
    if verse_ranges:
        book_codes = list(dict.fromkeys(vr.book_code for vr in verse_ranges))
        return QueryPlan(
            original_query=query,
            intent=QueryIntent.EXACT_PASSAGE,
            verse_ranges=verse_ranges,
            book_codes=book_codes,
            target_sources=[SourceType.SCRIPTURE, SourceType.BOOK_OUTLINE],
        )

    # Rule 2: book outline / macro structure questions.
    if _OUTLINE_RE.search(text):
        return QueryPlan(
            original_query=query,
            intent=QueryIntent.OUTLINE,
            book_codes=[book.code for book in find_book_mentions(text)],
            target_sources=[SourceType.BOOK_OUTLINE, SourceType.SCRIPTURE],
        )

    # Rule 3: concordance / word-frequency questions.
    if _CONCORDANCE_RE.search(text):
        return QueryPlan(
            original_query=query,
            intent=QueryIntent.CONCORDANCE,
            concordance_word=extract_concordance_word(text),
            target_sources=[SourceType.CONCORDANCE, SourceType.SCRIPTURE],
        )

    # Rule 4: topical index questions.
    if _TOPICAL_RE.search(text):
        return QueryPlan(
            original_query=query,
            intent=QueryIntent.TOPICAL,
            topic_keyword=extract_topic_keyword(text),
            target_sources=[SourceType.TOPICAL_INDEX, SourceType.SCRIPTURE],
        )

    # Rule 5: general hybrid search across all four corpora.
    return QueryPlan(
        original_query=query,
        intent=QueryIntent.HYBRID_GENERAL,
        target_sources=list(ALL_SOURCES),
    )

