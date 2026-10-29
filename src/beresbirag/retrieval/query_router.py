"""Deterministic (zero-LLM) query intent router for the BSB hybrid retrieval pipeline."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from beresbirag.models.books import resolve_book
from beresbirag.models.schemas import QueryIntent, SourceType, VerseRange
from beresbirag.retrieval.reference_parser import (
    _BOOK_RE,
    _CONTINUATION_RE,
    _REF_RE,
    extract_bible_references,
    find_book_mentions,
)

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
_CONCORDANCE_EXTRA_RE = re.compile(
    r"\bwhere\s+does\s+the\s+word\s+\S+.*?\bappears?\b"
    r"|\bhow\s+often\s+does\s+(?:the\s+word\s+)?\S+\s+(?:appear|occur)"
    r"|\bthe\s+word\s+(?:[\"“][^\"”]+[\"”]|(?<![A-Za-z])'[^']+'(?![A-Za-z]))",
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
    re.compile(
        rf"\bhow\s+often\s+does\s+(?:the\s+word\s+)?(?P<w>{_WORD})\s+(?:appear|occur)",
        re.IGNORECASE,
    ),
)

_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bverses\s+on\s+the\s+topic\s+of\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"\bwhat\s+verses\s+are\s+about\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(
        r"\btopic(?:al\s+index)?(?:\s+(?:entry|entries|list|references?))?\s*(?:of|for|on|about|:)\s+(?P<k>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bverses\s+(?:are\s+)?(?:about|on|regarding|concerning)\s+(?P<k>.+)", re.IGNORECASE
    ),
    re.compile(r"\b(?:about|regarding|concerning)\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"(?:^|\bthe\s+)(?P<k>[A-Za-z][\w' -]*?)\s+topic\b", re.IGNORECASE),
    # "the significance of 666 in the BSB Topical Index"
    re.compile(
        r"\b(?:of|for|on|about)\s+(?P<k>[\w'’ -]+?)\s+in\s+the\s+(?:bsb\s+)?topical\s+index\b",
        re.IGNORECASE,
    ),
)

# Topical questions that never say "topic", e.g. "What does the Bible say about grace?".
_IMPLICIT_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bwhat\s+does\s+(?:the\s+)?(?:bible|scriptures?)\s+say\s+about\s+(?P<k>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+does\s+(?:the\s+)?(?:bible|scriptures?)\s+teach\s+(?:about|on)\s+(?P<k>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"\btell\s+me\s+about\s+(?P<k>.+)", re.IGNORECASE),
    re.compile(r"\b(?:bible\s+)?verses?\s+(?:about|on|for)\s+(?P<k>.+)", re.IGNORECASE),
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
_TOPIC_DANGLING_RE = re.compile(
    r"\s+(?:in|from|of|on|at|for|and|or|to|within|with|by)$", re.IGNORECASE
)
_BOOK_CONTEXT = (
    r"\s+(?:in|from|within)\s+(?:the\s+)?(?:(?:book|gospel|letter|epistle)s?\s+(?:of\s+)?)?"
)

# Primary-intent precedence; also the order of secondary intents.
INTENT_PRECEDENCE: tuple[QueryIntent, ...] = (
    QueryIntent.EXACT_PASSAGE,
    QueryIntent.OUTLINE,
    QueryIntent.CONCORDANCE,
    QueryIntent.TOPICAL,
)
INTENT_SOURCES: dict[QueryIntent, tuple[SourceType, ...]] = {
    QueryIntent.EXACT_PASSAGE: (SourceType.SCRIPTURE, SourceType.BOOK_OUTLINE),
    QueryIntent.OUTLINE: (SourceType.BOOK_OUTLINE, SourceType.SCRIPTURE),
    QueryIntent.CONCORDANCE: (SourceType.CONCORDANCE, SourceType.SCRIPTURE),
    QueryIntent.TOPICAL: (SourceType.TOPICAL_INDEX, SourceType.SCRIPTURE),
    QueryIntent.HYBRID_GENERAL: ALL_SOURCES,
}


class QueryPlan(BaseModel):
    """Deterministic retrieval plan derived from a natural-language query."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    intent: QueryIntent = QueryIntent.HYBRID_GENERAL
    secondary_intents: list[QueryIntent] = Field(default_factory=list)
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
    def intents(self) -> list[QueryIntent]:
        """Primary intent followed by the secondary intents."""
        return [self.intent, *self.secondary_intents]

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
        text = _TOPIC_DANGLING_RE.sub("", text.strip())
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


def _strip_reference_spans(text: str) -> str:
    """Remove every valid Bible reference (with its continuations) from the text."""
    kept: list[str] = []
    position = 0
    for match in _REF_RE.finditer(text):
        if match.start() < position:
            continue
        parsed = extract_bible_references(match.group())
        if not parsed:
            continue
        end = match.end()
        while continuation := _CONTINUATION_RE.match(text, end):
            extended = extract_bible_references(text[match.start() : continuation.end()])
            if len(extended) <= len(parsed):
                break
            end, parsed = continuation.end(), extended
        kept.append(text[position : match.start()])
        position = end
    kept.append(text[position:])
    return "".join(kept)


def _strip_book_context(text: str) -> str:
    """Drop 'in <Book>' scoping phrases so they never leak into a topic keyword."""
    spans = [m.group("book") for m in _BOOK_RE.finditer(text) if find_book_mentions(m.group())]
    for book_text in dict.fromkeys(spans):
        text = re.sub(_BOOK_CONTEXT + re.escape(book_text) + r"\b\.?", "", text)
    return text


def _valid_topic(topic: str | None) -> str | None:
    """Reject empty, stopword, or book-name-only topics."""
    if not topic or topic in _STOPWORDS or resolve_book(topic) is not None:
        return None
    return topic


def _implicit_topic(text: str) -> str | None:
    for pattern in _IMPLICIT_TOPIC_PATTERNS:
        match = pattern.search(text)
        if match:
            quoted = _first_quoted(match.group("k"))
            topic = _valid_topic(quoted.lower() if quoted else _clean_topic(match.group("k")))
            if topic:
                return topic
    return None


def route_query(query: str) -> QueryPlan:
    """Classify a query into a multi-intent retrieval plan using deterministic rules (no LLM).

    Every detector runs on every query. The primary intent follows INTENT_PRECEDENCE; the other
    detected intents become `secondary_intents`, whose sources follow the primary's sources.
    """
    text = (query or "").strip()
    verse_ranges = extract_bible_references(text)
    # Topics are extracted with references and "in <Book>" scoping removed.
    topic_text = _strip_book_context(_strip_reference_spans(text))

    detected: set[QueryIntent] = set()
    if verse_ranges:
        detected.add(QueryIntent.EXACT_PASSAGE)
    if _OUTLINE_RE.search(text):
        detected.add(QueryIntent.OUTLINE)

    concordance_word: str | None = None
    if _CONCORDANCE_RE.search(text) or _CONCORDANCE_EXTRA_RE.search(text):
        detected.add(QueryIntent.CONCORDANCE)
        concordance_word = extract_concordance_word(text)

    topic_keyword: str | None = None
    if _TOPICAL_RE.search(text):
        detected.add(QueryIntent.TOPICAL)
        topic_keyword = _valid_topic(extract_topic_keyword(topic_text))
    implicit_topic = _implicit_topic(topic_text)
    if implicit_topic:
        detected.add(QueryIntent.TOPICAL)
        topic_keyword = topic_keyword or implicit_topic

    mentioned: list[str] = []
    for match in _BOOK_RE.finditer(text):
        book_text = match.group("book")
        # "Mark of the Beast" is a topic, not the Gospel of Mark.
        if topic_keyword and re.search(rf"\b{re.escape(book_text.lower())}\b", topic_keyword):
            continue
        mentioned.extend(book.code for book in find_book_mentions(match.group()))

    ordered = [intent for intent in INTENT_PRECEDENCE if intent in detected]
    primary = ordered[0] if ordered else QueryIntent.HYBRID_GENERAL
    target_sources = list(
        dict.fromkeys(src for intent in (ordered or [primary]) for src in INTENT_SOURCES[intent])
    )
    return QueryPlan(
        original_query=query,
        intent=primary,
        secondary_intents=ordered[1:],
        verse_ranges=verse_ranges,
        book_codes=list(dict.fromkeys([*(vr.book_code for vr in verse_ranges), *mentioned])),
        target_sources=target_sources,
        concordance_word=concordance_word,
        topic_keyword=topic_keyword,
    )
