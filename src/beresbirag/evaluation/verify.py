"""Ground-truth verification: prove every evaluation case is satisfiable by the real BSB corpus."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from beresbirag.config import Settings, get_settings
from beresbirag.evaluation.cases import AnswerCase, RetrievalCase
from beresbirag.ingestion.concordance_parser import iter_concordance_chunks
from beresbirag.ingestion.pdf_outline_parser import BookOutlineIndex
from beresbirag.ingestion.topical_parser import iter_topical_chunks
from beresbirag.ingestion.usj_parser import iter_usj_scripture_chunks
from beresbirag.llm.citations import parse_label
from beresbirag.models.schemas import SourceType
from beresbirag.retrieval.hybrid_retriever import _topic_variants
from beresbirag.retrieval.query_router import route_query
from beresbirag.retrieval.reference_parser import BSB_VERSE_COUNTS

_VERSE_KEY_RE = re.compile(r"^(?P<book>[1-3]?[A-Z]{2,3}) (?P<chapter>\d{1,3}):(?P<verse>\d{1,3})$")


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    case_id: str
    message: str


@dataclass
class CorpusIndex:
    """The slice of the real corpus needed to check case expectations."""

    topics: set[str] = field(default_factory=set)
    concordance_words: set[str] = field(default_factory=set)
    scripture_by_verse: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    scripture_by_book: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    topic_texts: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    concordance_texts: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    outline_texts: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def verse_key_exists(key: str) -> bool:
    """True if a normalized verse key ('JHN 3:16') exists in BSB versification."""
    match = _VERSE_KEY_RE.match(key.strip().upper())
    if match is None:
        return False
    counts = BSB_VERSE_COUNTS.get(match["book"], ())
    chapter, verse = int(match["chapter"]), int(match["verse"])
    return 1 <= chapter <= len(counts) and 1 <= verse <= counts[chapter - 1]


def _case_books(cases: Iterable[RetrievalCase]) -> set[str]:
    books: set[str] = set()
    for case in cases:
        exp = case.expected
        books.update(ref.split(" ", 1)[0].upper() for ref in [*exp.verse_refs_any, *exp.verse_refs_all])
        if exp.book_code:
            books.add(exp.book_code.upper())
    return books & set(BSB_VERSE_COUNTS)


def build_corpus_index(
    retrieval_cases: Sequence[RetrievalCase],
    answer_cases: Sequence[AnswerCase] = (),
    settings: Settings | None = None,
) -> CorpusIndex:
    """Scan the four raw BSB sources once, keeping only what the cases reference."""
    settings = settings or get_settings()
    for path in (
        settings.usj_zip_file,
        settings.outlines_pdf_file,
        settings.topical_xlsx_file,
        settings.concordance_xlsx_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Raw BSB file required for case verification is missing: {path}")

    books = _case_books(retrieval_cases)
    topics_needed = {c.expected.topic.lower() for c in retrieval_cases if c.expected.topic}
    words_needed = {
        c.expected.concordance_word.lower() for c in retrieval_cases if c.expected.concordance_word
    }

    index = CorpusIndex()
    outline_index = BookOutlineIndex.from_pdf(settings.outlines_pdf_file)
    for chunk in outline_index.to_chunk_documents():
        if chunk.metadata.book_code:
            index.outline_texts[chunk.metadata.book_code].append(chunk.text)
    for chunk in iter_usj_scripture_chunks(
        settings.usj_zip_file, outline_index=outline_index, book_filter=books or {"GEN"}
    ):
        index.scripture_by_book[chunk.metadata.book_code or ""].append(chunk.text)
        for ref in chunk.metadata.verse_refs:
            index.scripture_by_verse[ref.upper()].append(chunk.text)
    for chunk in iter_topical_chunks(settings.topical_xlsx_file):
        topic = (chunk.metadata.topic or "").lower()
        index.topics.add(topic)
        if topic in topics_needed:
            index.topic_texts[topic].append(chunk.text)
    for chunk in iter_concordance_chunks(settings.concordance_xlsx_file):
        word = (chunk.metadata.concordance_word or "").lower()
        index.concordance_words.add(word)
        if word in words_needed:
            index.concordance_texts[word].append(chunk.text)
    return index


def _contains(texts: Iterable[str], needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in text.lower() for text in texts)


def _verify_retrieval_case(case: RetrievalCase, corpus: CorpusIndex) -> list[str]:
    exp = case.expected
    problems: list[str] = []
    refs = [*exp.verse_refs_any, *exp.verse_refs_all]
    problems.extend(f"verse key {ref!r} does not exist" for ref in refs if not verse_key_exists(ref))
    if exp.book_code and exp.book_code.upper() not in BSB_VERSE_COUNTS:
        problems.append(f"book code {exp.book_code!r} does not exist")
    if exp.topic and exp.topic.lower() not in corpus.topics:
        problems.append(f"topic {exp.topic!r} is absent from the topical index")
    if exp.concordance_word and exp.concordance_word.lower() not in corpus.concordance_words:
        problems.append(f"concordance word {exp.concordance_word!r} is absent from the concordance")

    if exp.text_contains:
        if exp.source_type == SourceType.TOPICAL_INDEX and exp.topic:
            texts = corpus.topic_texts.get(exp.topic.lower(), [])
        elif exp.source_type == SourceType.CONCORDANCE and exp.concordance_word:
            texts = corpus.concordance_texts.get(exp.concordance_word.lower(), [])
        elif exp.source_type == SourceType.BOOK_OUTLINE and exp.book_code:
            texts = corpus.outline_texts.get(exp.book_code.upper(), [])
        elif refs:
            texts = [t for ref in refs for t in corpus.scripture_by_verse.get(ref.upper(), [])]
        elif exp.book_code:
            texts = corpus.scripture_by_book.get(exp.book_code.upper(), [])
        else:
            texts = []
            problems.append("text_contains needs verse refs, a topic, a word, or a book to check")
        if texts and not _contains(texts, exp.text_contains):
            problems.append(f"text {exp.text_contains!r} not found in the referenced BSB text")
        elif not texts and refs and all(verse_key_exists(ref) for ref in refs):
            problems.append(f"no indexed chunk covers {refs} to check {exp.text_contains!r}")

    if case.unanswerable:
        plan = route_query(case.query)
        if plan.verse_ranges:
            problems.append(f"unanswerable query parses to real verses {plan.expanded_refs[:3]}")
        if plan.topic_keyword and set(_topic_variants(plan.topic_keyword)) & corpus.topics:
            problems.append(f"unanswerable query topic {plan.topic_keyword!r} exists in the index")
        if plan.concordance_word and plan.concordance_word in corpus.concordance_words:
            problems.append(
                f"unanswerable query word {plan.concordance_word!r} exists in the concordance"
            )
    return problems


def _verify_answer_case(case: AnswerCase, corpus: CorpusIndex) -> list[str]:
    problems: list[str] = []
    for item in case.must_cite:
        if _VERSE_KEY_RE.match(item.strip().upper()):
            if not verse_key_exists(item):
                problems.append(f"must_cite verse key {item!r} does not exist")
            continue
        label = parse_label(item)
        if label is None:
            problems.append(f"must_cite item {item!r} is neither a verse key nor a BSB label")
        elif label[0] == "topic" and not set(_topic_variants(label[1])) & corpus.topics:
            problems.append(f"must_cite topic {label[1]!r} is absent from the topical index")
        elif label[0] == "concordance" and label[1] not in corpus.concordance_words:
            problems.append(f"must_cite word {label[1]!r} is absent from the concordance")
    return problems


def verify_cases(
    retrieval_cases: Sequence[RetrievalCase],
    answer_cases: Sequence[AnswerCase],
    corpus: CorpusIndex,
) -> list[VerificationIssue]:
    """Return one issue per unsatisfiable expectation (empty list = every case is grounded)."""
    issues: list[VerificationIssue] = []
    for case in retrieval_cases:
        issues.extend(VerificationIssue(case.id, msg) for msg in _verify_retrieval_case(case, corpus))
    for case in answer_cases:
        issues.extend(VerificationIssue(case.id, msg) for msg in _verify_answer_case(case, corpus))
    return issues
