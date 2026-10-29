"""Retrieval and answer quality metrics for the BSB evaluation suites."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from beresbirag.evaluation.cases import AnswerCase, RetrievalCase
from beresbirag.llm.citations import parse_label
from beresbirag.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from beresbirag.models.schemas import (
    AnswerMode,
    CitationReport,
    RAGResponse,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import QueryPlan
from beresbirag.retrieval.reference_parser import extract_bible_references
from beresbirag.retrieval.reranker import is_exact_match

_VERSE_KEY_RE = re.compile(r"^[1-3]?[A-Z]{2,3} \d{1,3}:\d{1,3}$")


def _covered_refs(case: RetrievalCase, results: Iterable[RetrievalResult]) -> set[str]:
    source = case.expected.source_type
    return {
        ref.upper()
        for r in results
        if source is None or r.chunk.metadata.source_type == source
        for ref in r.chunk.metadata.verse_refs
    }


def hit_rank(case: RetrievalCase, results: Sequence[RetrievalResult]) -> int | None:
    """1-based rank at which the case is first fully satisfied, or None.

    That is the first chunk matching every per-chunk criterion, and, when `verse_refs_all` is
    set, the rank by which the results so far cover every required verse.
    """
    expected = case.expected
    first = next((i for i, r in enumerate(results, start=1) if expected.matches(r)), None)
    if first is None or not expected.verse_refs_all:
        return first
    required = {ref.upper() for ref in expected.verse_refs_all}
    for i in range(1, len(results) + 1):
        if required <= _covered_refs(case, results[:i]):
            return max(first, i)
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Fraction of cases whose hit rank is <= k."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank (misses contribute 0)."""
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank) / len(ranks)


def context_coverage(case: RetrievalCase, results: Sequence[RetrievalResult]) -> float | None:
    """Fraction of `verse_refs_all` present in the union of the results' verse_refs.

    Only chunks of the expected source type count. None when the case has no `verse_refs_all`.
    """
    required = {ref.upper() for ref in case.expected.verse_refs_all}
    if not required:
        return None
    return len(required & _covered_refs(case, results)) / len(required)


def has_false_exact_match(plan: QueryPlan, results: Sequence[RetrievalResult]) -> bool:
    """True if any result is presented as an exact structural match for an unanswerable plan."""
    return any(is_exact_match(result, plan) for result in results)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_refusal(response: RAGResponse) -> bool:
    """True if the pipeline declined to answer from the retrieved context."""
    if response.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT:
        return True
    return _normalize(INSUFFICIENT_CONTEXT_ANSWER) in _normalize(response.answer)


def citation_precision(*reports: CitationReport | None) -> float | None:
    """valid / (valid + partial + unsupported), micro-averaged over the given reports."""
    valid = total = 0
    for report in reports:
        if report is None:
            continue
        valid += len(report.valid)
        total += len(report.valid) + len(report.partial) + len(report.unsupported)
    return valid / total if total else None


def _cited_targets(response: RAGResponse) -> tuple[set[str], set[tuple[str, str]]]:
    """Verse keys and (kind, value) labels covered by the answer's valid/partial citations."""
    report = response.citation_report
    verses: set[str] = set()
    labels: set[tuple[str, str]] = set()
    if report is None:
        return verses, labels
    by_rank = {i: r for i, r in enumerate(response.retrieved_chunks, start=1)}
    for content in [*report.valid, *report.partial]:
        if content.isdigit() and int(content) in by_rank:
            meta = by_rank[int(content)].chunk.metadata
            verses.update(ref.upper() for ref in meta.verse_refs)
            if meta.source_type == SourceType.TOPICAL_INDEX and meta.topic:
                labels.add(("topic", meta.topic.lower()))
            elif meta.source_type == SourceType.CONCORDANCE and meta.concordance_word:
                labels.add(("concordance", meta.concordance_word.lower()))
            elif meta.source_type == SourceType.BOOK_OUTLINE and meta.book_code:
                labels.add(("outline", meta.book_code))
            continue
        label = parse_label(content)
        if label is not None:
            labels.add(label)
            continue
        for vr in extract_bible_references(content):
            verses.update(key.upper() for key in vr.expanded_refs)
    return verses, labels


def must_cite_recall(case: AnswerCase, response: RAGResponse) -> float | None:
    """Fraction of the case's must-cite verse keys / labels covered by supported citations."""
    if not case.must_cite:
        return None
    verses, labels = _cited_targets(response)
    hits = 0
    for item in case.must_cite:
        if _VERSE_KEY_RE.match(item.strip().upper()):
            hits += item.strip().upper() in verses
        else:
            label = parse_label(item)
            hits += label is not None and label in labels
    return hits / len(case.must_cite)


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); None for an empty sequence."""
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
