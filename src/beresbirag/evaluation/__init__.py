"""Retrieval and answer evaluation: typed cases, metrics, and ground-truth verification."""

from beresbirag.evaluation.cases import (
    KNOWN_GAP_TAG,
    AnswerCase,
    CaseFileError,
    RetrievalCase,
    RetrievalExpectation,
    load_answer_cases,
    load_cases,
    load_retrieval_cases,
)
from beresbirag.evaluation.metrics import (
    citation_precision,
    context_coverage,
    has_false_exact_match,
    hit_rank,
    is_refusal,
    mrr,
    must_cite_recall,
    percentile,
    recall_at_k,
)
from beresbirag.evaluation.verify import (
    CorpusIndex,
    VerificationIssue,
    build_corpus_index,
    verify_cases,
    verse_key_exists,
)

__all__ = [
    "AnswerCase",
    "CaseFileError",
    "CorpusIndex",
    "KNOWN_GAP_TAG",
    "RetrievalCase",
    "RetrievalExpectation",
    "VerificationIssue",
    "build_corpus_index",
    "citation_precision",
    "context_coverage",
    "has_false_exact_match",
    "hit_rank",
    "is_refusal",
    "load_answer_cases",
    "load_cases",
    "load_retrieval_cases",
    "mrr",
    "must_cite_recall",
    "percentile",
    "recall_at_k",
    "verify_cases",
    "verse_key_exists",
]
