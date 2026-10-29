"""Deterministic query routing, hybrid retrieval, and reranking for the BSB RAG pipeline."""

from beresbirag.retrieval.hybrid_retriever import (
    BOOK_SCOPED_SOURCES,
    BSBHybridRetriever,
    BSBLangChainRetriever,
)
from beresbirag.retrieval.query_router import (
    INTENT_PRECEDENCE,
    INTENT_SOURCES,
    QueryPlan,
    extract_concordance_word,
    extract_topic_keyword,
    route_query,
)
from beresbirag.retrieval.reference_parser import (
    MAX_EXPANDED_REFS,
    chapter_count,
    extract_bible_references,
    find_book_mentions,
    verse_count,
)
from beresbirag.retrieval.reranker import EXACT_MATCH_BONUS, PassageReranker

__all__ = [
    "BOOK_SCOPED_SOURCES",
    "BSBHybridRetriever",
    "BSBLangChainRetriever",
    "EXACT_MATCH_BONUS",
    "INTENT_PRECEDENCE",
    "INTENT_SOURCES",
    "MAX_EXPANDED_REFS",
    "PassageReranker",
    "QueryPlan",
    "chapter_count",
    "extract_bible_references",
    "extract_concordance_word",
    "extract_topic_keyword",
    "find_book_mentions",
    "route_query",
    "verse_count",
]

