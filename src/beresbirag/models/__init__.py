"""Shared models, schemas, and canonical BSB book registry."""

from beresbirag.models.books import (
    BOOKS_BY_CODE,
    BSB_BOOKS,
    BookInfo,
    TestamentType,
    format_canonical_ref,
    get_book_by_code,
    resolve_book,
)
from beresbirag.models.schemas import (
    BERESBIRAG_UUID_NAMESPACE,
    ChunkDocument,
    ChunkMetadata,
    CitationItem,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
    SourceType,
    VerseRange,
    make_point_uuid,
)

__all__ = [
    "BERESBIRAG_UUID_NAMESPACE",
    "BOOKS_BY_CODE",
    "BSB_BOOKS",
    "BookInfo",
    "ChunkDocument",
    "ChunkMetadata",
    "CitationItem",
    "QueryIntent",
    "RAGResponse",
    "RetrievalResult",
    "SourceType",
    "TestamentType",
    "VerseRange",
    "format_canonical_ref",
    "get_book_by_code",
    "make_point_uuid",
    "resolve_book",
]
