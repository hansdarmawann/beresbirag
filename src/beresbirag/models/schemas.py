"""Shared Pydantic v2 schemas and contracts for the beresbirag RAG pipeline."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

BERESBIRAG_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "beresbirag.bsb.bible")


def make_point_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID5 string from a human-readable chunk_id for Qdrant."""
    return str(uuid.uuid5(BERESBIRAG_UUID_NAMESPACE, chunk_id))


class SourceType(StrEnum):
    """Corpus source classification for Qdrant payload filtering."""

    SCRIPTURE = "scripture"
    BOOK_OUTLINE = "book_outline"
    TOPICAL_INDEX = "topical_index"
    CONCORDANCE = "concordance"


class QueryIntent(StrEnum):
    """Deterministic query classification intent."""

    EXACT_PASSAGE = "exact_passage"
    TOPICAL = "topical"
    CONCORDANCE = "concordance"
    OUTLINE = "outline"
    HYBRID_GENERAL = "hybrid_general"


class AnswerMode(StrEnum):
    """How a RAGResponse answer was produced."""

    LLM = "llm"
    FAST_PATH = "fast_path"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class VerseRange(BaseModel):
    """Parsed Bible reference or verse range."""

    model_config = ConfigDict(extra="forbid")

    book_code: str = Field(..., description="3-letter USJ book code, e.g. 'GEN', '1CO'")
    book_name: str = Field(..., description="Canonical English book name, e.g. 'Genesis'")
    start_chapter: int = Field(..., ge=1)
    start_verse: int | None = Field(default=None, ge=1)
    end_chapter: int = Field(..., ge=1)
    end_verse: int | None = Field(default=None, ge=1)
    canonical_ref: str = Field(..., description="Human-readable reference, e.g. 'Genesis 1:1-5'")
    expanded_refs: list[str] = Field(
        default_factory=list,
        description="Normalized USJ verse keys for exact Qdrant matching, e.g. ['GEN 1:1', 'GEN 1:2']",
    )


class ChunkMetadata(BaseModel):
    """Structured metadata stored in Qdrant payload for every chunk."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(..., description="Unique human-readable chunk identifier")
    source_type: SourceType = Field(..., description="Which of the 4 BSB corpora this chunk belongs to")
    book_code: str | None = Field(default=None, description="3-letter USJ code (e.g. 'GEN')")
    book_name: str | None = Field(default=None, description="Full English book name (e.g. 'Genesis')")
    testament: Literal["OT", "NT"] | None = Field(default=None, description="'OT' or 'NT'")
    canonical_order: int | None = Field(default=None, ge=1, le=66)
    chapter: int | None = Field(default=None, ge=1)
    verse_start: int | None = Field(default=None, ge=1)
    verse_end: int | None = Field(default=None, ge=1)
    canonical_ref: str | None = Field(
        default=None,
        description="Formatted citation string, e.g. 'Genesis 1:1-5' or 'Topical Index: Grace'",
    )
    verse_refs: list[str] = Field(
        default_factory=list,
        description="List of normalized verse keys covered by this chunk, e.g. ['GEN 1:1', 'GEN 1:2']",
    )
    section_heading: str | None = Field(
        default=None,
        description="USJ s1/s2 pericope heading, e.g. 'The Creation / The First Day'",
    )
    outline_path: str | None = Field(
        default=None,
        description="Hierarchical breadcrumb from bsb_book_outlines.pdf",
    )
    cross_references: list[str] = Field(
        default_factory=list,
        description="Resolved cross-references attached to this passage",
    )
    footnotes: list[str] = Field(
        default_factory=list,
        description="Resolved BSB footnotes attached to verses in this chunk",
    )
    topic: str | None = Field(
        default=None,
        description="Lowercase normalized topic name for topical_index chunks",
    )
    topic_display: str | None = Field(
        default=None,
        description="Original case topic name for topical_index chunks",
    )
    concordance_word: str | None = Field(
        default=None,
        description="Lowercase normalized headword for concordance chunks",
    )
    concordance_occurrences: int | None = Field(
        default=None,
        ge=0,
        description="Total occurrence count in BSB for concordance headword",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkDocument(BaseModel):
    """Canonical chunk unit produced by ingestion parsers and indexed in Qdrant."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'scripture:GEN:1:1-5:0'")
    text: str = Field(..., min_length=1, description="Self-contained context-enriched text for embedding & LLM")
    metadata: ChunkMetadata

    @property
    def point_id(self) -> str:
        """Deterministic UUID5 string suitable as a Qdrant point ID."""
        return make_point_uuid(self.chunk_id)

    def to_langchain_document(self) -> Document:
        """Convert to a LangChain Document with JSON-serializable metadata."""
        payload = self.metadata.model_dump(mode="json")
        payload["point_id"] = self.point_id
        return Document(page_content=self.text, metadata=payload, id=self.point_id)

    @classmethod
    def from_langchain_document(cls, doc: Document) -> ChunkDocument:
        """Reconstruct a ChunkDocument from a LangChain Document."""
        raw_meta = dict(doc.metadata or {})
        raw_meta.pop("point_id", None)
        raw_meta.pop("_id", None)
        raw_meta.pop("_collection_name", None)
        chunk_id = raw_meta.get("chunk_id") or str(doc.id or uuid.uuid4())
        raw_meta["chunk_id"] = chunk_id
        if "source_type" not in raw_meta:
            raw_meta["source_type"] = SourceType.SCRIPTURE.value
        meta = ChunkMetadata.model_validate(raw_meta)
        return cls(chunk_id=chunk_id, text=doc.page_content, metadata=meta)


class RetrievalResult(BaseModel):
    """A single retrieved and optionally reranked chunk with scores."""

    model_config = ConfigDict(extra="forbid")

    chunk: ChunkDocument
    score: float = Field(default=0.0, description="Vector / RRF similarity score")
    rerank_score: float | None = Field(default=None, description="Cross-encoder reranker score if reranked")
    rank: int = Field(default=1, ge=1, description="1-based final rank position")


class CitationItem(BaseModel):
    """Structured citation returned alongside the RAG answer."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_type: SourceType
    canonical_ref: str
    book_code: str | None = None
    section_heading: str | None = None
    outline_path: str | None = None
    score: float = 0.0
    used: bool = Field(default=False, description="True if the answer cites this chunk")


class CitationReport(BaseModel):
    """Classification of every bracket citation in an answer against the retrieved context."""

    model_config = ConfigDict(extra="forbid")

    valid: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    unparsed: list[str] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    has_citations: bool = False


class RAGResponse(BaseModel):
    """End-to-end response from the BSB RAG pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    intent: QueryIntent = QueryIntent.HYBRID_GENERAL
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    effort: str = "minimal"
    answer_mode: AnswerMode = AnswerMode.LLM
    citations: list[CitationItem] = Field(default_factory=list)
    citation_report: CitationReport | None = None
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
