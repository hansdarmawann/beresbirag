"""Hybrid retrieval orchestrator: query routing, exact payload lookup, RRF search, and reranking."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import models

from beresbirag.config import Settings, get_settings
from beresbirag.models.schemas import QueryIntent, RetrievalResult, SourceType, VerseRange
from beresbirag.retrieval.query_router import ALL_SOURCES, QueryPlan, route_query
from beresbirag.retrieval.reference_parser import MAX_EXPANDED_REFS
from beresbirag.retrieval.reranker import PassageReranker, deduplicate_results
from beresbirag.vectorstore.qdrant_manager import QdrantManager

MAX_CHAPTER_FILTERS = 10


def _topic_variants(topic: str) -> list[str]:
    base = topic.strip().lower()
    variants = [base]
    if base.endswith("s") and len(base) > 3:
        variants.append(base[:-1])
    else:
        variants.append(f"{base}s")
    return list(dict.fromkeys(v for v in variants if v))


class BSBHybridRetriever:
    """High-level BSB retriever combining exact payload matches, hybrid RRF search, and reranking."""

    def __init__(
        self,
        qdrant_manager: QdrantManager | None = None,
        reranker: PassageReranker | None = None,
        settings: Settings | None = None,
        exact_match_limit: int = 20,
    ) -> None:
        self.settings = settings or (qdrant_manager.settings if qdrant_manager else get_settings())
        self.qdrant_manager = qdrant_manager or QdrantManager(settings=self.settings)
        self.reranker = reranker or PassageReranker(settings=self.settings)
        self.exact_match_limit = exact_match_limit

    # ------------------------------------------------------------------ filters
    def _verse_range_filters(self, vr: VerseRange) -> list[models.Filter]:
        build = self.qdrant_manager.build_filter
        is_complete = vr.start_verse is not None and len(vr.expanded_refs) < MAX_EXPANDED_REFS
        if is_complete and vr.expanded_refs:
            flt = build(source_types=[SourceType.SCRIPTURE], verse_refs=vr.expanded_refs)
            return [flt] if flt is not None else []
        filters: list[models.Filter] = []
        last_chapter = min(vr.end_chapter, vr.start_chapter + MAX_CHAPTER_FILTERS - 1)
        for chapter in range(vr.start_chapter, last_chapter + 1):
            flt = build(
                source_types=[SourceType.SCRIPTURE],
                book_codes=[vr.book_code],
                chapter=chapter,
            )
            if flt is not None:
                filters.append(flt)
        return filters

    def _exact_filters(self, plan: QueryPlan) -> list[models.Filter]:
        """Build deterministic payload filters for every exact target in the plan."""
        build = self.qdrant_manager.build_filter
        filters: list[models.Filter] = []
        for vr in plan.verse_ranges:
            filters.extend(self._verse_range_filters(vr))
        if plan.concordance_word:
            flt = build(
                source_types=[SourceType.CONCORDANCE],
                concordance_word=plan.concordance_word,
            )
            if flt is not None:
                filters.append(flt)
        if plan.topic_keyword:
            for topic in _topic_variants(plan.topic_keyword):
                flt = build(source_types=[SourceType.TOPICAL_INDEX], topic=topic)
                if flt is not None:
                    filters.append(flt)
        if plan.intent == QueryIntent.OUTLINE and plan.book_codes:
            flt = build(source_types=[SourceType.BOOK_OUTLINE], book_codes=plan.book_codes)
            if flt is not None:
                filters.append(flt)
        return filters

    def _hybrid_filter(self, plan: QueryPlan) -> models.Filter | None:
        """Restrict hybrid search to the plan's target sources (and books for exact passages)."""
        sources = None if set(plan.target_sources) >= set(ALL_SOURCES) else plan.target_sources
        book_codes = plan.book_codes if plan.intent == QueryIntent.EXACT_PASSAGE else None
        return self.qdrant_manager.build_filter(source_types=sources, book_codes=book_codes)

    # ---------------------------------------------------------------- retrieval
    def fetch_exact_matches(self, plan: QueryPlan) -> list[RetrievalResult]:
        """Deterministically scroll Qdrant for chunks whose payload matches the plan exactly."""
        if not (plan.has_exact_targets or (plan.intent == QueryIntent.OUTLINE and plan.book_codes)):
            return []
        exact: list[RetrievalResult] = []
        for flt in self._exact_filters(plan):
            exact.extend(self.qdrant_manager.scroll_by_filter(flt, limit=self.exact_match_limit))
        return deduplicate_results(exact)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        initial_k: int | None = None,
    ) -> tuple[QueryPlan, list[RetrievalResult]]:
        """Route, fetch exact + hybrid candidates, rerank, and return the top-k results."""
        final_k = top_k or self.settings.retrieval_top_k
        candidate_k = max(initial_k or self.settings.retrieval_initial_k, final_k)

        plan = route_query(query)
        exact = self.fetch_exact_matches(plan)
        hybrid = self.qdrant_manager.hybrid_search(
            query=query,
            k=candidate_k,
            qdrant_filter=self._hybrid_filter(plan),
        )
        candidates = deduplicate_results([*exact, *hybrid])
        ranked = self.reranker.rerank(query, candidates, plan=plan, top_k=final_k)
        return plan, ranked

    # --------------------------------------------------------------- formatting
    @staticmethod
    def format_context_for_llm(results: list[RetrievalResult]) -> str:
        """Format results as numbered, citation-anchored context blocks for small LLMs."""
        blocks: list[str] = []
        for idx, result in enumerate(results, start=1):
            meta = result.chunk.metadata
            reference = meta.canonical_ref or result.chunk.chunk_id
            header = f"=== SOURCE [{idx}] | Reference: {reference} | Type: {meta.source_type.value} ==="
            blocks.append(f"{header}\n{result.chunk.text.strip()}")
        return "\n\n".join(blocks)

    def as_langchain_retriever(
        self,
        top_k: int | None = None,
        initial_k: int | None = None,
    ) -> BaseRetriever:
        """Return a LangChain BaseRetriever adapter for LCEL chains."""
        return BSBLangChainRetriever(hybrid_retriever=self, top_k=top_k, initial_k=initial_k)


class BSBLangChainRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter around BSBHybridRetriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: Any
    top_k: int | None = None
    initial_k: int | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        plan, results = self.hybrid_retriever.retrieve(
            query, top_k=self.top_k, initial_k=self.initial_k
        )
        documents: list[Document] = []
        for result in results:
            doc = result.chunk.to_langchain_document()
            doc.metadata.update(
                {
                    "rank": result.rank,
                    "score": result.score,
                    "rerank_score": result.rerank_score,
                    "intent": plan.intent.value,
                }
            )
            documents.append(doc)
        return documents

