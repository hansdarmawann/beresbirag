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
from beresbirag.retrieval.formatting import format_context_blocks
from beresbirag.retrieval.query_router import ALL_SOURCES, INTENT_SOURCES, QueryPlan, route_query
from beresbirag.retrieval.reference_parser import MAX_EXPANDED_REFS
from beresbirag.retrieval.reranker import PassageReranker, deduplicate_results
from beresbirag.vectorstore.qdrant_manager import QdrantManager

MAX_CHAPTER_FILTERS = 10

# Sources whose chunks carry a book_code and can therefore be scoped to mentioned books.
BOOK_SCOPED_SOURCES: frozenset[SourceType] = frozenset(
    {SourceType.SCRIPTURE, SourceType.BOOK_OUTLINE}
)


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
        if QueryIntent.OUTLINE in plan.intents and plan.book_codes:
            flt = build(source_types=[SourceType.BOOK_OUTLINE], book_codes=plan.book_codes)
            if flt is not None:
                filters.append(flt)
        return filters

    def _hybrid_filter(self, plan: QueryPlan) -> models.Filter | None:
        """Restrict hybrid search to the plan's target sources, book-scoping where possible.

        With book codes, scripture/outline chunks must belong to those books while the other
        target sources (topical index, concordance) stay searchable in a second `should` branch.
        """
        build = self.qdrant_manager.build_filter
        targets = list(plan.target_sources)
        if not plan.book_codes:
            sources = None if set(targets) >= set(ALL_SOURCES) else targets
            return build(source_types=sources)

        scoped = [src for src in targets if src in BOOK_SCOPED_SOURCES]
        others = [src for src in targets if src not in BOOK_SCOPED_SOURCES]
        branches = [
            flt
            for flt in (
                build(source_types=scoped, book_codes=plan.book_codes) if scoped else None,
                build(source_types=others) if others else None,
            )
            if flt is not None
        ]
        if len(branches) == 1:
            return branches[0]
        return models.Filter(should=branches) if branches else None

    # ---------------------------------------------------------------- retrieval
    def fetch_exact_matches(self, plan: QueryPlan) -> list[RetrievalResult]:
        """Deterministically scroll Qdrant for chunks whose payload matches the plan exactly."""
        if not (plan.has_exact_targets or (QueryIntent.OUTLINE in plan.intents and plan.book_codes)):
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
        if not plan.secondary_intents:
            return plan, self.reranker.rerank(query, candidates, plan=plan, top_k=final_k)
        ranked = self.reranker.rerank(query, candidates, plan=plan)
        return plan, self._cover_intents(plan, ranked, exact, final_k)

    @staticmethod
    def _cover_intents(
        plan: QueryPlan,
        ranked: list[RetrievalResult],
        exact: list[RetrievalResult],
        k: int,
    ) -> list[RetrievalResult]:
        """Keep one exact match per detected intent in the top-k of a multi-intent query.

        Without this, several overlapping primary-intent chunks (e.g. scripture windows around
        Revelation 13:18) can crowd the secondary intent's exact match out of the context.
        Displaced results are taken from the lowest-ranked unprotected positions.
        """
        exact_ids = {r.chunk.chunk_id for r in exact}
        final = ranked[:k]
        protected: set[str] = set()
        for intent in plan.intents:
            source = INTENT_SOURCES[intent][0]

            def _is_target(result: RetrievalResult, source: SourceType = source) -> bool:
                return (
                    result.chunk.metadata.source_type == source
                    and result.chunk.chunk_id in exact_ids
                )

            present = next((r for r in final if _is_target(r)), None)
            if present is not None:
                protected.add(present.chunk.chunk_id)
                continue
            candidate = next((r for r in ranked[k:] if _is_target(r)), None)
            victims = [i for i, r in enumerate(final) if r.chunk.chunk_id not in protected]
            if candidate is None or not victims:
                continue
            final[victims[-1]] = candidate
            protected.add(candidate.chunk.chunk_id)
        order = {r.chunk.chunk_id: i for i, r in enumerate(ranked)}
        final.sort(key=lambda r: order[r.chunk.chunk_id])
        return [r.model_copy(update={"rank": i}) for i, r in enumerate(final, start=1)]

    # --------------------------------------------------------------- formatting
    @staticmethod
    def format_context_for_llm(results: list[RetrievalResult]) -> str:
        """Format results as numbered, citation-anchored context blocks for small LLMs."""
        return format_context_blocks(results, empty_message=None)

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

