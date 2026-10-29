"""Local cross-encoder reranking (FlashRank) with deterministic exact-match boosting."""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

from beresbirag.config import Settings, get_settings
from beresbirag.models.schemas import (
    ChunkMetadata,
    QueryIntent,
    RetrievalResult,
    SourceType,
    VerseRange,
)
from beresbirag.retrieval.query_router import QueryPlan

logger = logging.getLogger(__name__)

EXACT_MATCH_BONUS = 0.25

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEXICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be",
        "bible", "book", "by", "can", "did", "do", "does", "for", "from", "give", "how",
        "i", "in", "is", "it", "me", "of", "on", "or", "say", "says", "show", "tell",
        "that", "the", "this", "to", "verse", "verses", "was", "what", "when", "where",
        "which", "who", "why", "with",
    }
)  # fmt: skip


class _RankerLike(Protocol):
    def rerank(self, request: Any) -> list[dict[str, Any]]: ...


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _LEXICAL_STOPWORDS]


def _range_overlaps(meta: ChunkMetadata, vr: VerseRange) -> bool:
    """True if a scripture chunk's chapter/verse span overlaps a requested VerseRange."""
    if meta.book_code != vr.book_code or meta.chapter is None:
        return False
    chapter = meta.chapter
    if not vr.start_chapter <= chapter <= vr.end_chapter:
        return False
    chunk_start = meta.verse_start or 1
    chunk_end = meta.verse_end or chunk_start
    if chapter == vr.start_chapter and vr.start_verse is not None and chunk_end < vr.start_verse:
        return False
    if chapter == vr.end_chapter and vr.end_verse is not None and chunk_start > vr.end_verse:
        return False
    return True


def is_exact_match(result: RetrievalResult, plan: QueryPlan | None) -> bool:
    """Return True if a chunk exactly satisfies a structural target requested by the plan."""
    if plan is None:
        return False
    meta = result.chunk.metadata
    if plan.target_sources and meta.source_type not in plan.target_sources:
        return False

    if plan.verse_ranges:
        requested = set(plan.expanded_refs)
        if requested.intersection(r.upper() for r in meta.verse_refs):
            return True
        if meta.source_type == SourceType.SCRIPTURE and any(
            _range_overlaps(meta, vr) for vr in plan.verse_ranges
        ):
            return True

    if plan.concordance_word and meta.concordance_word:
        if meta.concordance_word.strip().lower() == plan.concordance_word.strip().lower():
            return True

    if plan.topic_keyword and meta.topic:
        if meta.topic.strip().lower() == plan.topic_keyword.strip().lower():
            return True

    if plan.intent == QueryIntent.OUTLINE and plan.book_codes and meta.book_code:
        if meta.source_type == SourceType.BOOK_OUTLINE and meta.book_code in plan.book_codes:
            return True

    return False


def deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Deduplicate by chunk_id, keeping the first occurrence but the best retrieval score."""
    merged: dict[str, RetrievalResult] = {}
    for result in results:
        cid = result.chunk.chunk_id
        existing = merged.get(cid)
        if existing is None:
            merged[cid] = result.model_copy()
        elif result.score > existing.score:
            merged[cid] = existing.model_copy(update={"score": result.score})
    return list(merged.values())


def _default_cache_dir(settings: Settings | None = None) -> str:
    return os.environ.get("FLASHRANK_CACHE_DIR") or (settings or get_settings()).model_cache_dir


class PassageReranker:
    """Cross-encoder reranker with a deterministic lexical+vector fallback and exact-match bonus."""

    def __init__(
        self,
        settings: Settings | None = None,
        ranker: _RankerLike | None = None,
        enabled: bool | None = None,
        exact_match_bonus: float = EXACT_MATCH_BONUS,
    ) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.reranker_enabled if enabled is None else enabled
        self.exact_match_bonus = exact_match_bonus
        self._ranker: _RankerLike | None = ranker
        self._load_failed = False

    @property
    def ranker(self) -> _RankerLike | None:
        """Lazily load the FlashRank cross-encoder; return None when disabled or unavailable."""
        if self._ranker is not None:
            return self._ranker
        if not self.enabled or self._load_failed:
            return None
        try:
            from flashrank import Ranker

            cache_dir = _default_cache_dir(self.settings)
            # FlashRank has no local_files_only flag and downloads missing model
            # directories even when HF_HUB_OFFLINE is set.
            if (
                self.settings.offline_mode
                and not (Path(cache_dir) / self.settings.reranker_model).is_dir()
            ):
                raise FileNotFoundError(
                    "Offline FlashRank cache missing; run scripts/prefetch_models.py online first"
                )
            self._ranker = Ranker(
                model_name=self.settings.reranker_model,
                cache_dir=cache_dir,
                log_level="WARNING",
            )
        except Exception as exc:  # offline, missing model, or runtime error
            logger.warning("FlashRank unavailable (%s); using score-based fallback.", exc)
            self._load_failed = True
            return None
        return self._ranker

    def _cross_encoder_scores(
        self, query: str, results: list[RetrievalResult]
    ) -> list[float] | None:
        ranker = self.ranker
        if ranker is None:
            return None
        try:
            from flashrank import RerankRequest

            passages = [
                {"id": idx, "text": r.chunk.text, "meta": {"chunk_id": r.chunk.chunk_id}}
                for idx, r in enumerate(results)
            ]
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        except Exception as exc:
            logger.warning("FlashRank rerank failed (%s); using score-based fallback.", exc)
            return None
        scores = [0.0] * len(results)
        for item in ranked:
            idx = int(item["id"])
            if 0 <= idx < len(scores):
                score = float(item.get("score", 0.0))
                scores[idx] = score if math.isfinite(score) else 0.0
        return scores

    @staticmethod
    def _fallback_scores(query: str, results: list[RetrievalResult]) -> list[float]:
        """Fuse min-max normalized retrieval scores with lexical query-term coverage."""
        raw = [r.score for r in results]
        lo, hi = min(raw), max(raw)
        span = hi - lo
        query_terms = set(_tokens(query))
        fused: list[float] = []
        for result, score in zip(results, raw, strict=True):
            vector_part = (score - lo) / span if span > 0 else 0.5
            if query_terms:
                chunk_terms = set(_tokens(result.chunk.text))
                lexical_part = len(query_terms & chunk_terms) / len(query_terms)
            else:
                lexical_part = 0.0
            fused.append(0.5 * vector_part + 0.5 * lexical_part)
        return fused

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        plan: QueryPlan | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Deduplicate, score, boost exact structural matches, and return ranked results."""
        candidates = deduplicate_results(results)
        if not candidates:
            return []

        base_scores = self._cross_encoder_scores(query, candidates)
        if base_scores is None:
            base_scores = self._fallback_scores(query, candidates)

        scored: list[tuple[float, int, RetrievalResult]] = []
        for idx, (result, base) in enumerate(zip(candidates, base_scores, strict=True)):
            bonus = self.exact_match_bonus if is_exact_match(result, plan) else 0.0
            scored.append((base + bonus, idx, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = top_k if top_k is not None else len(scored)
        return [
            result.model_copy(update={"rerank_score": round(final, 6), "rank": rank})
            for rank, (final, _, result) in enumerate(scored[:limit], start=1)
        ]
