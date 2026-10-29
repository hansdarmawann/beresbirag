"""End-to-end Berean Standard Bible (BSB) RAG pipeline with LangChain LCEL."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate

from beresbirag.config import Settings, get_settings
from beresbirag.llm.factory import create_chat_model, resolve_llm_target
from beresbirag.llm.prompts import BSB_RAG_PROMPT
from beresbirag.models.schemas import (
    AnswerMode,
    CitationItem,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
)

# Mirrors rule 4 of SYSTEM_PROMPT; returned without calling the LLM when retrieval is empty.
INSUFFICIENT_CONTEXT_ANSWER = (
    "The retrieved Berean Standard Bible context does not contain sufficient "
    "information to answer this question."
)


def format_context_blocks(results: list[RetrievalResult]) -> str:
    """Format retrieved results into numbered, citation-anchored context blocks."""
    if not results:
        return "No relevant context found in the Berean Standard Bible corpus."

    blocks: list[str] = []
    for idx, res in enumerate(results, start=1):
        meta = res.chunk.metadata
        ref = meta.canonical_ref or res.chunk.chunk_id
        source_type_val = (
            meta.source_type.value
            if hasattr(meta.source_type, "value")
            else str(meta.source_type)
        )
        header = f"=== SOURCE [{idx}] | Reference: {ref} | Type: {source_type_val} ==="
        blocks.append(f"{header}\n{res.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_citations(results: list[RetrievalResult]) -> list[CitationItem]:
    """Construct structured CitationItem objects from retrieved chunks."""
    citations: list[CitationItem] = []
    for res in results:
        meta = res.chunk.metadata
        canonical_ref = meta.canonical_ref or res.chunk.chunk_id
        score_val = float(
            res.rerank_score if res.rerank_score is not None else res.score
        )
        citations.append(
            CitationItem(
                chunk_id=res.chunk.chunk_id,
                source_type=meta.source_type,
                canonical_ref=canonical_ref,
                book_code=meta.book_code,
                section_heading=meta.section_heading,
                outline_path=meta.outline_path,
                score=score_val,
            )
        )
    return citations


class BSBRAGPipeline:
    """End-to-end Berean Standard Bible RAG pipeline powered by LangChain LCEL."""

    def __init__(
        self,
        retriever: Any,
        llm: BaseChatModel | None = None,
        prompt: BasePromptTemplate | None = None,
        settings: Settings | None = None,
        provider: Literal["openai", "anthropic"] | None = None,
        model_name: str | None = None,
        effort: Literal["minimal", "low", "medium", "high"] | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.provider, self.model_name = resolve_llm_target(
            provider, model_name, self.settings
        )
        self.effort = effort or self.settings.llm_effort

        self.llm = llm or create_chat_model(
            provider=self.provider,
            model_name=self.model_name,
            effort=self.effort,
            settings=self.settings,
        )
        self.prompt = prompt or BSB_RAG_PROMPT
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _extract_intent(self, plan: Any) -> QueryIntent:
        if plan is not None:
            plan_intent = getattr(plan, "intent", None)
            if plan_intent is not None:
                if isinstance(plan_intent, QueryIntent):
                    return plan_intent
                try:
                    return QueryIntent(str(plan_intent))
                except ValueError:
                    return QueryIntent.HYBRID_GENERAL
        return QueryIntent.HYBRID_GENERAL

    def _resolve_model_metadata(self) -> tuple[str, str]:
        model_label = (
            getattr(self.llm, "model_name", None)
            or getattr(self.llm, "model", None)
            or str(self.model_name)
        )
        provider_label = str(self.provider)
        class_name = self.llm.__class__.__name__.lower()
        if "anthropic" in class_name:
            provider_label = "anthropic"
        elif "openai" in class_name:
            provider_label = "openai"
        return provider_label, model_label

    def _insufficient_context_response(
        self, query: str, intent: QueryIntent
    ) -> RAGResponse:
        provider_label, model_label = self._resolve_model_metadata()
        return RAGResponse(
            query=query,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            intent=intent,
            provider=provider_label,
            model=model_label,
            effort=self.effort,
            answer_mode=AnswerMode.INSUFFICIENT_CONTEXT,
            citations=[],
            retrieved_chunks=[],
        )

    def ask(self, query: str, top_k: int | None = None) -> RAGResponse:
        """Synchronously query the BSB RAG pipeline."""
        effective_top_k = (
            top_k if top_k is not None else self.settings.retrieval_top_k
        )
        plan, results = self.retriever.retrieve(query, top_k=effective_top_k)

        intent = self._extract_intent(plan)
        if not results:
            return self._insufficient_context_response(query, intent)
        context_str = format_context_blocks(results)
        answer = self.chain.invoke({"context": context_str, "question": query})

        citations = build_citations(results)
        provider_label, model_label = self._resolve_model_metadata()

        return RAGResponse(
            query=query,
            answer=str(answer).strip(),
            intent=intent,
            provider=provider_label,
            model=model_label,
            effort=self.effort,
            citations=citations,
            retrieved_chunks=results,
        )

    async def aask(self, query: str, top_k: int | None = None) -> RAGResponse:
        """Asynchronously query the BSB RAG pipeline."""
        effective_top_k = (
            top_k if top_k is not None else self.settings.retrieval_top_k
        )
        plan, results = await asyncio.to_thread(
            self.retriever.retrieve, query, top_k=effective_top_k
        )

        intent = self._extract_intent(plan)
        if not results:
            return self._insufficient_context_response(query, intent)
        context_str = format_context_blocks(results)
        answer = await self.chain.ainvoke({"context": context_str, "question": query})

        citations = build_citations(results)
        provider_label, model_label = self._resolve_model_metadata()

        return RAGResponse(
            query=query,
            answer=str(answer).strip(),
            intent=intent,
            provider=provider_label,
            model=model_label,
            effort=self.effort,
            citations=citations,
            retrieved_chunks=results,
        )
