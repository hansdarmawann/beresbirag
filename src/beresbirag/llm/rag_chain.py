"""End-to-end Berean Standard Bible (BSB) RAG pipeline with LangChain LCEL."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate

from beresbirag.config import Settings, get_settings
from beresbirag.llm.citations import strip_unsupported_citations, validate_citations
from beresbirag.llm.context_budget import estimate_tokens, fit_context
from beresbirag.llm.factory import ProviderUnavailableError, create_chat_model, resolve_llm_target
from beresbirag.llm.fast_path import try_fast_answer
from beresbirag.llm.prompts import (
    BSB_RAG_PROMPT,
    BSB_RAG_PROMPT_COMPACT,
    INSUFFICIENT_CONTEXT_ANSWER,
)
from beresbirag.models.schemas import (
    AnswerMode,
    CitationItem,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
)
from beresbirag.retrieval.formatting import format_context_blocks as _retrieval_format_context_blocks
from beresbirag.retrieval.query_router import QueryPlan


@dataclass
class PreparedQuery:
    query: str
    plan: QueryPlan | None
    results: list[RetrievalResult]
    context: str
    fast_answer: str | None
    answer_mode: AnswerMode
    retrieval_ms: float = 0.0
    prepared_at: float = 0.0


DEFAULT_EMPTY_CONTEXT = "No relevant context found in the Berean Standard Bible corpus."


def format_context_blocks(
    results: list[RetrievalResult],
    empty_message: str | None = DEFAULT_EMPTY_CONTEXT,
) -> str:
    """Format retrieved results into numbered, citation-anchored context blocks."""
    msg = empty_message if empty_message is not None else DEFAULT_EMPTY_CONTEXT
    return _retrieval_format_context_blocks(results, empty_message=msg)


def build_citations(results: list[RetrievalResult]) -> list[CitationItem]:
    """Construct structured CitationItem objects from retrieved chunks."""
    citations: list[CitationItem] = []
    for res in results:
        meta = res.chunk.metadata
        canonical_ref = meta.canonical_ref or res.chunk.chunk_id
        score_val = float(res.rerank_score if res.rerank_score is not None else res.score)
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
        provider: Literal["openai", "anthropic", "ollama"] | None = None,
        model_name: str | None = None,
        effort: Literal["minimal", "low", "medium", "high"] | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.provider, self.model_name = resolve_llm_target(provider, model_name, self.settings)
        self.effort = effort or self.settings.llm_effort

        self._llm = llm
        self.prompt = (
            prompt
            if prompt is not None
            else (
                BSB_RAG_PROMPT_COMPACT
                if self.provider == "ollama" or self.settings.resource_profile == "low"
                else BSB_RAG_PROMPT
            )
        )

    @property
    def llm(self) -> BaseChatModel:
        """Construct only when generation or explicit model access is requested."""
        if self._llm is None:
            self._llm = create_chat_model(
                provider=self.provider,
                model_name=self.model_name,
                effort=self.effort,
                settings=self.settings,
            )
        return self._llm

    @property
    def chain(self):
        if self.settings.offline_mode and self.provider in {"openai", "anthropic"}:
            raise ProviderUnavailableError("offline mode: use --provider ollama")
        return self.prompt | self.llm | StrOutputParser()

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
            getattr(self._llm, "model_name", None)
            or getattr(self._llm, "model", None)
            or str(self.model_name)
        )
        provider_label = str(self.provider)
        class_name = self._llm.__class__.__name__.lower()
        if "anthropic" in class_name:
            provider_label = "anthropic"
        elif "openai" in class_name:
            provider_label = "openai"
        elif "ollama" in class_name:
            provider_label = "ollama"
        return provider_label, model_label

    def prepare(self, query: str, top_k: int | None = None) -> PreparedQuery:
        """Retrieve, select deterministic answers, and budget copies of source text."""
        start_t = perf_counter()
        effective_top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        plan, results = self.retriever.retrieve(query, top_k=effective_top_k)

        fast_answer = (
            try_fast_answer(plan, results)
            if self.settings.fast_path_enabled and plan is not None
            else None
        )
        mode = (
            AnswerMode.FAST_PATH
            if fast_answer is not None
            else AnswerMode.LLM
            if results
            else AnswerMode.INSUFFICIENT_CONTEXT
        )
        results = fit_context(
            results,
            query,
            self.settings.context_token_budget,
            self.settings.context_chunk_token_cap,
        )
        if mode == AnswerMode.LLM and self.provider == "ollama":
            # Include source headers, question, prompt and output reserve. Refuse
            # an indivisible oversized line instead of letting Ollama truncate it.
            while (
                self._prompt_tokens(query, results) + self.settings.llm_num_predict
                > self.settings.llm_num_ctx
            ):
                if len(results) > 1:
                    results = results[:-1]
                else:
                    lines = results[0].chunk.text.splitlines()
                    if len(lines) <= 1:
                        raise ValueError(
                            "Question/prompt and minimum context exceed LLM_NUM_CTX; shorten the question or increase LLM_NUM_CTX."
                        )
                    result = results[0].model_copy(deep=True)
                    result.chunk.text = "\n".join(lines[:-1])
                    results = [result]
        context_str = format_context_blocks(results)
        prepared_at = perf_counter()
        retrieval_ms = max(0.0, (prepared_at - start_t) * 1000.0)
        return PreparedQuery(
            query,
            plan,
            results,
            context_str,
            fast_answer,
            mode,
            retrieval_ms=round(retrieval_ms, 2),
            prepared_at=prepared_at,
        )

    def _prompt_tokens(self, query: str, results: list[RetrievalResult]) -> int:
        prompt = self.prompt.invoke({"context": format_context_blocks(results), "question": query})
        return estimate_tokens(prompt.to_string()) + 32  # chat template safety margin

    def stream(self, prepared: PreparedQuery) -> Iterator[str]:
        if prepared.answer_mode == AnswerMode.FAST_PATH:
            yield prepared.fast_answer or ""
        elif prepared.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT:
            yield INSUFFICIENT_CONTEXT_ANSWER
        else:
            yield from self.chain.stream({"context": prepared.context, "question": prepared.query})

    async def astream(self, prepared: PreparedQuery) -> AsyncIterator[str]:
        if prepared.answer_mode != AnswerMode.LLM:
            for answer in self.stream(prepared):
                yield answer
        else:
            async for token in self.chain.astream(
                {"context": prepared.context, "question": prepared.query}
            ):
                yield token

    def finalize(self, prepared: PreparedQuery, answer: str) -> RAGResponse:
        provider_label, model_label = self._resolve_model_metadata()
        text = str(answer).strip()
        mode = prepared.answer_mode
        report = validate_citations(text, prepared.results)
        used = set(report.used_chunk_ids)
        citations = [
            item.model_copy(update={"used": item.chunk_id in used})
            for item in build_citations(prepared.results)
        ]
        citations.sort(key=lambda item: not item.used)  # stable: used first

        # Fast-path answers are verbatim BSB text and are never rewritten.
        if self.settings.citation_policy == "strict" and mode != AnswerMode.FAST_PATH:
            text = strip_unsupported_citations(text, report.unsupported)
            if (
                mode == AnswerMode.LLM
                and not (report.valid or report.partial)
                and INSUFFICIENT_CONTEXT_ANSWER not in text
            ):
                text = INSUFFICIENT_CONTEXT_ANSWER
                mode = AnswerMode.INSUFFICIENT_CONTEXT

        now = perf_counter()
        retrieval_ms = max(0.0, float(getattr(prepared, "retrieval_ms", 0.0)))
        prepared_at = float(getattr(prepared, "prepared_at", 0.0))
        generation_ms = (
            max(0.0, round((now - prepared_at) * 1000.0, 2))
            if prepared_at > 0.0
            else 0.0
        )
        total_ms = round(retrieval_ms + generation_ms, 2)

        return RAGResponse(
            query=prepared.query,
            answer=text,
            intent=self._extract_intent(prepared.plan),
            provider=provider_label,
            model=model_label,
            effort=self.effort,
            answer_mode=mode,
            citations=citations,
            citation_report=report,
            retrieved_chunks=prepared.results,
            timings={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        )

    def ask(self, query: str, top_k: int | None = None) -> RAGResponse:
        prepared = self.prepare(query, top_k)
        return self.finalize(prepared, "".join(self.stream(prepared)))

    async def aask(self, query: str, top_k: int | None = None) -> RAGResponse:
        prepared = await asyncio.to_thread(self.prepare, query, top_k)
        answer = "".join([token async for token in self.astream(prepared)])
        return self.finalize(prepared, answer)
