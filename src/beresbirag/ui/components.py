"""Reusable UI components and rendering helpers for Streamlit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from beresbirag.diagnostics import CheckResult

from beresbirag.llm.factory import (
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
)
from beresbirag.models.schemas import (
    AnswerMode,
    CitationItem,
    CitationReport,
    QueryIntent,
    RetrievalResult,
)
from beresbirag.retrieval.query_router import QueryPlan
from beresbirag.vectorstore.qdrant_manager import CollectionConfigError


def render_intent_badge(plan: QueryPlan | QueryIntent | str | None) -> None:
    """Render a badge showing the classified query intent."""
    if plan is None:
        intent: Any = QueryIntent.HYBRID_GENERAL
    elif hasattr(plan, "intent"):
        intent = plan.intent
    else:
        intent = plan

    intent_val = intent.value if hasattr(intent, "value") else str(intent)
    st.markdown(
        f'<span style="background-color: #2b313e; color: #58a6ff; padding: 3px 8px; '
        f'border-radius: 6px; font-size: 0.85em; font-weight: 600;">'
        f"Intent: {intent_val}</span>",
        unsafe_allow_html=True,
    )


def render_citation_report(
    report: CitationReport | dict[str, Any] | None,
    *,
    citation_policy: str = "annotate",
    answer_mode: AnswerMode | str | None = None,
) -> None:
    """Render valid, partial, and unsupported citation counts as colored chips, plus strict warnings."""
    if report is None:
        return
    parsed = (
        CitationReport.model_validate(report)
        if isinstance(report, dict)
        else report
    )
    valid_count = len(parsed.valid)
    partial_count = len(parsed.partial)
    unsupported_count = len(parsed.unsupported)
    unsupported_suffix = (
        f" ({', '.join(parsed.unsupported)})" if parsed.unsupported else ""
    )

    chips_html = (
        '<div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0;">'
        f'<span style="background-color: #1f4d2e; color: #7ee787; padding: 3px 10px; '
        f'border-radius: 12px; font-size: 0.85em; font-weight: 600;">'
        f"Valid: {valid_count}</span>"
        f'<span style="background-color: #4d3810; color: #f2cc60; padding: 3px 10px; '
        f'border-radius: 12px; font-size: 0.85em; font-weight: 600;">'
        f"Partial: {partial_count}</span>"
        f'<span style="background-color: #5c1e24; color: #ffa198; padding: 3px 10px; '
        f'border-radius: 12px; font-size: 0.85em; font-weight: 600;">'
        f"Unsupported: {unsupported_count}{unsupported_suffix}</span>"
        "</div>"
    )
    st.markdown(chips_html, unsafe_allow_html=True)

    mode_str = (
        answer_mode.value if hasattr(answer_mode, "value") else str(answer_mode or "")
    )
    if citation_policy == "strict" and (
        unsupported_count > 0 or mode_str == AnswerMode.INSUFFICIENT_CONTEXT.value
    ):
        st.warning(
            "Strict citation policy active: unsupported citations were stripped or the answer "
            "was replaced because no grounded citation remained."
        )


def render_citations_table(citations: list[CitationItem]) -> None:
    """Render citations in a clean table format with used citations highlighted."""
    if not citations:
        return

    table_data = []
    for idx, cit in enumerate(citations, start=1):
        src_val = (
            cit.source_type.value
            if hasattr(cit.source_type, "value")
            else str(cit.source_type)
        )
        ref_label = f"★ {cit.canonical_ref}" if cit.used else cit.canonical_ref
        table_data.append(
            {
                "#": idx,
                "Used": "✅ Yes" if cit.used else "No",
                "Reference": ref_label,
                "Source Type": src_val,
                "Score": f"{cit.score:.3f}",
                "Chunk ID": cit.chunk_id,
            }
        )

    st.subheader("Citations & Grounding Sources")
    st.dataframe(table_data, use_container_width=True, hide_index=True)


def render_retrieved_chunks(results: list[RetrievalResult]) -> None:
    """Render retrieved chunks as individual expanders with literal text content."""
    if not results:
        st.info("No relevant chunks retrieved.")
        return

    st.subheader(f"Retrieved Chunks ({len(results)})")
    for res in results:
        meta = res.chunk.metadata
        ref = meta.canonical_ref or res.chunk.chunk_id
        src_type = (
            meta.source_type.value
            if hasattr(meta.source_type, "value")
            else str(meta.source_type)
        )
        score_val = (
            f"{res.rerank_score:.3f}"
            if res.rerank_score is not None
            else f"{res.score:.3f}"
        )
        title = f"#{res.rank} · {ref} · {src_type} · {score_val}"

        with st.expander(title):
            # Render using st.text to preserve literal square brackets and formatting
            st.text(res.chunk.text)


def render_status_panel(checks: list[CheckResult]) -> None:
    """Render diagnostic check results with status icons."""
    icon_map = {
        "pass": "✅",
        "warn": "⚠️",
        "fail": "❌",
    }
    for check in checks:
        icon = icon_map.get(check.status, "ℹ️")
        st.write(f"**{icon} {check.name}**: {check.detail}")


def render_error(exc: Exception) -> None:
    """Map known exceptions to actionable error notices in Streamlit."""
    if isinstance(exc, MissingAPIKeyError):
        st.error(f"Missing API Key: {exc}")
    elif isinstance(exc, ProviderModelMismatchError):
        st.error(f"Provider/Model Mismatch: {exc} Please choose a model from the list.")
    elif isinstance(exc, ProviderUnavailableError):
        st.error(f"Provider Unavailable: {exc} (offline mode: use Ollama)")
    elif isinstance(exc, CollectionConfigError):
        st.error(f"Collection Configuration Error: {exc} Recreate collection with --recreate.")
    elif (
        isinstance(exc, ConnectionError)
        or "Failed to establish a new connection" in str(exc)
        or "ConnectError" in exc.__class__.__name__
    ):
        st.error(f"Connection Error: {exc} Check services or run `beresbirag doctor`.")
    else:
        st.error(f"Error: {exc}")
