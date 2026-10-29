"""Rank-preserving context compression without changing stored corpus text."""

from __future__ import annotations

import re

from beresbirag.models.schemas import RetrievalResult, SourceType


def estimate_tokens(text: str) -> int:
    """Corpus-based estimate, not a model-specific tokenizer guarantee."""
    return (len(text) + 3) // 4


def _clean_text(result: RetrievalResult, question: str) -> str:
    source = result.chunk.metadata.source_type
    keep_notes = bool(re.search(r"footnote|manuscript|literally|variant|note", question, re.I))
    keep_cross = bool(re.search(r"cross.?ref|parallel|cited", question, re.I))
    lines = []
    in_notes = False
    for line in result.chunk.text.splitlines():
        stripped = line.strip()
        if source == SourceType.SCRIPTURE:
            if stripped.startswith("Footnotes:"):
                in_notes = True
            if in_notes and not keep_notes:
                continue
            if not keep_cross:
                closing = "]" if line.startswith("[") and line.endswith("]") else ""
                line = re.sub(r"(?:\s*\|\s*)?Cross-Refs:.*$", closing, line)
        elif source == SourceType.TOPICAL_INDEX:
            if stripped.startswith(("Additional References:", "All Verse References:")):
                continue
        elif source == SourceType.CONCORDANCE and stripped.startswith("Verse Citations:"):
            continue
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def fit_context(
    results: list[RetrievalResult],
    question: str,
    total_budget: int | None,
    per_chunk_cap: int | None,
) -> list[RetrievalResult]:
    """Fit whole lines in rank order; retain the first line if no line fits.

    The at-least-one rule takes precedence for an indivisible oversized line.
    The pipeline separately checks the complete Ollama prompt before generation.
    """
    if total_budget is None and per_chunk_cap is None:
        return results
    fitted: list[RetrievalResult] = []
    used = 0
    for result in results:
        text = _clean_text(result, question)
        lines = text.splitlines() or [result.chunk.text.splitlines()[0]]
        if per_chunk_cap is not None:
            kept: list[str] = []
            for line in lines:
                if estimate_tokens("\n".join([*kept, line])) > per_chunk_cap:
                    break
                kept.append(line)
            text = "\n".join(kept or lines[:1])
        else:
            text = "\n".join(lines)
        cost = estimate_tokens(text)
        if total_budget is not None and used + cost > total_budget and fitted:
            break
        copy = result.model_copy(deep=True)
        copy.chunk.text = text
        fitted.append(copy)
        used += cost
        if total_budget is not None and used >= total_budget:
            break
    return fitted
