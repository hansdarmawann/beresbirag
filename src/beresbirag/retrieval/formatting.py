"""Context formatting utilities for retrieval results."""

from __future__ import annotations

from beresbirag.models.schemas import RetrievalResult


def format_context_blocks(
    results: list[RetrievalResult],
    empty_message: str | None = None,
) -> str:
    """Format retrieved results into numbered, citation-anchored context blocks.

    If results is empty:
    - returns empty_message if provided
    - returns "" if empty_message is None
    """
    if not results:
        return empty_message or ""

    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        meta = result.chunk.metadata
        reference = meta.canonical_ref or result.chunk.chunk_id
        source_type_val = (
            meta.source_type.value if hasattr(meta.source_type, "value") else str(meta.source_type)
        )
        header = f"=== SOURCE [{idx}] | Reference: {reference} | Type: {source_type_val} ==="
        blocks.append(f"{header}\n{result.chunk.text.strip()}")
    return "\n\n".join(blocks)
