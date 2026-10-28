"""Bounded topic aggregation for the BSB Topical Index workbook."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from beresbirag.config import get_settings, resolve_data_file
from beresbirag.ingestion.xlsx_stream import _slug, _verse_keys, iter_xlsx_rows
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType


def _topic_chunks(
    topic: str, passages: list[tuple[str, str]], refs: list[str], total: int, size: int
) -> Iterator[ChunkDocument]:
    parts = max(1, (len(passages) + size - 1) // size)
    for part_idx in range(parts):
        selected = passages[part_idx * size : (part_idx + 1) * size]
        selected_refs = [ref for ref, _ in selected]
        additional = [ref for ref in refs if ref not in selected_refs][:50]
        lines = [
            f"[Source: BSB Topical Index | Topic: {topic} | Total References: {total}]",
            f'Key Scripture Passages for Topic "{topic}":',
            *(f"- [{ref}] {context}" for ref, context in selected),
        ]
        if additional:
            lines.append("Additional References: " + "; ".join(additional))
        lines.append("All Verse References: " + "; ".join(dict.fromkeys(selected_refs + additional)))
        chunk_id = f"topical:{_slug(topic)}:{part_idx}"
        canonical = f"Topical Index: {topic}"
        if parts > 1:
            canonical += f" (Part {part_idx + 1})"
        yield ChunkDocument(
            chunk_id=chunk_id,
            text="\n".join(lines),
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                source_type=SourceType.TOPICAL_INDEX,
                canonical_ref=canonical,
                topic=topic.lower(),
                topic_display=topic,
                verse_refs=_verse_keys(selected_refs + additional),
            ),
        )


def iter_topical_chunks(
    xlsx_path: str | Path | None = None,
    max_topics: int | None = None,
    max_full_verses_per_chunk: int = 15,
    max_chunks_per_topic: int = 3,
) -> Iterator[ChunkDocument]:
    """Emit capped passage chunks per topic while retaining at most 50 extra citations."""
    if max_full_verses_per_chunk <= 0 or max_chunks_per_topic <= 0:
        raise ValueError("Chunk size and maximum chunks must be positive")
    if max_topics is not None and max_topics <= 0:
        return
    path = resolve_data_file(xlsx_path) if xlsx_path is not None else get_settings().topical_xlsx_file
    topic = ""
    passages: list[tuple[str, str]] = []
    refs: list[str] = []
    total = completed = 0
    capacity = max_full_verses_per_chunk * max_chunks_per_topic
    with closing(iter_xlsx_rows(path)) as rows:
        for _, row in rows:
            if row.get("B") == "Top" and row.get("C", "").strip():
                if topic:
                    yield from _topic_chunks(topic, passages, refs, total, max_full_verses_per_chunk)
                    completed += 1
                    if max_topics is not None and completed >= max_topics:
                        return
                topic = row["C"].strip()
                passages, refs, total = [], [], 0
            elif topic and row.get("E", "").strip():
                ref, context = row["E"].strip(), row.get("F", "").strip()
                total += 1
                if len(refs) < capacity + 50 and ref not in refs:
                    refs.append(ref)
                if context and len(passages) < capacity:
                    passages.append((ref, context))
        if topic:
            yield from _topic_chunks(topic, passages, refs, total, max_full_verses_per_chunk)
