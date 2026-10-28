"""Ingestion parsers and unified pipeline for the 4 Berean Standard Bible (BSB) corpora."""

from __future__ import annotations

from beresbirag.ingestion.concordance_parser import iter_concordance_chunks
from beresbirag.ingestion.pdf_outline_parser import (
    BookOutlineIndex,
    OutlineNode,
    parse_book_outlines,
    parse_verse_range,
)
from beresbirag.ingestion.pipeline import (
    DEFAULT_SOURCES,
    IngestionPipeline,
    IngestionStats,
    parse_book_list,
    parse_source_list,
)
from beresbirag.ingestion.topical_parser import iter_topical_chunks
from beresbirag.ingestion.usj_parser import iter_usj_scripture_chunks, parse_usj_book
from beresbirag.ingestion.xlsx_stream import iter_xlsx_rows, normalize_ref_to_usj_keys

__all__ = [
    "BookOutlineIndex",
    "DEFAULT_SOURCES",
    "IngestionPipeline",
    "IngestionStats",
    "OutlineNode",
    "iter_concordance_chunks",
    "iter_topical_chunks",
    "iter_usj_scripture_chunks",
    "iter_xlsx_rows",
    "normalize_ref_to_usj_keys",
    "parse_book_list",
    "parse_book_outlines",
    "parse_source_list",
    "parse_usj_book",
    "parse_verse_range",
]

