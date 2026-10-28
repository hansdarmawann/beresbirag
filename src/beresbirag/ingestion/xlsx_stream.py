"""Stream OpenXML rows without loading a worksheet into memory."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from beresbirag.models.books import resolve_book

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REF = re.compile(r"^(.+?)\s+(\d+):(\d+)(?:\s*[-–—]\s*(\d+))?$")


def iter_xlsx_rows(
    xlsx_path: str | Path, max_rows: int | None = None
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield numbered rows from sheet1; max_rows counts physical rows, including headers.

    The shared-string lookup remains resident, but completed XML elements are
    detached from their parents so worksheet memory does not grow with row count.
    """
    if max_rows is not None and max_rows <= 0:
        return
    with ZipFile(xlsx_path) as archive:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                events = ET.iterparse(stream, events=("start", "end"))
                _, root = next(events)
                for event, el in events:
                    if event == "end" and el.tag == f"{_NS}si":
                        strings.append("".join(el.itertext()))
                        el.clear()
                        root.remove(el)

        with archive.open("xl/worksheets/sheet1.xml") as stream:
            sheet_data = None
            count = 0
            for event, el in ET.iterparse(stream, events=("start", "end")):
                if event == "start" and el.tag == f"{_NS}sheetData":
                    sheet_data = el
                if event != "end" or el.tag != f"{_NS}row":
                    continue
                row: dict[str, str] = {}
                row_num = int(el.attrib.get("r", 0))
                for cell in el:
                    if cell.tag != f"{_NS}c":
                        continue
                    col = "".join(ch for ch in cell.attrib.get("r", "") if ch.isalpha())
                    value = cell.findtext(f"{_NS}v", default="")
                    inline = cell.find(f"{_NS}is")
                    if cell.get("t") == "s" and value:
                        value = strings[int(value)]
                    elif inline is not None:
                        value = "".join(inline.itertext())
                    row[col] = value
                el.clear()
                if sheet_data is not None:
                    sheet_data.remove(el)
                yield row_num, row
                count += 1
                if max_rows is not None and count >= max_rows:
                    return


def normalize_ref_to_usj_keys(raw_ref: str, max_span: int = 25) -> list[str]:
    """Expand a single verse or same-chapter range, capped at max_span.

    Unknown books, malformed citations, and reversed ranges return no keys.
    Cross-chapter ranges require verse-count data and are not expanded here.
    """
    match = _REF.fullmatch(raw_ref.strip())
    if match is None or max_span <= 0:
        return []
    book = resolve_book(match[1])
    chapter, first = int(match[2]), int(match[3])
    last = int(match[4]) if match[4] else first
    if book is None or chapter < 1 or first < 1 or last < first:
        return []
    return [f"{book.code} {chapter}:{v}" for v in range(first, min(last + 1, first + max_span))]


def _slug(display: str) -> str:
    """Keep readable words while preserving punctuation and case to avoid collisions."""
    return quote(display, safe="-_")


def _verse_keys(refs: list[str], limit: int = 60) -> list[str]:
    keys: dict[str, None] = {}
    for ref in refs:
        for key in normalize_ref_to_usj_keys(ref):
            keys.setdefault(key, None)
            if len(keys) >= limit:
                return list(keys)
    return list(keys)
