"""Synthetic OpenXML cases and acceptance checks against the original BSB files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest

from beresbirag.config import resolve_data_file
from beresbirag.ingestion.concordance_parser import iter_concordance_chunks
from beresbirag.ingestion.topical_parser import iter_topical_chunks
from beresbirag.ingestion.xlsx_stream import iter_xlsx_rows, normalize_ref_to_usj_keys
from beresbirag.models.schemas import ChunkDocument, SourceType

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def workbook(tmp_path: Path, rows: list[dict[str, str]], shared: bool = False) -> Path:
    """Build an XLSX ZIP in memory using inline strings or rich shared strings."""
    sheet = ET.Element("worksheet", xmlns=_NS)
    data = ET.SubElement(sheet, "sheetData")
    strings = ET.Element("sst", xmlns=_NS)
    index = 0
    for row_num, values in enumerate(rows, 1):
        row = ET.SubElement(data, "row", r=str(row_num))
        for col, text in values.items():
            cell = ET.SubElement(row, "c", r=f"{col}{row_num}", t="s" if shared else "inlineStr")
            if shared:
                ET.SubElement(cell, "v").text = str(index)
                container = ET.SubElement(strings, "si")
                index += 1
            else:
                container = ET.SubElement(cell, "is")
            # Rich text runs must concatenate without losing spaces.
            for piece in (text[:2], text[2:]):
                ET.SubElement(ET.SubElement(container, "r"), "t").text = piece
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))
        if shared:
            archive.writestr("xl/sharedStrings.xml", ET.tostring(strings))
    path = tmp_path / "fixture.xlsx"
    path.write_bytes(buffer.getvalue())
    return path


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("Revelation 13:16–18", ["REV 13:16", "REV 13:17", "REV 13:18"]),
        ("2 Chronicles 9:13", ["2CH 9:13"]),
        ("1Ch 25:9", ["1CH 25:9"]),
        (" Gen. 1:1—2 ", ["GEN 1:1", "GEN 1:2"]),
        ("John 3:16-15", []),
        ("Unknown 1:1", []),
        ("Genesis 0:1", []),
        ("Genesis 1:0", []),
        ("Genesis 1:1-2:3", []),
        ("header", []),
        ("", []),
    ],
)
def test_reference_normalization(ref: str, expected: list[str]) -> None:
    assert normalize_ref_to_usj_keys(ref) == expected


def test_reference_span_is_bounded() -> None:
    assert normalize_ref_to_usj_keys("Ps 119:1-176", 2) == ["PSA 119:1", "PSA 119:2"]
    assert normalize_ref_to_usj_keys("Gen 1:1", 0) == []


@pytest.mark.parametrize("shared", [False, True])
def test_row_stream(tmp_path: Path, shared: bool) -> None:
    rows = [{"A": "hello world", "AA": "", "F": "a & b"}, {}, {"B": "last"}]
    path = workbook(tmp_path, rows, shared)
    assert list(iter_xlsx_rows(path)) == list(enumerate(rows, 1))
    assert list(iter_xlsx_rows(path, max_rows=1)) == [(1, rows[0])]
    assert list(iter_xlsx_rows(path, max_rows=0)) == []


def test_numeric_cached_formula_and_blank_cells(tmp_path: Path) -> None:
    path = tmp_path / "numeric.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{_NS}"><sheetData><row r="7">'
            '<c r="A7"><v>12</v></c><c r="B7"><f>1+1</f><v>2</v></c>'
            '<c r="C7"/></row></sheetData></worksheet>',
        )
    assert list(iter_xlsx_rows(path)) == [(7, {"A": "12", "B": "2", "C": ""})]


def test_topical_parts_caps_and_final_flush(tmp_path: Path) -> None:
    rows = [{"B": "Source", "C": "Topic"}, {"B": "Top", "C": " Grace "}]
    rows += [{"E": f"John 1:{v}", "F": f"Passage {v}"} for v in range(1, 8)]
    rows += [{"E": "John 1:1"}, {"E": "Genesis 1:1"}, {"B": "Top", "C": "Empty"}]
    path = workbook(tmp_path, rows)
    chunks = list(iter_topical_chunks(path, max_full_verses_per_chunk=2, max_chunks_per_topic=3))
    assert len(chunks) == 4
    assert [c.chunk_id for c in chunks[:3]] == [f"topical:Grace:{n}" for n in range(3)]
    for idx, chunk in enumerate(chunks[:3]):
        assert chunk.metadata.topic == "grace"
        assert chunk.metadata.topic_display == "Grace"
        assert chunk.metadata.canonical_ref == f"Topical Index: Grace (Part {idx + 1})"
        assert chunk.text.count("- [") == 2
        assert f"JHN 1:{2 * idx + 1}" in chunk.metadata.verse_refs
        assert len(chunk.metadata.verse_refs) == len(set(chunk.metadata.verse_refs))
        assert "Total References: 9" in chunk.text
    assert "Passage 7" not in "\n".join(c.text for c in chunks)
    assert chunks[-1].metadata.topic == "empty"
    assert len(list(iter_topical_chunks(path, max_topics=1))) == 1
    assert list(iter_topical_chunks(path, max_topics=0)) == []


def test_topical_citation_and_metadata_limits(tmp_path: Path) -> None:
    rows = [{"B": "Top", "C": "References"}]
    rows += [{"E": f"Psalm 119:{v}"} for v in range(1, 151)]
    chunk = next(iter_topical_chunks(workbook(tmp_path, rows)))
    assert "Total References: 150" in chunk.text
    citations = chunk.text.split("Additional References: ")[1].splitlines()[0].split("; ")
    assert len(citations) == 50
    assert len(chunk.metadata.verse_refs) == 50


def test_concordance_counts_fallback_and_limits(tmp_path: Path) -> None:
    rows = [{"G": "Entry", "E": "Occ"}, {"E": "0", "G": " Love (100 Occurrences)"}]
    rows += [{"B": "Gen", "H": f"Gen 1:{v}", "I": f"Context {v}"} for v in range(1, 71)]
    rows += [{"B": "Mat", "H": f"Mat 1:{v}", "I": "More"} for v in range(1, 31)]
    rows += [{"E": "0", "G": "Fallback", "F": "1"}, {"H": "1Ch 25:9", "I": "last"}]
    path = workbook(tmp_path, rows, shared=True)
    chunks = list(iter_concordance_chunks(path, max_snippets=3))
    assert len(chunks) == 2
    assert chunks[0].metadata.concordance_word == "love"
    assert chunks[0].metadata.concordance_occurrences == 100
    assert "Gen (70), Mat (30)" in chunks[0].text
    assert chunks[0].text.count("- [") == 3
    assert len(chunks[0].text.split("Verse Citations: ")[1].split("; ")) == 80
    assert len(chunks[0].metadata.verse_refs) == 60
    assert chunks[1].metadata.concordance_occurrences == 1
    assert chunks[1].metadata.verse_refs == ["1CH 25:9"]
    assert "1Ch (1)" in chunks[1].text
    assert len(list(iter_concordance_chunks(path, max_entries=1))) == 1
    assert list(iter_concordance_chunks(path, max_entries=0)) == []
    assert "- [" not in next(iter_concordance_chunks(path, max_snippets=0)).text


def test_empty_workbooks_and_invalid_options(tmp_path: Path) -> None:
    path = workbook(tmp_path, [])
    assert list(iter_topical_chunks(path)) == []
    assert list(iter_concordance_chunks(path)) == []
    with pytest.raises(ValueError):
        list(iter_topical_chunks(path, max_full_verses_per_chunk=0))
    with pytest.raises(ValueError):
        list(iter_topical_chunks(path, max_chunks_per_topic=0))
    with pytest.raises(ValueError):
        list(iter_concordance_chunks(path, max_snippets=-1))


def test_punctuation_does_not_collide(tmp_path: Path) -> None:
    path = workbook(tmp_path, [{"B": "Top", "C": name} for name in ("A B", "A-B", "A/B")])
    assert len({chunk.chunk_id for chunk in iter_topical_chunks(path)}) == 3


def test_real_topical_workbook() -> None:
    if not resolve_data_file("bsb_topical_index.xlsx").is_file():
        pytest.skip("Original BSB Topical Index workbook is unavailable")
    start = perf_counter()
    chunks = list(iter_topical_chunks(max_topics=15))
    elapsed = perf_counter() - start
    assert elapsed < 5, f"15 topics took {elapsed:.2f}s"
    assert len({c.metadata.topic for c in chunks}) == 15
    assert all(isinstance(c, ChunkDocument) for c in chunks)
    assert all(c.metadata.source_type == SourceType.TOPICAL_INDEX for c in chunks)
    first = chunks[0]
    assert first.metadata.topic == "666"
    assert {"REV 13:16", "2CH 9:13"} <= set(first.metadata.verse_refs)
    assert "And the second beast required all people" in first.text


def test_real_concordance_workbook() -> None:
    if not resolve_data_file("bsb_concordance.xlsx").is_file():
        pytest.skip("Original BSB Concordance workbook is unavailable")
    chunks = list(iter_concordance_chunks(max_entries=15))
    assert len(chunks) == 15
    assert all(isinstance(c, ChunkDocument) for c in chunks)
    assert all(c.metadata.source_type == SourceType.CONCORDANCE for c in chunks)
    assert chunks[0].metadata.concordance_word == "10"
    assert chunks[0].metadata.concordance_occurrences == 2
    assert "GEN 32:15" in chunks[0].metadata.verse_refs
    assert chunks[1].metadata.concordance_word == "12"
    assert chunks[1].metadata.concordance_occurrences == 26
    assert "1CH 25:9" in chunks[1].metadata.verse_refs
