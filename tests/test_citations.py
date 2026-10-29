"""Citation validation (valid / partial / unsupported / unparsed) and annotate/strict policies."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from beresbirag.config import Settings
from beresbirag.llm.citations import extract_citations, parse_label, validate_citations
from beresbirag.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from beresbirag.llm.rag_chain import BSBRAGPipeline, PreparedQuery, format_context_blocks
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import route_query


def _result(chunk_id: str, text: str, **meta) -> RetrievalResult:
    return RetrievalResult(
        chunk=ChunkDocument(
            chunk_id=chunk_id, text=text, metadata=ChunkMetadata(chunk_id=chunk_id, **meta)
        )
    )


GEN_1_1_5 = _result(
    "scripture:GEN:1:1-5:0",
    "[Genesis 1:1] In the beginning God created the heavens and the earth.",
    source_type=SourceType.SCRIPTURE,
    book_code="GEN",
    chapter=1,
    verse_start=1,
    verse_end=5,
    canonical_ref="Genesis 1:1-5",
    verse_refs=[f"GEN 1:{v}" for v in range(1, 6)],
)
JHN_3_16 = _result(
    "scripture:JHN:3:16-16:0",
    "[John 3:16] For God so loved the world.",
    source_type=SourceType.SCRIPTURE,
    book_code="JHN",
    chapter=3,
    verse_start=16,
    verse_end=16,
    canonical_ref="John 3:16",
    verse_refs=["JHN 3:16"],
)
TOPIC_666 = _result(
    "topical:666:0",
    "[Source: BSB Topical Index | Topic: 666]",
    source_type=SourceType.TOPICAL_INDEX,
    canonical_ref="Topical Index: 666",
    topic="666",
)
TOPIC_BAPTISMS = _result(
    "topical:Baptisms:0",
    "[Source: BSB Topical Index | Topic: Baptisms]",
    source_type=SourceType.TOPICAL_INDEX,
    canonical_ref="Topical Index: Baptisms",
    topic="baptisms",
)
CONC_12 = _result(
    "concordance:12:0",
    '[Source: BSB Concordance | Headword: "12"]',
    source_type=SourceType.CONCORDANCE,
    canonical_ref="Concordance: 12 (26 Occurrences)",
    concordance_word="12",
)
OUTLINE_GEN = _result(
    "outline:GEN:overview",
    "[Source: BSB Book Outline | Book: Genesis (GEN)]",
    source_type=SourceType.BOOK_OUTLINE,
    book_code="GEN",
    canonical_ref="Genesis Outline (1:1-50:26)",
)


class _Retriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int | None = None):
        return route_query(query), self.results


def _pipeline(answer: str, results: list[RetrievalResult], policy: str) -> BSBRAGPipeline:
    return BSBRAGPipeline(
        retriever=_Retriever(results),
        llm=FakeListChatModel(responses=[answer]),
        settings=Settings(_env_file=None, CITATION_POLICY=policy),
        provider="openai",
        model_name="gpt-4o-mini",
    )


def test_extract_citations_dedupes_in_order() -> None:
    answer = "A [Genesis 1:1], B [1], again [Genesis 1:1] and [ see above ]."
    assert extract_citations(answer) == ["Genesis 1:1", "1", "see above"]


def test_scripture_citation_classification() -> None:
    answer = (
        "Creation [Genesis 1:1]. Light and more [Genesis 1:4-8]. "
        "Invented [Revelation 99:99]. Vague [see above]. Elsewhere [John 3:16]."
    )
    report = validate_citations(answer, [GEN_1_1_5])
    assert report.valid == ["Genesis 1:1"]
    assert report.partial == ["Genesis 1:4-8"]
    assert report.unsupported == ["Revelation 99:99", "John 3:16"]
    assert report.unparsed == ["see above"]
    assert report.used_chunk_ids == ["scripture:GEN:1:1-5:0"]
    assert report.has_citations


def test_multi_reference_and_fabricated_segments() -> None:
    report = validate_citations(
        "[Genesis 1:1; John 3:16] [Genesis 1:2; Revelation 99:99]", [GEN_1_1_5]
    )
    assert report.partial == ["Genesis 1:1; John 3:16", "Genesis 1:2; Revelation 99:99"]
    both = validate_citations("[Genesis 1:1; John 3:16]", [GEN_1_1_5, JHN_3_16])
    assert both.valid == ["Genesis 1:1; John 3:16"]
    assert set(both.used_chunk_ids) == {GEN_1_1_5.chunk.chunk_id, JHN_3_16.chunk.chunk_id}


def test_numeric_citations() -> None:
    assert validate_citations("[2]", [GEN_1_1_5]).unsupported == ["2"]
    report = validate_citations("[1]", [GEN_1_1_5])
    assert report.valid == ["1"]
    assert report.used_chunk_ids == [GEN_1_1_5.chunk.chunk_id]


def test_label_citations_checked_against_chunk_metadata() -> None:
    results = [TOPIC_666, TOPIC_BAPTISMS, CONC_12, OUTLINE_GEN]
    answer = (
        "[BSB Topical Index: 666] [Topical Index: Grace] [BSB Topical Index: Baptism] "
        "[BSB Concordance: 12] [Concordance: 13] [BSB Book Outline: Genesis] "
        "[Book Outline: Exodus] [Concordance: 12 (26 Occurrences)] [Genesis Outline (1:1-50:26)]"
    )
    report = validate_citations(answer, results)
    assert report.valid == [
        "BSB Topical Index: 666",
        "BSB Topical Index: Baptism",
        "BSB Concordance: 12",
        "BSB Book Outline: Genesis",
        "Concordance: 12 (26 Occurrences)",
        "Genesis Outline (1:1-50:26)",
    ]
    assert report.unsupported == ["Topical Index: Grace", "Concordance: 13", "Book Outline: Exodus"]
    assert set(report.used_chunk_ids) == {
        "topical:666:0", "topical:Baptisms:0", "concordance:12:0", "outline:GEN:overview",
    }  # fmt: skip


def test_parse_label() -> None:
    assert parse_label("BSB Topical Index: Grace (Part 2)") == ("topic", "grace")
    assert parse_label("Concordance: 12 (26 Occurrences)") == ("concordance", "12")
    assert parse_label("Book Outline: 1 John") == ("outline", "1JN")
    assert parse_label("Romans Outline (1:1-16:27)") == ("outline", "ROM")
    assert parse_label("Genesis 1:1") is None


def test_no_citations() -> None:
    report = validate_citations("No brackets here.", [GEN_1_1_5])
    assert not report.has_citations
    assert report.used_chunk_ids == []


def test_annotate_policy_keeps_text_and_orders_used_first() -> None:
    answer = "God created [Genesis 1:1] but see [Revelation 99:99]."
    response = _pipeline(answer, [JHN_3_16, GEN_1_1_5], "annotate").ask("What does Genesis say?")
    assert response.answer == answer
    assert response.answer_mode == AnswerMode.LLM
    assert [c.chunk_id for c in response.citations] == [
        GEN_1_1_5.chunk.chunk_id,
        JHN_3_16.chunk.chunk_id,
    ]
    assert [c.used for c in response.citations] == [True, False]
    assert response.citation_report is not None
    assert response.citation_report.unsupported == ["Revelation 99:99"]


def test_strict_policy_removes_unsupported_and_appends_footer() -> None:
    answer = "God created [Genesis 1:1] and loved the world [John 3:16]."
    response = _pipeline(answer, [GEN_1_1_5], "strict").ask("What does Genesis say?")
    assert "[John 3:16]" not in response.answer.split("\n\n>")[0]
    assert response.answer.startswith("God created [Genesis 1:1] and loved the world.")
    assert "Removed 1 citation(s) not found in the retrieved BSB context: [John 3:16]" in (
        response.answer
    )
    assert response.answer_mode == AnswerMode.LLM


def test_strict_policy_only_unsupported_becomes_insufficient_context() -> None:
    answer = "It says so [Revelation 99:99] and [John 3:16]."
    response = _pipeline(answer, [GEN_1_1_5], "strict").ask("What does Genesis say?")
    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT
    assert response.citation_report is not None
    assert response.citation_report.unsupported == ["Revelation 99:99", "John 3:16"]


@pytest.mark.parametrize("policy", ["annotate", "strict"])
def test_fast_path_answers_are_never_rewritten(policy: str) -> None:
    pipeline = _pipeline("unused", [GEN_1_1_5], policy)
    prepared = PreparedQuery(
        query="Genesis 1:1",
        plan=route_query("Genesis 1:1"),
        results=[GEN_1_1_5],
        context=format_context_blocks([GEN_1_1_5]),
        fast_answer="**Genesis 1:1** (BSB) [John 3:16]",
        answer_mode=AnswerMode.FAST_PATH,
    )
    response = pipeline.finalize(prepared, "**Genesis 1:1** (BSB) [John 3:16]")
    assert response.answer == "**Genesis 1:1** (BSB) [John 3:16]"
    assert response.answer_mode == AnswerMode.FAST_PATH
    assert response.citation_report is not None
    assert response.citation_report.unsupported == ["John 3:16"]
