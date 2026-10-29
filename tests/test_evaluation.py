"""Evaluation cases, metrics, ground-truth verification, and the two evaluation CLIs."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from beresbirag.evaluation import (
    AnswerCase,
    CaseFileError,
    CorpusIndex,
    RetrievalCase,
    citation_precision,
    context_coverage,
    has_false_exact_match,
    hit_rank,
    is_refusal,
    load_answer_cases,
    load_cases,
    load_retrieval_cases,
    mrr,
    must_cite_recall,
    percentile,
    recall_at_k,
    verify_cases,
    verse_key_exists,
)
from beresbirag.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    CitationReport,
    RAGResponse,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import route_query

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _load_script(name: str) -> Any:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scripture(chunk_id: str, refs: list[str], text: str = "text") -> RetrievalResult:
    return RetrievalResult(
        chunk=ChunkDocument(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                source_type=SourceType.SCRIPTURE,
                book_code=refs[0].split()[0],
                verse_refs=refs,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Case files
# ---------------------------------------------------------------------------


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "X1", "category": "exact", "query": "John 3:16"}\n'
        '{"id": "X1", "category": "exact", "query": "John 3:17"}\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseFileError, match="duplicate case id 'X1'"):
        load_cases(path)


def test_load_cases_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "X1", "category": "exact", "query": "q", "expected": {"verse_ref": "JHN 3:16"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseFileError, match="verse_ref"):
        load_cases(path)
    path.write_text('{"id": "A1", "query": "q", "must_quote": []}\n', encoding="utf-8")
    with pytest.raises(CaseFileError, match="must_quote"):
        load_cases(path, AnswerCase)


def test_committed_retrieval_cases_meet_coverage_minimums() -> None:
    cases = load_retrieval_cases(REPO_ROOT / "eval" / "retrieval_cases.jsonl")
    core = [c for c in cases if not c.known_gap]
    counts = Counter(c.category for c in core)
    minimums = {
        "exact": 10, "range_completeness": 8, "footnote_crossref": 5, "outline": 5,
        "topical_explicit": 5, "topical_implicit_paraphrase": 10, "concordance": 5,
        "mixed_intent": 6, "book_scoped_semantic": 6, "unanswerable": 5,
    }  # fmt: skip
    for category, minimum in minimums.items():
        assert counts[category] >= minimum, category
    assert len(core) >= 60
    assert {f"Q{n:02d}" for n in range(1, 16)} <= {c.id for c in cases}
    q01 = next(c for c in cases if c.id == "Q01")
    assert q01.expected.verse_refs_all == ["GEN 1:1", "GEN 1:2", "GEN 1:3"]
    readme = [c for c in cases if c.category == "mixed_intent" and "666" in c.query]
    assert readme and "Revelation 13:18" in readme[0].query
    indonesian = [c for c in cases if "id-lang" in c.tags]
    assert len(indonesian) == 3 and all(c.known_gap for c in indonesian)
    assert all(c.unanswerable for c in cases if c.category == "unanswerable")


def test_committed_answer_cases_meet_coverage_minimums() -> None:
    cases = load_answer_cases(REPO_ROOT / "eval" / "answer_cases.jsonl")
    assert len(cases) >= 25
    assert sum(c.should_refuse for c in cases) >= 5
    indonesian = [c for c in cases if "id-lang" in c.tags]
    assert len(indonesian) == 3 and all(c.known_gap for c in indonesian)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_context_coverage_counts_required_verses() -> None:
    case = RetrievalCase(
        id="R",
        category="range_completeness",
        query="Genesis 1:1-5",
        expected={"source_type": "scripture", "verse_refs_all": [f"GEN 1:{v}" for v in range(1, 6)]},
    )
    results = [_scripture("a", ["GEN 1:1", "GEN 1:2"]), _scripture("b", ["GEN 1:3", "GEN 2:1"])]
    assert context_coverage(case, results) == pytest.approx(0.6)
    assert hit_rank(case, results) is None  # incomplete range is not a hit
    results.append(_scripture("c", ["GEN 1:4", "GEN 1:5"]))
    assert context_coverage(case, results) == 1.0
    assert hit_rank(case, results) == 3
    no_range = RetrievalCase(id="N", category="exact", query="q", expected={"verse_refs_any": ["GEN 1:1"]})
    assert context_coverage(no_range, results) is None
    assert hit_rank(no_range, results) == 1


def test_rank_metrics() -> None:
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)
    assert recall_at_k([], 5) == 0.0


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(8.8)
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 10.0
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile(values, 101)


def test_citation_precision() -> None:
    a = CitationReport(valid=["x", "y"], partial=["z"], unsupported=["w"])
    b = CitationReport(valid=["v"], unparsed=["see above"])
    assert citation_precision(a) == 0.5
    assert citation_precision(a, b) == 0.6  # micro-average, unparsed excluded
    assert citation_precision(CitationReport(unparsed=["n"]), None) is None


def _response(answer: str, mode: AnswerMode, report: CitationReport | None = None, results=None):
    return RAGResponse(
        query="q",
        answer=answer,
        answer_mode=mode,
        citation_report=report,
        retrieved_chunks=results or [],
    )


def test_is_refusal() -> None:
    assert is_refusal(_response("", AnswerMode.INSUFFICIENT_CONTEXT))
    assert is_refusal(_response(f"Sorry. {INSUFFICIENT_CONTEXT_ANSWER.upper()}", AnswerMode.LLM))
    assert not is_refusal(_response("God created [Genesis 1:1].", AnswerMode.LLM))


def test_must_cite_recall_handles_verses_labels_and_numbers() -> None:
    case = AnswerCase(
        id="A", query="q", must_cite=["REV 13:18", "BSB Topical Index: 666", "GEN 1:2"]
    )
    report = CitationReport(valid=["Revelation 13:16-18", "Topical Index: 666"], unsupported=["Genesis 1:2"])
    assert must_cite_recall(case, _response("", AnswerMode.LLM, report)) == pytest.approx(2 / 3)
    numeric = CitationReport(valid=["1"])
    results = [_scripture("gen", ["GEN 1:1", "GEN 1:2"])]
    assert must_cite_recall(
        AnswerCase(id="B", query="q", must_cite=["GEN 1:2"]),
        _response("", AnswerMode.LLM, numeric, results),
    ) == 1.0
    assert must_cite_recall(AnswerCase(id="C", query="q"), _response("", AnswerMode.LLM)) is None


def test_false_exact_match_for_unanswerable_plans() -> None:
    topical = RetrievalResult(
        chunk=ChunkDocument(
            chunk_id="topical:666:0",
            text="t",
            metadata=ChunkMetadata(chunk_id="topical:666:0", source_type=SourceType.TOPICAL_INDEX, topic="666"),
        )
    )
    assert has_false_exact_match(route_query("Topic: 666"), [topical])
    assert not has_false_exact_match(route_query("Topic: cryptocurrency"), [topical])


# ---------------------------------------------------------------------------
# Ground-truth verification (hand-built corpus index, no raw files needed)
# ---------------------------------------------------------------------------


def test_verse_key_exists() -> None:
    assert verse_key_exists("JHN 3:16")
    assert verse_key_exists("jud 1:25")
    assert not verse_key_exists("GEN 51:1")
    assert not verse_key_exists("REV 99:99")
    assert not verse_key_exists("John 3:16")


def test_verify_cases_reports_unsatisfiable_expectations() -> None:
    corpus = CorpusIndex(topics={"666", "anxiety"}, concordance_words={"grace", "internet"})
    corpus.scripture_by_verse["JHN 3:16"].append("[John 3:16] For God so loved the world")
    retrieval = [
        RetrievalCase(id="OK", category="exact", query="John 3:16",
                      expected={"verse_refs_any": ["JHN 3:16"], "text_contains": "so loved"}),
        RetrievalCase(id="BADVERSE", category="exact", query="q", expected={"verse_refs_any": ["GEN 51:1"]}),
        RetrievalCase(id="BADTEXT", category="exact", query="q",
                      expected={"verse_refs_any": ["JHN 3:16"], "text_contains": "invented words"}),
        RetrievalCase(id="BADTOPIC", category="topical_explicit", query="q", expected={"topic": "grace"}),
        RetrievalCase(id="BADWORD", category="concordance", query="q", expected={"concordance_word": "xyz"}),
        RetrievalCase(id="ANSWERABLE", category="unanswerable", query="Topic: anxiety", unanswerable=True),
        RetrievalCase(id="WORDEXISTS", category="unanswerable", unanswerable=True,
                      query="How many times does the word 'internet' appear?"),
        RetrievalCase(id="UNANSWERABLE", category="unanswerable", query="Genesis 51:1", unanswerable=True),
    ]  # fmt: skip
    answers = [
        AnswerCase(id="AOK", query="q", must_cite=["JHN 3:16", "BSB Topical Index: 666"]),
        AnswerCase(id="ABAD", query="q", must_cite=["BSB Topical Index: Grace", "XYZ"]),
    ]
    issues = verify_cases(retrieval, answers, corpus)
    failing = {issue.case_id for issue in issues}
    assert failing == {"BADVERSE", "BADTEXT", "BADTOPIC", "BADWORD", "ANSWERABLE", "WORDEXISTS", "ABAD"}


# ---------------------------------------------------------------------------
# CLIs
# ---------------------------------------------------------------------------


def test_evaluate_answers_fake_llm_end_to_end(tmp_path: Path) -> None:
    module = _load_script("evaluate_answers")
    cases = tmp_path / "answers.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                {"id": "S1", "query": "What does John 3:16 say?", "must_cite": ["JHN 3:16"]},
                {"id": "S2", "query": "Topic: 666", "must_cite": ["BSB Topical Index: 666"]},
                {"id": "S3", "query": "What is the capital of France?", "should_refuse": True},
                {"id": "S4", "query": "Apa isi Yohanes 3:16?", "tags": ["id-lang", "known-gap"]},
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        module.app,
        ["--fake-llm", "--sample-ram", "--cases", str(cases), "--report-json", str(report_path)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["fake_llm"] is True
    (model,) = report["models"]
    summary = model["summary"]
    assert summary["cases"] == 3 and summary["errors"] == 0
    assert summary["citation_precision"] == 1.0
    assert summary["must_cite_recall"] == 1.0
    assert summary["should_refuse_recall"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["peak_ram_gb"] > 0
    assert model["known_gap_summary"]["cases"] == 1
    assert [c["id"] for c in model["cases"]] == ["S1", "S2", "S3", "S4"]


def test_parse_models_splits_on_first_colon() -> None:
    module = _load_script("evaluate_answers")
    assert module.parse_models("ollama:qwen2.5:1.5b, openai:gpt-4o-mini,anthropic") == [
        ("ollama", "qwen2.5:1.5b"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-haiku-5-5"),
    ]


class _StubRetriever:
    """Returns the same scripture chunk for every query."""

    def retrieve(self, query: str, top_k: int | None = None):
        return route_query(query), [_scripture("scripture:JHN:1:1-1:0", ["JHN 1:1"], "In the beginning was the Word")]


def test_evaluate_retrieval_legacy_report_structure(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("evaluate_retrieval")
    monkeypatch.setattr(module, "build_in_memory_benchmark_retriever", lambda **_: _StubRetriever())
    report_path = tmp_path / "legacy.json"
    result = runner.invoke(
        module.app, ["--legacy", "--min-recall-at-5", "0", "--report-json", str(report_path)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "legacy"
    assert {"recall_at_1", "recall_at_5", "mrr", "results"} <= report.keys()
    assert len(report["results"]) == 15
    q02 = next(r for r in report["results"] if r["case_id"] == "Q02")
    assert q02["hit_rank"] == 1 and q02["hit_at_1"] and q02["hit_at_5"]
    assert report["recall_at_5"] == pytest.approx(1 / 15)

    legacy = module.evaluate_benchmark(_StubRetriever(), cases=module.BENCHMARK_QUERIES, top_k=5)
    assert isinstance(legacy, module.BenchmarkReport) and len(legacy.results) == 15

    gated = runner.invoke(module.app, ["--legacy", "--report-json", str(report_path)])
    assert gated.exit_code == 1  # default 0.90 Recall@5 gate still applies


def test_evaluate_retrieval_case_outcomes(tmp_path: Path) -> None:
    module = _load_script("evaluate_retrieval")
    cases = [
        RetrievalCase(id="HIT", category="exact", query="John 1:1",
                      expected={"source_type": "scripture", "verse_refs_any": ["JHN 1:1"]}),
        RetrievalCase(id="MISS", category="exact", query="John 3:16", expected={"verse_refs_any": ["JHN 3:16"]}),
        RetrievalCase(id="U", category="unanswerable", query="Genesis 51:1", unanswerable=True),
        RetrievalCase(id="G", category="id_lang", query="Yohanes 1:1", tags=["known-gap"],
                      expected={"verse_refs_any": ["JHN 1:1"]}),
    ]  # fmt: skip
    outcomes = module.evaluate_cases(_StubRetriever(), cases, top_k=5)
    assert [o.passed for o in outcomes] == [True, False, True, True]
    summary = module.render_case_report(outcomes)
    assert summary["overall"]["recall_at_5"] == 0.5
    assert summary["categories"]["unanswerable"]["pass_rate"] == 1.0
    assert summary["known_gap"]["KNOWN-GAP TOTAL"]["n"] == 1
