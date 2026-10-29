"""Acceptance tests for advanced Streamlit UI features and feedback loop (TASK-12)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from beresbirag.config import get_settings
from beresbirag.llm.rag_chain import PreparedQuery
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    CitationItem,
    CitationReport,
    QueryIntent,
    RAGResponse,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval.query_router import QueryPlan
from beresbirag.ui.services import set_service_factory

APP_PATH = str(
    Path(__file__).resolve().parent.parent / "src" / "beresbirag" / "ui" / "streamlit_app.py"
)


def _create_app_test() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=25)


def _sample_results() -> list[RetrievalResult]:
    chunk = ChunkDocument(
        chunk_id="scripture:JHN:3:16:0",
        text="[John 3:16] For God so loved the world...",
        metadata=ChunkMetadata(
            chunk_id="scripture:JHN:3:16:0",
            source_type=SourceType.SCRIPTURE,
            canonical_ref="John 3:16",
            book_code="JHN",
            chapter=3,
            verse_start=16,
            verse_end=16,
            verse_refs=["JHN 3:16"],
        ),
    )
    return [RetrievalResult(chunk=chunk, score=0.95, rerank_score=0.98, rank=1)]


class FakeAdvancedPipeline:
    def __init__(self, unsupported: list[str] | None = None) -> None:
        self.unsupported = unsupported or ["Romans 8:28"]

    def prepare(self, query: str, top_k: int | None = None) -> PreparedQuery:
        return PreparedQuery(
            query=query,
            plan=QueryPlan(original_query=query, intent=QueryIntent.EXACT_PASSAGE),
            results=_sample_results(),
            context="[John 3:16] For God so loved the world...",
            fast_answer=None,
            answer_mode=AnswerMode.LLM,
            retrieval_ms=4.2,
        )

    def stream(self, prepared: PreparedQuery) -> Iterator[str]:
        yield "God loved the world [John 3:16] [Romans 8:28]."

    def finalize(self, prepared: PreparedQuery, answer: str) -> RAGResponse:
        report = CitationReport(
            valid=["John 3:16"],
            partial=[],
            unsupported=list(self.unsupported),
            unparsed=[],
            used_chunk_ids=["scripture:JHN:3:16:0"],
            has_citations=True,
        )
        return RAGResponse(
            query=prepared.query,
            answer=answer,
            intent=QueryIntent.EXACT_PASSAGE,
            provider="openai",
            model="gpt-4o-mini",
            answer_mode=AnswerMode.LLM,
            citations=[
                CitationItem(
                    chunk_id="scripture:JHN:3:16:0",
                    source_type=SourceType.SCRIPTURE,
                    canonical_ref="John 3:16",
                    book_code="JHN",
                    score=0.98,
                    used=True,
                )
            ],
            citation_report=report,
            retrieved_chunks=prepared.results,
            timings={"retrieval_ms": 4.2, "generation_ms": 12.5, "total_ms": 16.7},
        )


class FakeAdvancedFactory:
    def __init__(self, pipeline: FakeAdvancedPipeline | None = None) -> None:
        self._pipeline = pipeline or FakeAdvancedPipeline()

    def retriever(self) -> Any:
        return None

    def pipeline(self, provider: str, model: str, effort: str | None = None) -> Any:
        return self._pipeline


@pytest.fixture(autouse=True)
def reset_ui_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from beresbirag.diagnostics import CheckResult

    get_settings.cache_clear()
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    monkeypatch.setattr(
        "beresbirag.diagnostics.run_diagnostics",
        lambda settings: [CheckResult("Qdrant", "pass", "ok")],
    )
    yield
    set_service_factory(None)
    get_settings.cache_clear()


def test_history_holds_two_turns_and_renders_unsupported_chips_and_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))
    set_service_factory(FakeAdvancedFactory())

    app = _create_app_test()
    app.run()

    # Turn 1
    app.radio[0].set_value("Answer with LLM")
    app.text_area[0].input("John 3:16")
    app.button[4].click()
    app.run()

    # Turn 2
    app.text_area[0].input("Explain John 3:16")
    app.button[4].click()
    app.run()

    assert not app.exception
    assert len(app.session_state["history"]) == 2
    assert app.session_state["history"][0]["query"] == "John 3:16"
    assert app.session_state["history"][1]["query"] == "Explain John 3:16"

    # Verify unsupported citation chip rendered
    assert any("Unsupported: 1" in m.value and "Romans 8:28" in m.value for m in app.markdown)

    # Click 👎 feedback button and verify JSONL line written to tmp_path
    down_buttons = [b for b in app.button if b.label == "👎"]
    assert down_buttons
    down_buttons[0].click()
    app.run()

    feedback_file = tmp_path / "feedback.jsonl"
    assert feedback_file.is_file()
    lines = [
        line for line in feedback_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query"] == "Explain John 3:16"
    assert record["rating"] == "down"
    assert "timings" in record and "citation_report" in record


def test_password_gate_blocks_until_correct_password_entered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UI_PASSWORD", "top-secret-pw")
    set_service_factory(FakeAdvancedFactory())

    app = _create_app_test()
    app.run()

    # Main query text_area is not rendered while locked
    assert len(app.text_area) == 0
    assert len(app.text_input) == 1

    # Wrong password keeps gate locked and shows error
    app.text_input[0].input("wrong-password")
    app.run()
    assert len(app.error) >= 1
    assert len(app.text_area) == 0

    # Correct password unlocks full UI
    app.text_input[0].input("top-secret-pw")
    app.run()
    assert len(app.text_area) == 1


def test_non_loopback_address_without_password_shows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orig_get_option = st.get_option

    def fake_get_option(key: str) -> Any:
        if key == "server.address":
            return "0.0.0.0"
        return orig_get_option(key)

    monkeypatch.setattr(st, "get_option", fake_get_option)
    set_service_factory(FakeAdvancedFactory())

    app = _create_app_test()
    app.run()

    assert len(app.error) >= 1
    assert "non-loopback" in app.error[0].value.lower()
    assert len(app.text_area) == 0


def test_feedback_to_eval_converts_only_down_entries(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "feedback_to_eval.py"
    spec = importlib.util.spec_from_file_location("feedback_to_eval_test", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    feedback_file = tmp_path / "feedback.jsonl"
    entries = [
        {"ts": "2026-10-29T00:00:00Z", "query": "Good question", "rating": "up"},
        {"ts": "2026-10-29T00:01:00Z", "query": "Missed verse question", "rating": "down"},
        {"ts": "2026-10-29T00:02:00Z", "query": "Another bad answer", "rating": "👎"},
    ]
    feedback_file.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )

    draft_out = tmp_path / "answer_cases.draft.jsonl"
    cases = mod.convert_feedback_to_draft_cases(
        feedback_file=feedback_file,
        out_file=draft_out,
    )
    assert len(cases) == 2
    assert cases[0]["query"] == "Missed verse question"
    assert cases[0]["must_cite"] == []
    assert cases[0]["tags"] == ["from-feedback"]
    assert cases[1]["query"] == "Another bad answer"

    # Refuses to overwrite curated answer_cases.jsonl
    with pytest.raises(ValueError, match="answer_cases.jsonl"):
        mod.convert_feedback_to_draft_cases(
            feedback_file=feedback_file,
            out_file=tmp_path / "answer_cases.jsonl",
        )
