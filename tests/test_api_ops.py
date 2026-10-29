"""Acceptance tests for operational API hardening (TASK-12)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import beresbirag.api as api_mod
from beresbirag.api import app
from beresbirag.config import Settings, get_settings
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.models.schemas import (
    ChunkDocument,
    ChunkMetadata,
    QueryIntent,
    RetrievalResult,
    SourceType,
)


def _sample_results() -> list[RetrievalResult]:
    chunk = ChunkDocument(
        chunk_id="scripture:GEN:1:1:0",
        text="[Genesis 1:1] In the beginning God created the heavens and the earth.",
        metadata=ChunkMetadata(
            chunk_id="scripture:GEN:1:1:0",
            source_type=SourceType.SCRIPTURE,
            book_code="GEN",
            book_name="Genesis",
            chapter=1,
            verse_start=1,
            verse_end=1,
            canonical_ref="Genesis 1:1",
            verse_refs=["GEN 1:1"],
        ),
    )
    return [RetrievalResult(chunk=chunk, score=0.95, rerank_score=0.99, rank=1)]


class FakePlan:
    def __init__(self, intent: QueryIntent = QueryIntent.EXACT_PASSAGE) -> None:
        self.intent = intent


class FakeQdrantManager:
    def __init__(
        self,
        *,
        exists: bool = True,
        points: int = 42,
        valid: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._exists = exists
        self._points = points
        self._valid = valid
        self._metadata = metadata
        self.client = None

    def collection_exists(self) -> bool:
        return self._exists

    def count_points(self) -> int:
        return self._points

    def validate_collection(self) -> None:
        if not self._valid:
            raise RuntimeError("Invalid collection")

    def read_index_metadata(self) -> dict[str, Any] | None:
        return self._metadata


class FakeRetriever:
    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        qdrant_manager: Any = None,
        error: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.results = results if results is not None else _sample_results()
        self.qdrant_manager = qdrant_manager or FakeQdrantManager()
        self.error = error
        self.delay_s = delay_s

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> tuple[FakePlan, list[RetrievalResult]]:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        k = top_k or 6
        return FakePlan(), self.results[:k]


@pytest.fixture(autouse=True)
def clean_overrides() -> Iterator[None]:
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_lifespan_builds_retriever_once_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counting_build_retriever(settings: Settings) -> FakeRetriever:
        nonlocal calls
        calls += 1
        return FakeRetriever()

    monkeypatch.setattr(api_mod, "build_retriever", counting_build_retriever)

    with TestClient(app) as client:
        for _ in range(3):
            resp = client.post("/retrieve", json={"query": "Genesis 1:1", "top_k": 1})
            assert resp.status_code == 200
            assert len(resp.json()) == 1

    assert calls == 1


def test_livez_health_and_readyz_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_manager = FakeQdrantManager(exists=False, points=0)
    monkeypatch.setattr(
        api_mod,
        "build_retriever",
        lambda settings: FakeRetriever(qdrant_manager=missing_manager),
    )

    with TestClient(app) as client:
        r_live = client.get("/livez")
        assert r_live.status_code == 200
        assert r_live.json() == {"status": "alive"}

        r_health = client.get("/health")
        assert r_health.status_code == 200

        r_ready_fail = client.get("/readyz")
        assert r_ready_fail.status_code == 503
        assert "collection_populated" in r_ready_fail.json()["failing_checks"]

    healthy_manager = FakeQdrantManager(
        exists=True,
        points=100,
        metadata={
            "dense_model": "BAAI/bge-small-en-v1.5",
            "sparse_model": "Qdrant/bm25",
        },
    )
    monkeypatch.setattr(
        api_mod,
        "build_retriever",
        lambda settings: FakeRetriever(qdrant_manager=healthy_manager),
    )

    with TestClient(app) as client:
        r_ready_ok = client.get("/readyz")
        assert r_ready_ok.status_code == 200
        assert r_ready_ok.json()["status"] == "ready"
        assert r_ready_ok.json()["failing_checks"] == []


def test_api_key_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Settings(api_keys="k1,k2")
    app.dependency_overrides[get_settings] = lambda: cfg
    monkeypatch.setattr(api_mod, "build_retriever", lambda s: FakeRetriever())

    with TestClient(app) as client:
        # /livez requires no API key
        assert client.get("/livez").status_code == 200

        # POST /retrieve without header -> 401
        r_unauth = client.post("/retrieve", json={"query": "Genesis 1:1"})
        assert r_unauth.status_code == 401

        # POST /retrieve with wrong key -> 401
        r_wrong = client.post(
            "/retrieve",
            json={"query": "Genesis 1:1"},
            headers={"X-API-Key": "wrong"},
        )
        assert r_wrong.status_code == 401

        # POST /retrieve with valid key -> 200
        r_ok = client.post(
            "/retrieve",
            json={"query": "Genesis 1:1"},
            headers={"X-API-Key": "k1"},
        )
        assert r_ok.status_code == 200


def test_rate_limiter_enforces_per_minute_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Settings(rate_limit_per_minute=2)
    app.dependency_overrides[get_settings] = lambda: cfg
    monkeypatch.setattr(api_mod, "build_retriever", lambda s: FakeRetriever())

    with TestClient(app) as client:
        assert client.post("/retrieve", json={"query": "Genesis 1:1"}).status_code == 200
        assert client.post("/retrieve", json={"query": "Genesis 1:1"}).status_code == 200
        r_limited = client.post("/retrieve", json={"query": "Genesis 1:1"})
        assert r_limited.status_code == 429
        assert "Retry-After" in r_limited.headers
        assert int(r_limited.headers["Retry-After"]) >= 1


def test_allowed_models_and_max_query_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Settings(
        allowed_models="openai:gpt-4o-mini",
        max_query_chars=25,
    )
    app.dependency_overrides[get_settings] = lambda: cfg
    monkeypatch.setattr(api_mod, "build_retriever", lambda s: FakeRetriever())
    monkeypatch.setattr(
        api_mod,
        "build_pipeline",
        lambda retriever, settings, **kw: BSBRAGPipeline(
            retriever=retriever,
            llm=FakeListChatModel(responses=["In the beginning [Genesis 1:1]."]),
            settings=settings,
        ),
    )

    with TestClient(app) as client:
        # Model outside ALLOWED_MODELS -> 422
        r_bad_model = client.post(
            "/query",
            json={
                "query": "Genesis 1:1",
                "provider": "openai",
                "model": "gpt-4.1-expensive",
            },
        )
        assert r_bad_model.status_code == 422

        # Query longer than MAX_QUERY_CHARS -> 422
        r_long_query = client.post(
            "/retrieve",
            json={"query": "x" * 26},
        )
        assert r_long_query.status_code == 422


def test_internal_error_redacts_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_retriever = FakeRetriever(error=RuntimeError(r"C:\secret\path\credentials.db"))
    monkeypatch.setattr(api_mod, "build_retriever", lambda s: failing_retriever)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/retrieve", json={"query": "Genesis 1:1"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal error"
        assert "error_id" in body and len(body["error_id"]) >= 32
        assert "secret" not in resp.text.lower()


def test_concurrency_semaphore_returns_503_when_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Settings(max_concurrent_requests=1, queue_timeout_s=0.1)
    app.dependency_overrides[get_settings] = lambda: cfg

    entered = threading.Event()
    release = threading.Event()

    class BlockingRetriever:
        qdrant_manager = FakeQdrantManager()

        def retrieve(
            self, query: str, top_k: int | None = None
        ) -> tuple[FakePlan, list[RetrievalResult]]:
            entered.set()
            release.wait(timeout=2.0)
            return FakePlan(), _sample_results()

    monkeypatch.setattr(api_mod, "build_retriever", lambda s: BlockingRetriever())

    with TestClient(app) as client:
        first_status: list[int] = []

        def _first_call() -> None:
            r = client.post("/retrieve", json={"query": "Genesis 1:1"})
            first_status.append(r.status_code)

        t = threading.Thread(target=_first_call)
        t.start()
        assert entered.wait(timeout=2.0)

        second_resp = client.post("/retrieve", json={"query": "Genesis 1:1"})
        assert second_resp.status_code == 503
        assert second_resp.json() == {"detail": "Server busy, retry later"}

        release.set()
        t.join(timeout=2.0)
        assert first_status == [200]


def test_query_stream_sse_events_and_timings_and_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_retriever = FakeRetriever()
    fake_llm = FakeListChatModel(responses=["In the beginning God created [Genesis 1:1]."])
    monkeypatch.setattr(api_mod, "build_retriever", lambda s: fake_retriever)
    monkeypatch.setattr(
        api_mod,
        "build_pipeline",
        lambda retriever, settings, **kw: BSBRAGPipeline(
            retriever=retriever,
            llm=fake_llm,
            settings=settings,
            provider=kw.get("provider"),
            model_name=kw.get("model_name"),
            effort=kw.get("effort"),
        ),
    )

    with TestClient(app) as client:
        # 1. Non-streaming /query timings and echoed X-Request-ID
        r_query = client.post(
            "/query",
            json={"query": "What does Genesis 1:1 say?", "top_k": 1},
            headers={"X-Request-ID": "req-test-123"},
        )
        assert r_query.status_code == 200
        assert r_query.headers.get("X-Request-ID") == "req-test-123"
        q_body = r_query.json()
        timings = q_body["timings"]
        assert set(timings.keys()) >= {"retrieval_ms", "generation_ms", "total_ms"}
        assert timings["retrieval_ms"] >= 0.0
        assert timings["generation_ms"] >= 0.0
        assert timings["total_ms"] >= 0.0

        # 2. SSE /query/stream emits meta -> >=1 token -> done in order
        with client.stream(
            "POST",
            "/query/stream",
            json={"query": "What does Genesis 1:1 say?", "top_k": 1},
            headers={"X-Request-ID": "req-stream-456"},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            assert stream_resp.headers.get("X-Request-ID") == "req-stream-456"
            raw_text = "".join(stream_resp.iter_text())

        events: list[tuple[str, Any]] = []
        for block in raw_text.strip().split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            ev_name = ""
            ev_data = ""
            for line in lines:
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    ev_data = line.split(":", 1)[1].strip()
            if ev_name:
                events.append((ev_name, json.loads(ev_data)))

        event_names = [e[0] for e in events]
        assert event_names[0] == "meta"
        assert event_names[-1] == "done"
        assert "token" in event_names[1:-1]

        meta_payload = events[0][1]
        assert meta_payload["intent"] == "exact_passage"
        assert meta_payload["retrieved"][0]["canonical_ref"] == "Genesis 1:1"

        done_payload = events[-1][1]
        assert "Genesis 1:1" in done_payload["answer"]
        assert done_payload["timings"]["total_ms"] >= 0.0
