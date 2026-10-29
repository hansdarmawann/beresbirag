"""Answer-level evaluation across LLMs: citation precision, must-cite recall, refusals, latency.

Runs ./eval/answer_cases.jsonl against the configured (existing) Qdrant collection for each
`provider:model` in --models, or with --fake-llm against a deterministic in-memory corpus.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import psutil
import typer
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from rich import box
from rich.console import Console
from rich.table import Table

from beresbirag.config import Settings, get_settings
from beresbirag.evaluation import (
    AnswerCase,
    citation_precision,
    is_refusal,
    load_answer_cases,
    must_cite_recall,
    percentile,
)
from beresbirag.llm.factory import DEFAULT_MODELS, create_chat_model
from beresbirag.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from beresbirag.llm.rag_chain import BSBRAGPipeline, format_context_blocks
from beresbirag.models.schemas import (
    AnswerMode,
    ChunkDocument,
    ChunkMetadata,
    RAGResponse,
    SourceType,
)
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager

try:  # langchain-core >= 0.3.49
    from langchain_core.callbacks import get_usage_metadata_callback
except ImportError:  # pragma: no cover - older langchain-core
    get_usage_metadata_callback = None  # type: ignore[assignment]

app = typer.Typer(
    name="evaluate-answers",
    help="Evaluate grounded BSB answers (citations, refusals, latency, RAM) across LLMs.",
    add_completion=False,
)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "eval" / "answer_cases.jsonl"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
MAX_CONSECUTIVE_ERRORS = 3

JUDGE_PROMPT = (
    "You are checking whether an answer is fully supported by the SOURCES.\n"
    'Reply with JSON only: {{"supported": true|false, "unsupported_claims": ["..."]}}.\n\n'
    "SOURCES:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
)


# ---------------------------------------------------------------------------
# Model targets
# ---------------------------------------------------------------------------


def parse_models(spec: str) -> list[tuple[str, str]]:
    """Parse 'provider:model,...' splitting each item on the FIRST colon (ollama tags keep theirs)."""
    targets: list[tuple[str, str]] = []
    for item in (part.strip() for part in spec.split(",")):
        if not item:
            continue
        provider, _, model = item.partition(":")
        provider = provider.strip().lower()
        if provider not in DEFAULT_MODELS:
            raise typer.BadParameter(f"Unknown provider {provider!r} in {item!r}")
        targets.append((provider, model.strip() or DEFAULT_MODELS[provider]))
    if not targets:
        raise typer.BadParameter("--models is empty")
    return targets


class CitingFakeChatModel(SimpleChatModel):
    """Deterministic CI model: cites SOURCE [1] when it shares a content word with the question."""

    model_name: str = "fake-citing-llm"

    @property
    def _llm_type(self) -> str:
        return "fake-citing-chat"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\n".join(str(m.content) for m in messages)
        header = re.search(r"=== SOURCE \[1\] \| Reference: (.+?) \| Type:", prompt)
        sources, _, question = prompt.partition("=== SOURCE [1]")[2].rpartition("Question")
        words = set(re.findall(r"[a-z0-9]{3,}", question.lower())) - {
            "what", "does", "say", "the", "answer", "about", "bible", "bsb",
        }  # fmt: skip
        if header is None or not any(word in sources.lower() for word in words):
            return INSUFFICIENT_CONTEXT_ANSWER
        return f"According to the Berean Standard Bible [{header.group(1)}]."


def _synthetic_chunks() -> list[ChunkDocument]:
    def chunk(chunk_id: str, text: str, **meta: Any) -> ChunkDocument:
        return ChunkDocument(
            chunk_id=chunk_id, text=text, metadata=ChunkMetadata(chunk_id=chunk_id, **meta)
        )

    return [
        chunk(
            "scripture:GEN:1:1-3:0",
            "[Book: Genesis | Passage: Genesis 1:1-3]\n"
            "[Genesis 1:1] In the beginning God created the heavens and the earth.\n"
            "[Genesis 1:2] Now the earth was formless and void.\n"
            "[Genesis 1:3] And God said, Let there be light, and there was light.",
            source_type=SourceType.SCRIPTURE, book_code="GEN", book_name="Genesis", chapter=1,
            verse_start=1, verse_end=3, canonical_ref="Genesis 1:1-3",
            verse_refs=["GEN 1:1", "GEN 1:2", "GEN 1:3"],
        ),
        chunk(
            "scripture:JHN:3:16-17:0",
            "[Book: John | Passage: John 3:16-17]\n"
            "[John 3:16] For God so loved the world that He gave His one and only Son.\n"
            "[John 3:17] For God did not send His Son into the world to condemn the world.",
            source_type=SourceType.SCRIPTURE, book_code="JHN", book_name="John", chapter=3,
            verse_start=16, verse_end=17, canonical_ref="John 3:16-17",
            verse_refs=["JHN 3:16", "JHN 3:17"],
        ),
        chunk(
            "scripture:REV:13:18-18:0",
            "[Book: Revelation | Passage: Revelation 13:18]\n"
            "[Revelation 13:18] Let the one who has insight calculate the number of the beast; "
            "that number is 666.",
            source_type=SourceType.SCRIPTURE, book_code="REV", book_name="Revelation",
            chapter=13, verse_start=18, verse_end=18, canonical_ref="Revelation 13:18",
            verse_refs=["REV 13:18"],
        ),
        chunk(
            "topical:666:0",
            "[Source: BSB Topical Index | Topic: 666]\n- [Revelation 13:18] the number is 666",
            source_type=SourceType.TOPICAL_INDEX, canonical_ref="Topical Index: 666",
            topic="666", topic_display="666", verse_refs=["REV 13:18"],
        ),
        chunk(
            "concordance:grace:0",
            '[Source: BSB Concordance | Headword: "grace" | Total Occurrences: 136]',
            source_type=SourceType.CONCORDANCE, canonical_ref="Concordance: grace (136 Occurrences)",
            concordance_word="grace", concordance_occurrences=136,
        ),
        chunk(
            "outline:ROM:overview",
            "[Source: BSB Book Outline | Book: Romans (ROM)]\n2. Justification: Faith in the Savior",
            source_type=SourceType.BOOK_OUTLINE, book_code="ROM", book_name="Romans",
            canonical_ref="Romans Outline (1:1-16:27)",
        ),
    ]  # fmt: skip


def build_synthetic_retriever(settings: Settings) -> BSBHybridRetriever:
    """In-memory Qdrant over a tiny synthetic BSB corpus with deterministic embeddings."""
    manager = QdrantManager(
        settings=settings,
        dense_embeddings=DeterministicHashEmbeddings(size=384),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=384,
    )
    manager.ensure_collection(recreate=True)
    manager.upsert_chunks(_synthetic_chunks())
    return BSBHybridRetriever(
        qdrant_manager=manager,
        reranker=PassageReranker(settings=settings, enabled=False),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class RamSampler:
    """Background thread recording peak system memory in use (psutil) every `interval` s."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, psutil.virtual_memory().used)
            self._stop.wait(self.interval)

    def __enter__(self) -> RamSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, psutil.virtual_memory().used)


@contextmanager
def _usage_tracker() -> Iterator[dict[str, int]]:
    """Yield a dict filled with summed input/output tokens when LangChain reports usage."""
    totals = {"input_tokens": 0, "output_tokens": 0}
    if get_usage_metadata_callback is None:
        yield totals
        return
    with get_usage_metadata_callback() as callback:
        yield totals
        for usage in callback.usage_metadata.values():
            totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)


def judge_answer(judge: BaseChatModel, response: RAGResponse) -> dict[str, Any]:
    """LLM-as-judge faithfulness check returning {supported, unsupported_claims}."""
    prompt = JUDGE_PROMPT.format(
        context=format_context_blocks(response.retrieved_chunks),
        question=response.query,
        answer=response.answer,
    )
    raw = judge.invoke(
        [SystemMessage(content="You are a strict fact-checker."), HumanMessage(content=prompt)]
    )
    text = str(getattr(raw, "content", raw))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    return {
        "supported": bool(parsed.get("supported")) if "supported" in parsed else None,
        "unsupported_claims": list(parsed.get("unsupported_claims") or []),
    }


@dataclass
class CaseRun:
    case: AnswerCase
    response: RAGResponse | None = None
    error: str | None = None
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    judge: dict[str, Any] | None = None

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.generation_s


@dataclass
class ModelRun:
    provider: str
    model: str
    runs: list[CaseRun] = field(default_factory=list)
    peak_ram_bytes: int | None = None
    error: str | None = None


def run_case(pipeline: BSBRAGPipeline, case: AnswerCase, judge: BaseChatModel | None) -> CaseRun:
    run = CaseRun(case=case)
    try:
        start = perf_counter()
        prepared = pipeline.prepare(case.query)
        run.retrieval_s = perf_counter() - start
        with _usage_tracker() as usage:
            start = perf_counter()
            answer = "".join(pipeline.stream(prepared))
            run.generation_s = perf_counter() - start
        run.input_tokens, run.output_tokens = usage["input_tokens"], usage["output_tokens"]
        run.response = pipeline.finalize(prepared, answer)
        if judge is not None and run.response.answer_mode == AnswerMode.LLM:
            run.judge = judge_answer(judge, run.response)
    except Exception as exc:  # recorded per case so one failure does not abort the sweep
        run.error = f"{type(exc).__name__}: {exc}"
    return run


def run_model(
    provider: str,
    model: str,
    cases: Sequence[AnswerCase],
    retriever: BSBHybridRetriever,
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    judge: BaseChatModel | None = None,
    sample_ram: bool = False,
) -> ModelRun:
    result = ModelRun(provider=provider, model=model)
    try:
        pipeline = BSBRAGPipeline(
            retriever=retriever,
            llm=llm,
            provider=provider,
            model_name=model,
            settings=settings,  # type: ignore[arg-type]
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    sampler = RamSampler() if sample_ram else None
    consecutive_errors = 0
    with sampler if sampler is not None else nullcontext():
        for case in cases:
            run = run_case(pipeline, case, judge)
            result.runs.append(run)
            consecutive_errors = consecutive_errors + 1 if run.error else 0
            console.print(
                f"  [{provider}:{model}] {case.id} "
                + (
                    f"[red]ERROR {run.error}[/red]"
                    if run.error
                    else f"{run.response.answer_mode.value} ({run.total_s:.1f}s)"
                )  # type: ignore[union-attr]
            )
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                result.error = f"aborted after {consecutive_errors} consecutive errors: {run.error}"
                break
    if sampler is not None:
        result.peak_ram_bytes = sampler.peak_bytes
    return result


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------


def summarize_runs(runs: Sequence[CaseRun], peak_ram_bytes: int | None = None) -> dict[str, Any]:
    """Aggregate answer metrics over successfully answered cases."""
    ok = [r for r in runs if r.response is not None]
    responses = [r.response for r in ok]
    refuse = [r for r in ok if r.case.should_refuse]
    answer = [r for r in ok if not r.case.should_refuse]
    recalls = [
        value
        for r in answer
        if (value := must_cite_recall(r.case, r.response)) is not None  # type: ignore[arg-type]
    ]
    judged = [r.judge for r in ok if r.judge and r.judge.get("supported") is not None]
    return {
        "cases": len(runs),
        "errors": len(runs) - len(ok),
        "citation_precision": citation_precision(*(r.citation_report for r in responses)),  # type: ignore[union-attr]
        "must_cite_recall": sum(recalls) / len(recalls) if recalls else None,
        "should_refuse_recall": (
            sum(is_refusal(r.response) for r in refuse) / len(refuse) if refuse else None  # type: ignore[arg-type]
        ),
        "false_refusal_rate": (
            sum(is_refusal(r.response) for r in answer) / len(answer) if answer else None  # type: ignore[arg-type]
        ),
        "answer_modes": dict(Counter(r.answer_mode.value for r in responses)),  # type: ignore[union-attr]
        "retrieval_p50_s": percentile([r.retrieval_s for r in ok], 50),
        "retrieval_p95_s": percentile([r.retrieval_s for r in ok], 95),
        "generation_p50_s": percentile([r.generation_s for r in ok], 50),
        "generation_p95_s": percentile([r.generation_s for r in ok], 95),
        "total_p95_s": percentile([r.total_s for r in ok], 95),
        "tokens_in": sum(r.input_tokens for r in ok),
        "tokens_out": sum(r.output_tokens for r in ok),
        "peak_ram_gb": round(peak_ram_bytes / 1e9, 3) if peak_ram_bytes else None,
        "judge_supported_rate": (
            sum(bool(j["supported"]) for j in judged) / len(judged) if judged else None
        ),
    }


def _case_detail(run: CaseRun) -> dict[str, Any]:
    response = run.response
    return {
        "id": run.case.id,
        "query": run.case.query,
        "tags": run.case.tags,
        "should_refuse": run.case.should_refuse,
        "error": run.error,
        "answer": response.answer if response else None,
        "answer_mode": response.answer_mode.value if response else None,
        "refused": is_refusal(response) if response else None,
        "must_cite_recall": must_cite_recall(run.case, response) if response else None,
        "citation_report": (
            response.citation_report.model_dump() if response and response.citation_report else None
        ),
        "retrieval_s": round(run.retrieval_s, 3),
        "generation_s": round(run.generation_s, 3),
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "judge": run.judge,
    }


def _fmt(value: Any, kind: str = "pct") -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "s":
        return f"{value:.2f}"
    return str(value)


def render_summary(title: str, rows: dict[str, dict[str, Any]]) -> None:
    table = Table(title=title, box=box.ASCII)
    columns = (
        ("Model", None), ("Cite prec", "pct"), ("Must-cite", "pct"), ("Refuse rec", "pct"),
        ("False ref", "pct"), ("Modes", "raw"), ("Ret p50/p95 s", None), ("Gen p50/p95 s", None),
        ("Tok in/out", None), ("Peak RAM GB", "raw"), ("Judge", "pct"), ("Errors", "raw"),
    )  # fmt: skip
    for name, _ in columns:
        table.add_column(name)
    for label, s in rows.items():
        table.add_row(
            label,
            _fmt(s["citation_precision"]),
            _fmt(s["must_cite_recall"]),
            _fmt(s["should_refuse_recall"]),
            _fmt(s["false_refusal_rate"]),
            ", ".join(f"{k}={v}" for k, v in s["answer_modes"].items()) or "-",
            f"{_fmt(s['retrieval_p50_s'], 's')}/{_fmt(s['retrieval_p95_s'], 's')}",
            f"{_fmt(s['generation_p50_s'], 's')}/{_fmt(s['generation_p95_s'], 's')}",
            f"{s['tokens_in']}/{s['tokens_out']}",
            _fmt(s["peak_ram_gb"], "raw"),
            _fmt(s["judge_supported_rate"]),
            str(s["errors"]),
        )
    console.print(table)


@app.command()
def main(
    cases: Annotated[Path, typer.Option("--cases", help="JSONL answer cases.")] = DEFAULT_CASES,
    models: str | None = typer.Option(
        None,
        "--models",
        help="Comma list of provider:model (split on the first colon), e.g. "
        "ollama:qwen2.5:1.5b,openai:gpt-4o-mini. Default: the configured provider/model.",
    ),
    fake_llm: bool = typer.Option(
        False, "--fake-llm", help="Deterministic smoke run: synthetic in-memory corpus + fake LLM."
    ),
    sample_ram: bool = typer.Option(
        False, "--sample-ram", help="Sample peak system RAM every 250 ms via psutil."
    ),
    judge: str | None = typer.Option(
        None, "--judge", help="Optional provider:model LLM-as-judge faithfulness check."
    ),
    citation_policy: str | None = typer.Option(
        None, "--citation-policy", help="annotate|strict (default: CITATION_POLICY)."
    ),
    top_k: int | None = typer.Option(None, "--top-k", help="Override RETRIEVAL_TOP_K."),
    report_json: Annotated[
        Path | None,
        typer.Option(
            "--report-json", help="JSON report path (default ./eval/reports/answers-<ts>.json)."
        ),
    ] = None,
) -> None:
    """Run the answer cases for each model and report citation, refusal, latency and RAM metrics."""
    suite = load_answer_cases(cases)
    update: dict[str, Any] = {}
    if citation_policy is not None:
        if citation_policy not in {"annotate", "strict"}:
            raise typer.BadParameter("--citation-policy must be 'annotate' or 'strict'")
        update["citation_policy"] = citation_policy
    if top_k is not None:
        update["retrieval_top_k"] = top_k

    if fake_llm:
        update.update(
            qdrant_mode="memory", qdrant_collection_name="bsb_eval_answers_fake",
            offline_mode=False, reranker_enabled=False, fast_path_enabled=False,
        )  # fmt: skip
        settings = get_settings().model_copy(update=update)
        retriever = build_synthetic_retriever(settings)
        targets = [("openai", "fake-citing-llm")]
    else:
        settings = get_settings().model_copy(update=update)
        retriever = BSBHybridRetriever(settings=settings)
        targets = parse_models(models or f"{settings.llm_provider}:{settings.llm_model or ''}")

    judge_model: BaseChatModel | None = None
    if judge:
        judge_provider, judge_name = parse_models(judge)[0]
        judge_model = create_chat_model(
            provider=judge_provider,
            model_name=judge_name,
            settings=settings,  # type: ignore[arg-type]
        )

    model_runs: list[ModelRun] = []
    for provider, model in targets:
        console.print(f"[bold]Evaluating {provider}:{model} on {len(suite)} cases[/bold]")
        model_runs.append(
            run_model(
                provider,
                model,
                suite,
                retriever,
                settings,
                llm=CitingFakeChatModel() if fake_llm else None,
                judge=judge_model,
                sample_ram=sample_ram,
            )  # fmt: skip
        )

    core_rows: dict[str, dict[str, Any]] = {}
    gap_rows: dict[str, dict[str, Any]] = {}
    for mr in model_runs:
        label = f"{mr.provider}:{mr.model}"
        core_rows[label] = summarize_runs(
            [r for r in mr.runs if not r.case.known_gap], mr.peak_ram_bytes
        )
        gaps = [r for r in mr.runs if r.case.known_gap]
        if gaps:
            gap_rows[label] = summarize_runs(gaps)
        if mr.error:
            console.print(f"[red]{label}: {mr.error}[/red]")
    render_summary("BSB Answer Evaluation", core_rows)
    if gap_rows:
        render_summary("Known gaps (id-lang etc., excluded from model choice)", gap_rows)

    target = report_json or REPORTS_DIR / f"answers-{datetime.now():%Y%m%d-%H%M%S}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases_file": str(cases),
        "fake_llm": fake_llm,
        "citation_policy": settings.citation_policy,
        "judge": judge,
        "models": [
            {
                "provider": mr.provider,
                "model": mr.model,
                "error": mr.error,
                "summary": core_rows[f"{mr.provider}:{mr.model}"],
                "known_gap_summary": gap_rows.get(f"{mr.provider}:{mr.model}"),
                "cases": [_case_detail(r) for r in mr.runs],
            }
            for mr in model_runs
        ],
    }
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"Report written to {target}")


if __name__ == "__main__":
    app()
