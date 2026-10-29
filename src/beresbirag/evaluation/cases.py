"""Typed, strictly validated retrieval and answer evaluation cases loaded from JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from beresbirag.models.schemas import RetrievalResult, SourceType

KNOWN_GAP_TAG = "known-gap"


class RetrievalExpectation(BaseModel):
    """Ground-truth criteria a retrieved chunk must satisfy (all set fields must hold)."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType | None = None
    verse_refs_any: list[str] = Field(default_factory=list)
    verse_refs_all: list[str] = Field(default_factory=list)
    book_code: str | None = None
    topic: str | None = None
    concordance_word: str | None = None
    text_contains: str | None = None

    def matches(self, result: RetrievalResult) -> bool:
        """True if a single chunk satisfies every per-chunk criterion.

        `verse_refs_all` is a context-level criterion (see metrics.context_coverage); a chunk
        holding any of those verses is still a per-chunk match when `verse_refs_any` is unset.
        """
        meta = result.chunk.metadata
        if self.source_type is not None and meta.source_type != self.source_type:
            return False
        if self.book_code is not None and (meta.book_code or "").upper() != self.book_code.upper():
            return False
        if self.topic is not None and (meta.topic or "").strip().lower() != self.topic.lower():
            return False
        if self.concordance_word is not None and (
            (meta.concordance_word or "").strip().lower() != self.concordance_word.lower()
        ):
            return False
        wanted = self.verse_refs_any or self.verse_refs_all
        if wanted and not {r.upper() for r in wanted} & {r.upper() for r in meta.verse_refs}:
            return False
        if self.text_contains is not None and (
            self.text_contains.lower() not in result.chunk.text.lower()
        ):
            return False
        return True


class RetrievalCase(BaseModel):
    """A single retrieval benchmark query with its ground truth."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expected: RetrievalExpectation = Field(default_factory=RetrievalExpectation)
    unanswerable: bool = False
    tags: list[str] = Field(default_factory=list)

    @property
    def known_gap(self) -> bool:
        return KNOWN_GAP_TAG in self.tags


class AnswerCase(BaseModel):
    """A single end-to-end answer case: what must be cited, or whether to refuse."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    must_cite: list[str] = Field(
        default_factory=list,
        description="Verse keys ('JHN 3:16') or labels ('BSB Topical Index: 666')",
    )
    should_refuse: bool = False
    tags: list[str] = Field(default_factory=list)

    @property
    def known_gap(self) -> bool:
        return KNOWN_GAP_TAG in self.tags


CaseT = TypeVar("CaseT", RetrievalCase, AnswerCase)


class CaseFileError(ValueError):
    """Raised when a JSONL case file is malformed, has unknown fields, or duplicate ids."""


def load_cases(path: str | Path, case_type: type[CaseT] = RetrievalCase) -> list[CaseT]:  # type: ignore[assignment]
    """Load and validate JSONL cases, rejecting unknown fields and duplicate ids."""
    cases: list[CaseT] = []
    seen: set[str] = set()
    errors: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            case = case_type.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {line_no}: {exc}")
            continue
        if case.id in seen:
            errors.append(f"line {line_no}: duplicate case id {case.id!r}")
            continue
        seen.add(case.id)
        cases.append(case)
    if errors:
        raise CaseFileError(f"Invalid case file {path}:\n" + "\n".join(errors))
    return cases


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    return load_cases(path, RetrievalCase)


def load_answer_cases(path: str | Path) -> list[AnswerCase]:
    return load_cases(path, AnswerCase)
