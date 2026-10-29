"""Lightweight hardware and service checks; never download or load neural models."""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import psutil

from beresbirag.config import Settings
from beresbirag.llm.factory import resolve_llm_target
from beresbirag.vectorstore.qdrant_manager import QdrantManager


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str


def check_avx2() -> CheckResult:
    try:
        system = platform.system()
        if system == "Windows":
            supported = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        elif system == "Linux":
            supported = "avx2" in Path("/proc/cpuinfo").read_text().split()
        else:
            return CheckResult("AVX2", "warn", "unknown on this platform")
        return CheckResult(
            "AVX2",
            "pass" if supported else "warn",
            "available" if supported else "not available; inference may be much slower",
        )
    except (OSError, AttributeError) as exc:
        return CheckResult("AVX2", "warn", f"unknown: {exc}")


def check_ollama(settings: Settings, transport: httpx.BaseTransport | None = None) -> CheckResult:
    try:
        _, model = resolve_llm_target(None, None, settings)
        with httpx.Client(transport=transport, timeout=5.0, trust_env=False) as client:
            response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            names = {item.get("name") for item in response.json().get("models", [])}

        def normalize(name: str) -> str:
            return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"

        present = normalize(model) in {normalize(name) for name in names if isinstance(name, str)}
        return CheckResult(
            "Ollama",
            "pass" if present else "fail",
            f"model {model} {'present' if present else 'missing; run ollama pull ' + model}",
        )
    except Exception as exc:
        return CheckResult("Ollama", "fail", str(exc))


def check_model_cache(settings: Settings) -> list[CheckResult]:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from flashrank.Config import model_file_map

    checks = []
    for label, cls, model in (
        ("Dense cache", TextEmbedding, settings.fastembed_dense_model),
        ("Sparse cache", SparseTextEmbedding, settings.fastembed_sparse_model),
    ):
        try:
            description = cls._get_model_description(model)
            path = cls.download_model(description, settings.model_cache_dir, local_files_only=True)
            required = [description.model_file, *description.additional_files]
            if not all((path / filename).is_file() for filename in required if filename):
                raise FileNotFoundError(f"incomplete cache for {model}")
            checks.append(CheckResult(label, "pass", str(path)))
        except Exception as exc:
            checks.append(
                CheckResult(label, "fail", f"{model}: {exc}; run scripts/prefetch_models.py online")
            )
    directory = (
        Path(os.environ.get("FLASHRANK_CACHE_DIR") or settings.model_cache_dir)
        / settings.reranker_model
    )
    required = [
        model_file_map.get(settings.reranker_model, "missing-model"),
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    present = all((directory / filename).is_file() for filename in required)
    checks.append(
        CheckResult(
            "Reranker cache",
            "pass" if present else "fail",
            f"{directory}: {'present' if present else 'missing or incomplete'}",
        )
    )
    return checks


def run_diagnostics(settings: Settings) -> list[CheckResult]:
    memory = psutil.virtual_memory()
    gib = 1024**3
    checks = [
        CheckResult(
            "RAM",
            "warn" if memory.available < 1.5 * gib else "pass",
            f"total {memory.total / gib:.2f} GiB; available {memory.available / gib:.2f} GiB",
        ),
        CheckResult(
            "CPU cores",
            "pass",
            f"physical {psutil.cpu_count(logical=False)}; logical {psutil.cpu_count()}",
        ),
        check_avx2(),
    ]
    manager = None
    try:
        manager = QdrantManager(settings=settings)
        if not manager.collection_exists():
            raise ValueError(f"collection {settings.qdrant_collection_name} missing")
        manager.validate_collection()
        count = manager.count_points()
        if count <= 0:
            raise ValueError("collection is empty")
        checks.append(
            CheckResult(
                "Qdrant", "pass", f"{manager.collection_name}: {count} points; schema valid"
            )
        )
    except Exception as exc:
        checks.append(CheckResult("Qdrant", "fail", str(exc)))
    finally:
        if manager is not None:
            manager.client.close()
    checks.extend(check_model_cache(settings))
    if settings.llm_provider == "ollama":
        checks.append(check_ollama(settings))
    inconsistent = settings.offline_mode and settings.llm_provider != "ollama"
    checks.append(
        CheckResult(
            "Offline consistency",
            "fail" if inconsistent else "pass",
            "offline mode: use --provider ollama" if inconsistent else "configuration consistent",
        )
    )
    return checks
