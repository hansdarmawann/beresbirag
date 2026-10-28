"""Application configuration and environment settings for beresbirag."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root() -> Path:
    """Locate the current workspace root (or main repo root when inside .worktrees/<name>)."""
    here = Path(__file__).resolve()
    # src/beresbirag/config.py -> parents[2] is workspace/worktree root
    wt_or_repo = here.parents[2]
    return wt_or_repo


def resolve_data_file(path: str | Path) -> Path:
    """Resolve a raw BSB data file path, falling back to the main repo root if inside a Git worktree."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()

    wt_root = find_repo_root()
    wt_candidate = wt_root / candidate.name
    if wt_candidate.is_file():
        return wt_candidate.resolve()

    # Check if wt_root is inside <main_repo>/.worktrees/<wt_name>
    if wt_root.parent.name in {".worktrees", "worktrees"}:
        main_repo_candidate = wt_root.parent.parent / candidate.name
        if main_repo_candidate.is_file():
            return main_repo_candidate.resolve()

    return candidate


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM Provider & Minimal-Effort Settings
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="minimal", alias="LLM_EFFORT"
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Dense & Sparse Embedding Settings
    embedding_provider: Literal["auto", "openai", "fastembed"] = Field(
        default="auto", alias="EMBEDDING_PROVIDER"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    fastembed_dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="FASTEMBED_DENSE_MODEL"
    )
    fastembed_sparse_model: str = Field(
        default="Qdrant/bm25", alias="FASTEMBED_SPARSE_MODEL"
    )

    # Qdrant Vector Database Settings
    qdrant_mode: Literal["memory", "local", "remote"] = Field(
        default="remote", alias="QDRANT_MODE"
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_local_path: str = Field(default="./qdrant_storage", alias="QDRANT_LOCAL_PATH")
    qdrant_collection_name: str = Field(default="bsb_rag", alias="QDRANT_COLLECTION_NAME")

    # Reranker & Retrieval Settings
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2", alias="RERANKER_MODEL"
    )
    retrieval_initial_k: int = Field(default=25, ge=1, le=200, alias="RETRIEVAL_INITIAL_K")
    retrieval_top_k: int = Field(default=6, ge=1, le=50, alias="RETRIEVAL_TOP_K")

    # Raw BSB Source File Paths
    bsb_usj_zip_path: str = Field(default="./bsb_usj.zip", alias="BSB_USJ_ZIP_PATH")
    bsb_outlines_pdf_path: str = Field(
        default="./bsb_book_outlines.pdf", alias="BSB_OUTLINES_PDF_PATH"
    )
    bsb_topical_xlsx_path: str = Field(
        default="./bsb_topical_index.xlsx", alias="BSB_TOPICAL_XLSX_PATH"
    )
    bsb_concordance_xlsx_path: str = Field(
        default="./bsb_concordance.xlsx", alias="BSB_CONCORDANCE_XLSX_PATH"
    )

    @property
    def resolved_embedding_provider(self) -> Literal["openai", "fastembed"]:
        """Resolve 'auto' to 'openai' when OPENAI_API_KEY is set, otherwise 'fastembed'."""
        if self.embedding_provider == "openai":
            return "openai"
        if self.embedding_provider == "fastembed":
            return "fastembed"
        if self.openai_api_key and self.openai_api_key.strip():
            return "openai"
        return "fastembed"

    @property
    def dense_vector_size(self) -> int:
        """Return dense embedding dimension for the active dense embedding model."""
        if self.resolved_embedding_provider == "openai":
            if "3-large" in self.openai_embedding_model:
                return 3072
            return 1536
        # Default FastEmbed BAAI/bge-small-en-v1.5 is 384-dim
        if "bge-base" in self.fastembed_dense_model:
            return 768
        if "bge-large" in self.fastembed_dense_model:
            return 1024
        return 384

    @property
    def usj_zip_file(self) -> Path:
        return resolve_data_file(self.bsb_usj_zip_path)

    @property
    def outlines_pdf_file(self) -> Path:
        return resolve_data_file(self.bsb_outlines_pdf_path)

    @property
    def topical_xlsx_file(self) -> Path:
        return resolve_data_file(self.bsb_topical_xlsx_path)

    @property
    def concordance_xlsx_file(self) -> Path:
        return resolve_data_file(self.bsb_concordance_xlsx_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached default Settings instance."""
    return Settings()
