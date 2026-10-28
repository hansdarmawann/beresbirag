"""Dense and sparse embedding factories for Qdrant hybrid search."""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, SparseEmbeddings, SparseVector

from beresbirag.config import Settings, get_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class FastEmbedDenseEmbeddings(Embeddings):
    """LangChain Embeddings adapter around fastembed.TextEmbedding."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [vec.tolist() for vec in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vecs = list(model.query_embed(text))
        return vecs[0].tolist()


class DeterministicHashEmbeddings(Embeddings):
    """Deterministic, zero-network token-hash dense embedder for fast unit and CI testing."""

    def __init__(self, size: int = 384) -> None:
        self.size = size

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.size
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


class DeterministicBM25SparseEmbeddings(SparseEmbeddings):
    """Deterministic lexical sparse embedder (BM25-style TF weighting) for offline testing or fallback."""

    def _encode_sparse(self, text: str) -> SparseVector:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tf_by_index: dict[int, float] = {}
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % 1_000_003
            tf_by_index[idx] = tf_by_index.get(idx, 0.0) + 1.0
        if not tf_by_index:
            return SparseVector(indices=[0], values=[0.0])
        sorted_items = sorted(tf_by_index.items(), key=lambda item: item[0])
        indices = [idx for idx, _ in sorted_items]
        values = [round(1.0 + math.log(tf), 4) for _, tf in sorted_items]
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._encode_sparse(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._encode_sparse(text)


def create_dense_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the configured dense embedding model ('openai' or 'fastembed')."""
    cfg = settings or get_settings()
    provider = cfg.resolved_embedding_provider
    if provider == "openai":
        return OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,  # type: ignore[arg-type]
        )
    return FastEmbedDenseEmbeddings(model_name=cfg.fastembed_dense_model)


def create_sparse_embeddings(settings: Settings | None = None) -> SparseEmbeddings:
    """Create the configured FastEmbed BM25 sparse embedding model."""
    cfg = settings or get_settings()
    return FastEmbedSparse(model_name=cfg.fastembed_sparse_model)
