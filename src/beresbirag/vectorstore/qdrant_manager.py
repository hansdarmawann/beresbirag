"""Qdrant collection, hybrid dense + sparse BM25 store, and payload filter manager."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient, models

from beresbirag.config import Settings, get_settings
from beresbirag.models.schemas import ChunkDocument, RetrievalResult, SourceType
from beresbirag.vectorstore.embeddings import (
    create_dense_embeddings,
    create_sparse_embeddings,
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

KEYWORD_INDEX_FIELDS: tuple[str, ...] = (
    "metadata.source_type",
    "metadata.book_code",
    "metadata.testament",
    "metadata.verse_refs",
    "metadata.topic",
    "metadata.concordance_word",
    "metadata.chunk_id",
)

INTEGER_INDEX_FIELDS: tuple[str, ...] = (
    "metadata.chapter",
    "metadata.canonical_order",
    "metadata.verse_start",
    "metadata.verse_end",
)

RECREATE_REMEDY = "Re-run: conda run -n beresbirag python ./scripts/ingest_all.py --recreate"


class CollectionConfigError(RuntimeError):
    """Raised when an existing Qdrant collection is incompatible with the active configuration."""


class QdrantManager:
    """Manages the Qdrant hybrid vector collection (`bsb_rag`) and search operations."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        dense_embeddings: Embeddings | None = None,
        sparse_embeddings: SparseEmbeddings | None = None,
        dense_vector_size: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.qdrant_collection_name
        self.client = client or self._create_client()
        self._dense_embeddings = dense_embeddings
        self._sparse_embeddings = sparse_embeddings
        self._dense_vector_size = dense_vector_size
        self._validated = False

    def _create_client(self) -> QdrantClient:
        mode = self.settings.qdrant_mode
        if mode == "memory":
            return QdrantClient(location=":memory:")
        if mode == "local":
            return QdrantClient(path=self.settings.qdrant_local_path)
        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
        )

    @property
    def dense_embeddings(self) -> Embeddings:
        if self._dense_embeddings is None:
            self._dense_embeddings = create_dense_embeddings(self.settings)
        return self._dense_embeddings

    @property
    def sparse_embeddings(self) -> SparseEmbeddings:
        if self._sparse_embeddings is None:
            self._sparse_embeddings = create_sparse_embeddings(self.settings)
        return self._sparse_embeddings

    @property
    def dense_vector_size(self) -> int:
        if self._dense_vector_size is not None:
            return self._dense_vector_size
        size_attr = getattr(self._dense_embeddings, "size", None)
        if isinstance(size_attr, int) and size_attr > 0:
            return size_attr
        return self.settings.dense_vector_size

    def collection_exists(self) -> bool:
        """Return True if the target Qdrant collection already exists."""
        return bool(self.client.collection_exists(self.collection_name))

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the hybrid Qdrant collection and payload indexes if needed."""
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection_name)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                        # Qdrant/bm25 emits TF-only weights; IDF must be applied server-side.
                        modifier=models.Modifier.IDF,
                    )
                },
            )
            self._ensure_payload_indexes()
            # Freshly created from the active configuration, so it is valid by construction.
            self._validated = True

    def validate_collection(self) -> None:
        """Raise CollectionConfigError if the existing collection mismatches the active config."""
        if not self.collection_exists():
            return
        params = self.client.get_collection(self.collection_name).config.params

        vectors = params.vectors
        dense_params = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
        if dense_params is None:
            raise CollectionConfigError(
                f"Collection '{self.collection_name}' has no '{DENSE_VECTOR_NAME}' dense vector. "
                f"{RECREATE_REMEDY}"
            )
        if dense_params.size != self.dense_vector_size:
            provider = self.settings.resolved_embedding_provider
            model = (
                self.settings.openai_embedding_model
                if provider == "openai"
                else self.settings.fastembed_dense_model
            )
            raise CollectionConfigError(
                f"Collection '{self.collection_name}' stores {dense_params.size}-dim dense vectors, "
                f"but the active embedding provider '{provider}' (model '{model}') produces "
                f"{self.dense_vector_size}-dim vectors. {RECREATE_REMEDY}"
            )

        sparse_params = (params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
        modifier = getattr(sparse_params, "modifier", None)
        if modifier != models.Modifier.IDF:
            raise CollectionConfigError(
                f"Collection '{self.collection_name}' sparse vector '{SPARSE_VECTOR_NAME}' has "
                f"modifier={modifier!s}, but BM25 requires modifier=idf. {RECREATE_REMEDY}"
            )

    def _ensure_validated(self) -> None:
        """Validate the collection once per manager instance (skipped while it does not exist)."""
        if self._validated or not self.collection_exists():
            return
        self.validate_collection()
        self._validated = True

    def _ensure_payload_indexes(self) -> None:
        """Create keyword and integer payload indexes for fast metadata filtering."""
        for field_name in KEYWORD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # In-memory Qdrant may ignore or no-op payload index creation
                pass

        for field_name in INTEGER_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            except Exception:
                pass

    def get_vector_store(self) -> QdrantVectorStore:
        """Return a LangChain QdrantVectorStore configured for Hybrid RRF retrieval."""
        self.ensure_collection(recreate=False)
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    def upsert_chunks(
        self,
        chunks: Iterable[ChunkDocument],
        batch_size: int = 64,
    ) -> int:
        """Convert ChunkDocuments to LangChain Documents and upsert in batches."""
        store = self.get_vector_store()
        total_upserted = 0
        batch_docs: list[Document] = []
        batch_ids: list[str] = []

        for chunk in chunks:
            doc = chunk.to_langchain_document()
            batch_docs.append(doc)
            batch_ids.append(chunk.point_id)
            if len(batch_docs) >= batch_size:
                store.add_documents(documents=batch_docs, ids=batch_ids)
                total_upserted += len(batch_docs)
                batch_docs.clear()
                batch_ids.clear()

        if batch_docs:
            store.add_documents(documents=batch_docs, ids=batch_ids)
            total_upserted += len(batch_docs)

        return total_upserted

    def count_points(self, qdrant_filter: models.Filter | None = None) -> int:
        """Return total number of points in the collection (optionally matching a filter)."""
        if not self.collection_exists():
            return 0
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        return int(res.count)

    @staticmethod
    def build_filter(
        *,
        source_types: list[SourceType | str] | None = None,
        book_codes: list[str] | None = None,
        testament: str | None = None,
        chapter: int | None = None,
        verse_refs: list[str] | None = None,
        topic: str | None = None,
        concordance_word: str | None = None,
    ) -> models.Filter | None:
        """Build a Qdrant payload Filter across indexed metadata fields."""
        must_conditions: list[models.Condition] = []

        if source_types:
            values = [
                st.value if isinstance(st, SourceType) else str(st)
                for st in source_types
            ]
            if len(values) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.source_type",
                        match=models.MatchValue(value=values[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.source_type",
                        match=models.MatchAny(any=values),
                    )
                )

        if book_codes:
            norm_codes = [c.strip().upper() for c in book_codes if c.strip()]
            if len(norm_codes) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.book_code",
                        match=models.MatchValue(value=norm_codes[0]),
                    )
                )
            elif norm_codes:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.book_code",
                        match=models.MatchAny(any=norm_codes),
                    )
                )

        if testament:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.testament",
                    match=models.MatchValue(value=testament.strip().upper()),
                )
            )

        if chapter is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.chapter",
                    match=models.MatchValue(value=int(chapter)),
                )
            )

        if verse_refs:
            norm_refs = [r.strip().upper() for r in verse_refs if r.strip()]
            if len(norm_refs) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.verse_refs",
                        match=models.MatchValue(value=norm_refs[0]),
                    )
                )
            elif norm_refs:
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.verse_refs",
                        match=models.MatchAny(any=norm_refs),
                    )
                )

        if topic:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.topic",
                    match=models.MatchValue(value=topic.strip().lower()),
                )
            )

        if concordance_word:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.concordance_word",
                    match=models.MatchValue(value=concordance_word.strip().lower()),
                )
            )

        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)

    def hybrid_search(
        self,
        query: str,
        k: int | None = None,
        qdrant_filter: models.Filter | None = None,
    ) -> list[RetrievalResult]:
        """Execute Qdrant Hybrid (Dense + Sparse BM25 RRF) search."""
        self._ensure_validated()
        top_k = k or self.settings.retrieval_initial_k
        store = self.get_vector_store()
        docs_and_scores = store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=qdrant_filter,
        )
        results: list[RetrievalResult] = []
        for idx, (doc, score) in enumerate(docs_and_scores, start=1):
            chunk = ChunkDocument.from_langchain_document(doc)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=idx,
                )
            )
        return results

    def scroll_by_filter(
        self,
        qdrant_filter: models.Filter,
        limit: int = 20,
    ) -> list[RetrievalResult]:
        """Deterministically fetch chunks matching an exact Qdrant payload filter."""
        self._ensure_validated()
        if not self.collection_exists():
            return []
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[RetrievalResult] = []
        for idx, record in enumerate(records, start=1):
            payload = dict(record.payload or {})
            page_content = str(payload.get("page_content") or "")
            metadata = dict(payload.get("metadata") or {})
            if not page_content:
                continue
            lc_doc = Document(
                page_content=page_content,
                metadata=metadata,
                id=str(record.id),
            )
            chunk = ChunkDocument.from_langchain_document(lc_doc)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=1.0,
                    rank=idx,
                )
            )
        return results
