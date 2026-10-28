"""Unit and integration tests for TASK-05: reference parser, query router, reranker, and hybrid retriever."""

from __future__ import annotations

from typing import Any

import pytest

from beresbirag.config import Settings
from beresbirag.models.schemas import (
    ChunkDocument,
    ChunkMetadata,
    QueryIntent,
    RetrievalResult,
    SourceType,
)
from beresbirag.retrieval import (
    EXACT_MATCH_BONUS,
    MAX_EXPANDED_REFS,
    BSBHybridRetriever,
    BSBLangChainRetriever,
    PassageReranker,
    extract_bible_references,
    route_query,
)
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager


def _sample_corpus_chunks() -> list[ChunkDocument]:
    """Create representative chunks across all 4 BSB corpora for integration testing."""
    return [
        ChunkDocument(
            chunk_id="scripture:GEN:1:1-5:0",
            text=(
                "[Book: Genesis | Passage: Genesis 1:1-5 | Testament: OT | "
                "Outline: 1. God and the World > a. The Creation | Heading: The Creation / The First Day]\n"
                "[Genesis 1:1] In the beginning God created the heavens and the earth.\n"
                "[Genesis 1:2] Now the earth was formless and void, and darkness was over the surface of the deep.\n"
                "[Genesis 1:3] And God said, 'Let there be light,' and there was light."
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:GEN:1:1-5:0",
                source_type=SourceType.SCRIPTURE,
                book_code="GEN",
                book_name="Genesis",
                testament="OT",
                canonical_order=1,
                chapter=1,
                verse_start=1,
                verse_end=5,
                canonical_ref="Genesis 1:1-5",
                verse_refs=["GEN 1:1", "GEN 1:2", "GEN 1:3", "GEN 1:4", "GEN 1:5"],
                section_heading="The Creation / The First Day",
                outline_path="1. God and the World > a. The Creation",
            ),
        ),
        ChunkDocument(
            chunk_id="scripture:JHN:3:16-18:0",
            text=(
                "[Book: John | Passage: John 3:16-18 | Testament: NT | Heading: God So Loved the World]\n"
                "[John 3:16] For God so loved the world that He gave His one and only Son, "
                "that everyone who believes in Him shall not perish but have eternal life."
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:JHN:3:16-18:0",
                source_type=SourceType.SCRIPTURE,
                book_code="JHN",
                book_name="John",
                testament="NT",
                canonical_order=43,
                chapter=3,
                verse_start=16,
                verse_end=18,
                canonical_ref="John 3:16-18",
                verse_refs=["JHN 3:16", "JHN 3:17", "JHN 3:18"],
                section_heading="God So Loved the World",
            ),
        ),
        ChunkDocument(
            chunk_id="scripture:PSA:23:1-6:0",
            text=(
                "[Book: Psalms | Passage: Psalm 23:1-6 | Testament: OT | Heading: The LORD Is My Shepherd]\n"
                "[Psalm 23:1] The LORD is my shepherd; I shall not want.\n"
                "[Psalm 23:2] He makes me lie down in green pastures; He leads me beside quiet waters."
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:PSA:23:1-6:0",
                source_type=SourceType.SCRIPTURE,
                book_code="PSA",
                book_name="Psalms",
                testament="OT",
                canonical_order=19,
                chapter=23,
                verse_start=1,
                verse_end=6,
                canonical_ref="Psalm 23:1-6",
                verse_refs=["PSA 23:1", "PSA 23:2", "PSA 23:3", "PSA 23:4", "PSA 23:5", "PSA 23:6"],
                section_heading="The LORD Is My Shepherd",
            ),
        ),
        ChunkDocument(
            chunk_id="scripture:1SA:9:15-17:0",
            text=(
                "[Book: 1 Samuel | Passage: 1 Samuel 9:15-17 | Testament: OT | Heading: Samuel Anoints Saul]\n"
                "[1 Samuel 9:17] When Samuel saw Saul, the LORD told him, "
                "'Here is the man of whom I spoke; he will govern My people as the first king of Israel.'"
            ),
            metadata=ChunkMetadata(
                chunk_id="scripture:1SA:9:15-17:0",
                source_type=SourceType.SCRIPTURE,
                book_code="1SA",
                book_name="1 Samuel",
                testament="OT",
                canonical_order=9,
                chapter=9,
                verse_start=15,
                verse_end=17,
                canonical_ref="1 Samuel 9:15-17",
                verse_refs=["1SA 9:15", "1SA 9:16", "1SA 9:17"],
                section_heading="Samuel Anoints Saul",
            ),
        ),
        ChunkDocument(
            chunk_id="outline:GEN:overview",
            text=(
                "[Source: BSB Book Outline | Book: Genesis (GEN) | Division: Complete Outline (1:1-50:26)]\n"
                "1. God and the World (1:1-2:25)\n"
                "  a. The Creation (1:1-2:3)\n"
                "2. Early History of Mankind (3:1-11:32)\n"
                "3. The Patriarchs (12:1-50:26)"
            ),
            metadata=ChunkMetadata(
                chunk_id="outline:GEN:overview",
                source_type=SourceType.BOOK_OUTLINE,
                book_code="GEN",
                book_name="Genesis",
                testament="OT",
                canonical_order=1,
                chapter=1,
                verse_start=1,
                verse_end=26,
                canonical_ref="Genesis Outline (1:1-50:26)",
            ),
        ),
        ChunkDocument(
            chunk_id="outline:ROM:overview",
            text=(
                "[Source: BSB Book Outline | Book: Romans (ROM) | Division: Complete Outline (1:1-16:27)]\n"
                "1. Introduction and Theme of the Gospel (1:1-17)\n"
                "2. Justification by Faith (1:18-5:21)\n"
                "3. Sanctification and Life in the Spirit (6:1-8:39)"
            ),
            metadata=ChunkMetadata(
                chunk_id="outline:ROM:overview",
                source_type=SourceType.BOOK_OUTLINE,
                book_code="ROM",
                book_name="Romans",
                testament="NT",
                canonical_order=45,
                chapter=1,
                verse_start=1,
                verse_end=27,
                canonical_ref="Romans Outline (1:1-16:27)",
            ),
        ),
        ChunkDocument(
            chunk_id="topical:forgiveness:0",
            text=(
                "[Source: BSB Topical Index | Topic: Forgiveness | Total References: 42]\n"
                "- [Ephesians 1:7] In Him we have redemption through His blood, the forgiveness of our trespasses.\n"
                "- [1 John 1:9] If we confess our sins, He is faithful and just to forgive us our sins."
            ),
            metadata=ChunkMetadata(
                chunk_id="topical:forgiveness:0",
                source_type=SourceType.TOPICAL_INDEX,
                canonical_ref="Topical Index: Forgiveness",
                topic="forgiveness",
                topic_display="Forgiveness",
                verse_refs=["EPH 1:7", "1JN 1:9"],
            ),
        ),
        ChunkDocument(
            chunk_id="concordance:grace:0",
            text=(
                '[Source: BSB Concordance | Headword: "grace" | Total Occurrences: 131]\n'
                "Book Distribution: Rom (21), 2Co (18), Eph (12), 1Co (10)\n"
                "- [Eph 2:8] For it is by grace you have been saved through faith.\n"
                "- [John 1:14] full of grace and truth."
            ),
            metadata=ChunkMetadata(
                chunk_id="concordance:grace:0",
                source_type=SourceType.CONCORDANCE,
                canonical_ref="Concordance: grace (131 Occurrences)",
                concordance_word="grace",
                concordance_occurrences=131,
                verse_refs=["EPH 2:8", "JHN 1:14"],
            ),
        ),
    ]


@pytest.fixture()
def in_memory_retriever() -> BSBHybridRetriever:
    """Create a BSBHybridRetriever backed by an in-memory Qdrant collection with sample chunks."""
    settings = Settings(
        QDRANT_MODE="memory",
        QDRANT_COLLECTION_NAME="test_bsb_retrieval_pipeline",
        RERANKER_ENABLED=False,
        RETRIEVAL_INITIAL_K=10,
        RETRIEVAL_TOP_K=4,
    )
    manager = QdrantManager(
        settings=settings,
        dense_embeddings=DeterministicHashEmbeddings(size=128),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=128,
    )
    manager.ensure_collection(recreate=True)
    manager.upsert_chunks(_sample_corpus_chunks())
    reranker = PassageReranker(settings=settings, enabled=False)
    return BSBHybridRetriever(qdrant_manager=manager, reranker=reranker, settings=settings)


# ---------------------------------------------------------------------------
# 1. Reference Parser Tests
# ---------------------------------------------------------------------------


def test_extract_bible_references_required_queries() -> None:
    # "What does Genesis 1:1-3 say?"
    gen_refs = extract_bible_references("What does Genesis 1:1-3 say?")
    assert len(gen_refs) == 1
    assert gen_refs[0].book_code == "GEN"
    assert gen_refs[0].canonical_ref == "Genesis 1:1-3"
    assert gen_refs[0].expanded_refs == ["GEN 1:1", "GEN 1:2", "GEN 1:3"]

    # "Compare John 1:1-5 and Hebrews 11:1-3"
    comp_refs = extract_bible_references("Compare John 1:1-5 and Hebrews 11:1-3")
    assert len(comp_refs) == 2
    assert comp_refs[0].book_code == "JHN"
    assert comp_refs[0].canonical_ref == "John 1:1-5"
    assert comp_refs[0].expanded_refs == [
        "JHN 1:1",
        "JHN 1:2",
        "JHN 1:3",
        "JHN 1:4",
        "JHN 1:5",
    ]
    assert comp_refs[1].book_code == "HEB"
    assert comp_refs[1].canonical_ref == "Hebrews 11:1-3"
    assert comp_refs[1].expanded_refs == ["HEB 11:1", "HEB 11:2", "HEB 11:3"]

    # "Read 1 Cor 13:4-7"
    cor_refs = extract_bible_references("Read 1 Cor 13:4-7")
    assert len(cor_refs) == 1
    assert cor_refs[0].book_code == "1CO"
    assert cor_refs[0].canonical_ref == "1 Corinthians 13:4-7"
    assert cor_refs[0].expanded_refs == ["1CO 13:4", "1CO 13:5", "1CO 13:6", "1CO 13:7"]

    # "Psalm 23:1"
    psa_refs = extract_bible_references("Psalm 23:1")
    assert len(psa_refs) == 1
    assert psa_refs[0].book_code == "PSA"
    assert psa_refs[0].canonical_ref == "Psalm 23:1"
    assert psa_refs[0].expanded_refs == ["PSA 23:1"]


def test_extract_bible_references_cross_chapter_whole_chapter_and_cap() -> None:
    # Cross-chapter range: Genesis 1:1-2:3 (31 verses in Gen 1 + 3 verses in Gen 2 = 34)
    cross = extract_bible_references("Explain Genesis 1:1-2:3 in context")
    assert len(cross) == 1
    assert cross[0].canonical_ref == "Genesis 1:1-2:3"
    assert cross[0].start_chapter == 1 and cross[0].start_verse == 1
    assert cross[0].end_chapter == 2 and cross[0].end_verse == 3
    assert len(cross[0].expanded_refs) == 34
    assert cross[0].expanded_refs[0] == "GEN 1:1"
    assert cross[0].expanded_refs[-1] == "GEN 2:3"

    # Whole chapter: Psalm 23 (6 verses) and 1 Corinthians 13 (13 verses)
    whole = extract_bible_references("Read Psalm 23 and 1 Corinthians 13")
    assert [r.canonical_ref for r in whole] == ["Psalm 23", "1 Corinthians 13"]
    assert len(whole[0].expanded_refs) == 6
    assert len(whole[1].expanded_refs) == 13

    # En-dash range + abbreviation with period
    endash = extract_bible_references("See Gen. 1:1 and Revelation 13:16–18")
    assert [r.canonical_ref for r in endash] == ["Genesis 1:1", "Revelation 13:16-18"]
    assert endash[1].expanded_refs == ["REV 13:16", "REV 13:17", "REV 13:18"]

    # Capped at MAX_EXPANDED_REFS (50) for long chapters like Psalm 119
    long_ch = extract_bible_references("Summarize Psalm 119")
    assert len(long_ch) == 1
    assert len(long_ch[0].expanded_refs) == MAX_EXPANDED_REFS
    assert long_ch[0].expanded_refs[0] == "PSA 119:1"
    assert long_ch[0].expanded_refs[-1] == "PSA 119:50"

    # Unrelated numeric phrases must not produce false-positive references
    assert extract_bible_references("Give me the top 10 verses about love") == []
    assert extract_bible_references("I am 3 years old and did my job 2 days ago") == []


# ---------------------------------------------------------------------------
# 2. Query Router Tests (All 5 QueryIntent Categories)
# ---------------------------------------------------------------------------


def test_route_query_all_five_intents() -> None:
    # 1. EXACT_PASSAGE
    p_exact = route_query("What does John 3:16-18 teach?")
    assert p_exact.intent == QueryIntent.EXACT_PASSAGE
    assert p_exact.book_codes == ["JHN"]
    assert len(p_exact.verse_ranges) == 1
    assert p_exact.target_sources == [SourceType.SCRIPTURE, SourceType.BOOK_OUTLINE]

    # 2. OUTLINE
    p_outline = route_query("What are the main divisions and outline of the book of Romans?")
    assert p_outline.intent == QueryIntent.OUTLINE
    assert p_outline.book_codes == ["ROM"]
    assert p_outline.target_sources == [SourceType.BOOK_OUTLINE, SourceType.SCRIPTURE]

    # 3. CONCORDANCE
    p_conc = route_query('How many occurrences of the word "grace" are in the concordance?')
    assert p_conc.intent == QueryIntent.CONCORDANCE
    assert p_conc.concordance_word == "grace"
    assert p_conc.target_sources == [SourceType.CONCORDANCE, SourceType.SCRIPTURE]

    p_conc_unquoted = route_query("How many occurrences of faith appear in the BSB?")
    assert p_conc_unquoted.intent == QueryIntent.CONCORDANCE
    assert p_conc_unquoted.concordance_word == "faith"

    # 4. TOPICAL
    p_topic = route_query("What verses are about forgiveness in the topical index?")
    assert p_topic.intent == QueryIntent.TOPICAL
    assert p_topic.topic_keyword == "forgiveness"
    assert p_topic.target_sources == [SourceType.TOPICAL_INDEX, SourceType.SCRIPTURE]

    # 5. HYBRID_GENERAL
    p_general = route_query("Who was anointed as the first king of Israel?")
    assert p_general.intent == QueryIntent.HYBRID_GENERAL
    assert set(p_general.target_sources) == {
        SourceType.SCRIPTURE,
        SourceType.TOPICAL_INDEX,
        SourceType.BOOK_OUTLINE,
        SourceType.CONCORDANCE,
    }


# ---------------------------------------------------------------------------
# 3. Reranker & Hybrid Retriever Integration Tests
# ---------------------------------------------------------------------------


def test_reranker_deduplication_and_exact_match_boost() -> None:
    chunks = _sample_corpus_chunks()
    gen_chunk = chunks[0]
    jhn_chunk = chunks[1]

    plan = route_query("What does Genesis 1:1-3 say?")
    candidates = [
        RetrievalResult(chunk=jhn_chunk, score=0.95, rank=1),
        RetrievalResult(chunk=gen_chunk, score=0.60, rank=2),
        RetrievalResult(chunk=gen_chunk, score=0.80, rank=3),  # duplicate with higher score
    ]

    class _StubRanker:
        def rerank(self, request: Any) -> list[dict[str, Any]]:
            # Give both chunks identical cross-encoder score so the +0.25 exact bonus decides #1
            return [{"id": p["id"], "score": 0.70} for p in request.passages]

    reranker = PassageReranker(ranker=_StubRanker(), enabled=True)
    ranked = reranker.rerank(plan.original_query, candidates, plan=plan, top_k=5)

    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "scripture:GEN:1:1-5:0"
    assert ranked[0].rank == 1
    assert ranked[0].score == pytest.approx(0.80)
    assert ranked[0].rerank_score == pytest.approx(0.70 + EXACT_MATCH_BONUS)
    assert ranked[1].chunk.chunk_id == "scripture:JHN:3:16-18:0"
    assert ranked[1].rank == 2


def test_hybrid_retriever_ranks_exact_target_first_for_all_query_types(
    in_memory_retriever: BSBHybridRetriever,
) -> None:
    # 1. EXACT_PASSAGE query -> Genesis 1:1-5 chunk ranked #1
    plan_exact, res_exact = in_memory_retriever.retrieve("What does Genesis 1:1-3 say?")
    assert plan_exact.intent == QueryIntent.EXACT_PASSAGE
    assert len(res_exact) >= 1
    assert res_exact[0].rank == 1
    assert res_exact[0].chunk.chunk_id == "scripture:GEN:1:1-5:0"

    # 2. OUTLINE query -> Romans outline chunk ranked #1
    plan_outline, res_outline = in_memory_retriever.retrieve(
        "What is the outline and structure of Romans?"
    )
    assert plan_outline.intent == QueryIntent.OUTLINE
    assert len(res_outline) >= 1
    assert res_outline[0].rank == 1
    assert res_outline[0].chunk.chunk_id == "outline:ROM:overview"

    # 3. CONCORDANCE query -> "grace" concordance chunk ranked #1
    plan_conc, res_conc = in_memory_retriever.retrieve(
        'How many occurrences of the word "grace" are in the concordance?'
    )
    assert plan_conc.intent == QueryIntent.CONCORDANCE
    assert len(res_conc) >= 1
    assert res_conc[0].rank == 1
    assert res_conc[0].chunk.chunk_id == "concordance:grace:0"

    # 4. TOPICAL query -> "forgiveness" topical index chunk ranked #1
    plan_topic, res_topic = in_memory_retriever.retrieve(
        "What verses are on the topic of forgiveness?"
    )
    assert plan_topic.intent == QueryIntent.TOPICAL
    assert len(res_topic) >= 1
    assert res_topic[0].rank == 1
    assert res_topic[0].chunk.chunk_id == "topical:forgiveness:0"

    # 5. HYBRID_GENERAL query -> 1 Samuel 9:15-17 chunk ranked #1
    plan_gen, res_gen = in_memory_retriever.retrieve(
        "Samuel saw Saul govern My people as the first king of Israel"
    )
    assert plan_gen.intent == QueryIntent.HYBRID_GENERAL
    assert len(res_gen) >= 1
    assert res_gen[0].rank == 1
    assert res_gen[0].chunk.chunk_id == "scripture:1SA:9:15-17:0"


def test_format_context_and_langchain_retriever_adapter(
    in_memory_retriever: BSBHybridRetriever,
) -> None:
    _, results = in_memory_retriever.retrieve("Psalm 23:1", top_k=2)
    assert results[0].chunk.chunk_id == "scripture:PSA:23:1-6:0"

    formatted = BSBHybridRetriever.format_context_for_llm(results)
    assert "=== SOURCE [1] | Reference: Psalm 23:1-6 | Type: scripture ===" in formatted
    assert "The LORD is my shepherd; I shall not want." in formatted

    lc_retriever = in_memory_retriever.as_langchain_retriever(top_k=2)
    assert isinstance(lc_retriever, BSBLangChainRetriever)
    docs = lc_retriever.invoke("Psalm 23:1")
    assert len(docs) >= 1
    assert docs[0].metadata["chunk_id"] == "scripture:PSA:23:1-6:0"
    assert docs[0].metadata["rank"] == 1
    assert docs[0].metadata["intent"] == QueryIntent.EXACT_PASSAGE.value

