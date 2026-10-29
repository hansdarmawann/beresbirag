"""Multi-intent routing, topic hygiene, and the book-scoped two-branch hybrid filter."""

from __future__ import annotations

import pytest
from qdrant_client import models

from beresbirag.config import Settings
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, QueryIntent, SourceType
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever
from beresbirag.retrieval.query_router import INTENT_SOURCES, route_query
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager

S = SourceType
I = QueryIntent  # noqa: E741

README_QUERY = "What is the significance of 666 in the BSB Topical Index and Revelation 13:18?"


@pytest.mark.parametrize(
    ("query", "intent", "secondary", "sources", "topic", "word", "books"),
    [
        (
            README_QUERY,
            I.EXACT_PASSAGE, [I.TOPICAL],
            [S.SCRIPTURE, S.BOOK_OUTLINE, S.TOPICAL_INDEX], "666", None, ["REV"],
        ),
        (
            "Explain the topic of baptism in Acts 2",
            I.EXACT_PASSAGE, [I.TOPICAL],
            [S.SCRIPTURE, S.BOOK_OUTLINE, S.TOPICAL_INDEX], "baptism", None, ["ACT"],
        ),
        (
            "What does the Bible say about forgiveness?",
            I.TOPICAL, [], [S.TOPICAL_INDEX, S.SCRIPTURE], "forgiveness", None, [],
        ),
        (
            "Tell me about grace",
            I.TOPICAL, [], [S.TOPICAL_INDEX, S.SCRIPTURE], "grace", None, [],
        ),
        (
            "Is there a verse about love in 1 John?",
            I.TOPICAL, [], [S.TOPICAL_INDEX, S.SCRIPTURE], "love", None, ["1JN"],
        ),
        (
            "Where does the word 'grace' appear?",
            I.CONCORDANCE, [], [S.CONCORDANCE, S.SCRIPTURE], None, "grace", [],
        ),
    ],
)  # fmt: skip
def test_problem_statement_queries(query, intent, secondary, sources, topic, word, books) -> None:
    plan = route_query(query)
    assert plan.intent == intent
    assert plan.secondary_intents == secondary
    assert plan.target_sources == sources
    assert plan.topic_keyword == topic
    assert plan.concordance_word == word
    assert plan.book_codes == books


def test_reference_spans_never_become_topics() -> None:
    plan = route_query("Tell me about John 3:16")
    assert plan.intent == I.EXACT_PASSAGE
    assert plan.topic_keyword is None
    assert plan.secondary_intents == []


def test_book_name_alone_is_not_a_topic() -> None:
    plan = route_query("Tell me about Genesis")
    assert plan.intent == I.HYBRID_GENERAL
    assert plan.topic_keyword is None
    assert plan.book_codes == ["GEN"]


def test_topic_words_are_not_book_mentions() -> None:
    plan = route_query("Topic: Mark of the Beast and Revelation 13:16-18")
    assert plan.topic_keyword == "mark of the beast"
    assert plan.book_codes == ["REV"]


def test_age_is_not_a_reference() -> None:
    plan = route_query("I am 25 years old")
    assert plan.verse_ranges == []
    assert plan.intent == I.HYBRID_GENERAL


@pytest.mark.parametrize(
    ("query", "concordance_word"),
    [
        ("How often does the word mercy occur?", "mercy"),
        ("How often does love appear?", "love"),
        ('Is the word "selah" common?', "selah"),
    ],
)
def test_new_concordance_patterns(query: str, concordance_word: str) -> None:
    plan = route_query(query)
    assert plan.intent == I.CONCORDANCE
    assert plan.concordance_word == concordance_word


def test_outline_secondary_and_precedence() -> None:
    plan = route_query("Outline of Romans and what Romans 8:28 says")
    assert plan.intent == I.EXACT_PASSAGE
    assert plan.secondary_intents == [I.OUTLINE]
    assert plan.book_codes == ["ROM"]


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("John 3:16", I.EXACT_PASSAGE),
        ("Genesis outline", I.OUTLINE),
        ("Occurrences of 12", I.CONCORDANCE),
        ("Topic: 666", I.TOPICAL),
        ("Who was anointed as the first king of Israel?", I.HYBRID_GENERAL),
    ],
)
def test_single_intent_sources_are_unchanged(query: str, intent: QueryIntent) -> None:
    plan = route_query(query)
    assert plan.intent == intent
    assert plan.secondary_intents == []
    assert plan.target_sources == list(INTENT_SOURCES[intent])


# ---------------------------------------------------------------------------
# Book-scoped hybrid filter on an in-memory Qdrant
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, **meta) -> ChunkDocument:
    return ChunkDocument(chunk_id=chunk_id, text=text, metadata=ChunkMetadata(chunk_id=chunk_id, **meta))


@pytest.fixture()
def retriever() -> BSBHybridRetriever:
    settings = Settings(
        _env_file=None, QDRANT_MODE="memory", QDRANT_COLLECTION_NAME="test_multi_intent"
    )
    manager = QdrantManager(
        settings=settings,
        dense_embeddings=DeterministicHashEmbeddings(size=384),
        sparse_embeddings=DeterministicBM25SparseEmbeddings(),
        dense_vector_size=384,
    )
    manager.ensure_collection(recreate=True)
    manager.upsert_chunks(
        [
            _chunk(
                "scripture:1JN:4:7-8:0",
                "[1 John 4:7] Beloved, let us love one another, because love comes from God.",
                source_type=S.SCRIPTURE, book_code="1JN", chapter=4, verse_start=7, verse_end=8,
                canonical_ref="1 John 4:7-8", verse_refs=["1JN 4:7", "1JN 4:8"],
            ),
            _chunk(
                "scripture:JHN:15:12-13:0",
                "[John 15:12] Love one another as I have loved you. Greater love has no one.",
                source_type=S.SCRIPTURE, book_code="JHN", chapter=15, verse_start=12,
                verse_end=13, canonical_ref="John 15:12-13", verse_refs=["JHN 15:12", "JHN 15:13"],
            ),
            _chunk(
                "scripture:REV:13:16-18:0",
                "[Revelation 13:18] Let the one who has insight calculate the number of the "
                "beast, for it is the number of a man, and that number is 666.",
                source_type=S.SCRIPTURE, book_code="REV", chapter=13, verse_start=16,
                verse_end=18, canonical_ref="Revelation 13:16-18",
                verse_refs=["REV 13:16", "REV 13:17", "REV 13:18"],
            ),
            _chunk(
                "topical:666:0",
                "[Source: BSB Topical Index | Topic: 666] Revelation 13:16-18",
                source_type=S.TOPICAL_INDEX, canonical_ref="Topical Index: 666", topic="666",
                verse_refs=["REV 13:16", "REV 13:17", "REV 13:18"],
            ),
            _chunk(
                "topical:Loving%20Others:0",
                "[Source: BSB Topical Index | Topic: Loving Others] love one another",
                source_type=S.TOPICAL_INDEX, canonical_ref="Topical Index: Loving Others",
                topic="loving others",
            ),
            _chunk(
                "concordance:love:0",
                '[Source: BSB Concordance | Headword: "love"] love love love',
                source_type=S.CONCORDANCE, canonical_ref="Concordance: love (373 Occurrences)",
                concordance_word="love",
            ),
            _chunk(
                "outline:REV:overview",
                "[Source: BSB Book Outline | Book: Revelation (REV)] The Letters to the Seven Churches",
                source_type=S.BOOK_OUTLINE, book_code="REV", canonical_ref="Revelation Outline (1:1-22:21)",
            ),
        ]  # fmt: skip
    )
    return BSBHybridRetriever(
        qdrant_manager=manager,
        reranker=PassageReranker(settings=settings, enabled=False),
        settings=settings,
    )


def _match_values(flt: models.Filter, key: str) -> set[str]:
    values: set[str] = set()
    for cond in flt.must or []:
        if isinstance(cond, models.FieldCondition) and cond.key == key:
            match = cond.match
            values.update(match.any if isinstance(match, models.MatchAny) else [match.value])
    return values


def test_book_scoped_filter_has_two_should_branches(retriever: BSBHybridRetriever) -> None:
    flt = retriever._hybrid_filter(route_query("Is there a verse about love in 1 John?"))
    assert flt is not None and flt.should is not None and not flt.must
    scoped, others = flt.should
    assert _match_values(scoped, "metadata.source_type") == {"scripture"}
    assert _match_values(scoped, "metadata.book_code") == {"1JN"}
    assert _match_values(others, "metadata.source_type") == {"topical_index"}
    assert _match_values(others, "metadata.book_code") == set()


def test_book_scoped_filter_without_other_sources_is_single_branch(
    retriever: BSBHybridRetriever,
) -> None:
    flt = retriever._hybrid_filter(route_query("John 3:16"))
    assert flt is not None and not flt.should
    assert _match_values(flt, "metadata.source_type") == {"scripture", "book_outline"}
    assert _match_values(flt, "metadata.book_code") == {"JHN"}


def test_unscoped_filter_unchanged(retriever: BSBHybridRetriever) -> None:
    assert retriever._hybrid_filter(route_query("Who was the first king of Israel?")) is None
    flt = retriever._hybrid_filter(route_query("Topic: 666"))
    assert flt is not None
    assert _match_values(flt, "metadata.source_type") == {"topical_index", "scripture"}


def test_book_scoped_search_keeps_non_scripture_targets(retriever: BSBHybridRetriever) -> None:
    _, results = retriever.retrieve("Is there a verse about love in 1 John?", top_k=6)
    ids = [r.chunk.chunk_id for r in results]
    assert "scripture:1JN:4:7-8:0" in ids
    assert "topical:Loving%20Others:0" in ids  # other-sources branch still searched
    assert "scripture:JHN:15:12-13:0" not in ids  # scripture is scoped to 1 John
    assert "concordance:love:0" not in ids  # not a target source


def test_readme_mixed_query_returns_scripture_and_topic(retriever: BSBHybridRetriever) -> None:
    plan, results = retriever.retrieve(README_QUERY, top_k=6)
    ids = [r.chunk.chunk_id for r in results]
    assert plan.topic_keyword == "666"
    assert "scripture:REV:13:16-18:0" in ids
    assert "topical:666:0" in ids


def _ranked(chunk: ChunkDocument, rank: int):
    from beresbirag.models.schemas import RetrievalResult

    return RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank)


def test_secondary_intent_exact_match_is_not_crowded_out() -> None:
    plan = route_query("Where does the word 'beast' appear, and what does Revelation 13:18 say?")
    rev = [
        _chunk(f"scripture:REV:13:{v}-18:0", "beast 666", source_type=S.SCRIPTURE,
               book_code="REV", verse_refs=["REV 13:18"])
        for v in (11, 14, 16)
    ]  # fmt: skip
    outline = _chunk("outline:REV:overview", "outline", source_type=S.BOOK_OUTLINE, book_code="REV")
    concordance = _chunk(
        "concordance:beast:0", "beast", source_type=S.CONCORDANCE, concordance_word="beast"
    )
    ranked = [_ranked(c, i) for i, c in enumerate([*rev, outline, concordance], start=1)]
    exact = [ranked[0], ranked[1], ranked[2], ranked[4]]

    top3 = BSBHybridRetriever._cover_intents(plan, ranked, exact, 3)
    assert [r.chunk.chunk_id for r in top3] == [
        "scripture:REV:13:11-18:0", "scripture:REV:13:14-18:0", "concordance:beast:0",
    ]  # fmt: skip
    assert [r.rank for r in top3] == [1, 2, 3]

    # The primary intent's best exact match is never displaced.
    top1 = BSBHybridRetriever._cover_intents(plan, ranked, exact, 1)
    assert [r.chunk.chunk_id for r in top1] == ["scripture:REV:13:11-18:0"]
