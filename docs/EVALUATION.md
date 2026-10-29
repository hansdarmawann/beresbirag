# Evaluating beresbirag

This page explains how to measure retrieval and answer quality, what each metric means, and how to
choose the offline LLM for a small machine. Every number in it comes from a command you can rerun.

There are three tools:

| Tool | What it checks | Needs |
| :--- | :--- | :--- |
| `scripts/evaluate_retrieval.py --verify-cases` | Every test case is satisfiable by the real BSB files | The four raw BSB files |
| `scripts/evaluate_retrieval.py` | Retrieval: Recall@1/@5, MRR, range coverage, latency per category | Raw BSB files (in-memory slice) or an indexed Qdrant |
| `scripts/evaluate_answers.py` | Answers: citation precision, must-cite recall, refusals, latency, tokens, RAM | An indexed Qdrant plus each LLM under test (or `--fake-llm`) |

> [!NOTE]
> Inside a Git worktree, the `beresbirag` conda environment imports the package from the main
> checkout (editable install). To run the scripts against the worktree's own code, set
> `PYTHONPATH=src` first, for example `$env:PYTHONPATH = "src"` in PowerShell. pytest already does
> this through `pythonpath = ["src"]`.

## 1. Test cases

Cases live in two JSONL files, one JSON object per line. Unknown fields and duplicate ids are
rejected when a file is loaded.

### `eval/retrieval_cases.jsonl`

```json
{"id": "R05", "category": "range_completeness", "query": "Genesis 1:31-2:3",
 "expected": {"source_type": "scripture", "verse_refs_all": ["GEN 1:31", "GEN 2:1", "GEN 2:2", "GEN 2:3"]},
 "tags": ["cross-chapter"]}
```

`expected` fields are all optional; every field that is set must hold:

| Field | Meaning |
| :--- | :--- |
| `source_type` | The chunk comes from `scripture`, `book_outline`, `topical_index` or `concordance` |
| `verse_refs_any` | The chunk covers at least one of these verse keys |
| `verse_refs_all` | The top-k results together cover all of these verse keys (range completeness) |
| `book_code` | The chunk belongs to this book (USJ code such as `ROM`) |
| `topic` / `concordance_word` | The chunk is this topical-index topic / concordance headword |
| `text_contains` | The chunk text contains this string (case-insensitive) |

`unanswerable: true` marks a query that has no exact answer in the corpus, such as a
nonexistent chapter or an absent topic or word. `tags` are free-form. The tag `known-gap` marks a
case the system is known to fail today (see [Known gaps](#5-known-gaps)).

Categories and their current counts (known-gap cases are not counted):

| Category | N | What it tests |
| :--- | ---: | :--- |
| `exact` | 11 | Single verses, abbreviations (`Rom 8:28`, `Ps 119:105`), single-chapter books (`Jude 3`) |
| `range_completeness` | 8 | Every verse of a range is retrieved, including the cross-chapter `Genesis 1:31-2:3` |
| `footnote_crossref` | 6 | Footnote and cross-reference text attached to a verse |
| `outline` | 5 | Book outline chunks |
| `topical_explicit` | 6 | Queries that name the topical index or a topic |
| `topical_implicit_paraphrase` | 11 | "What does the Bible say about X?" and paraphrases with no topic keyword |
| `concordance` | 6 | Word-occurrence questions |
| `mixed_intent` | 6 | Two intents in one query, including the README example |
| `book_scoped_semantic` | 7 | Semantic questions limited to one book ("…love in 1 John?") |
| `unanswerable` | 6 | Nonexistent chapters, absent topics/words, out-of-scope questions |

The 15 original benchmark queries keep their ids `Q01`–`Q15` and their meaning. `Q01`
(`Genesis 1:1-3`) now requires all three verses rather than only `GEN 1:1`.

### `eval/answer_cases.jsonl`

```json
{"id": "A02", "query": "What is the significance of 666 in the BSB Topical Index and Revelation 13:18?",
 "must_cite": ["REV 13:18", "BSB Topical Index: 666"]}
{"id": "N02", "query": "What does Genesis 51 say?", "should_refuse": true}
```

`must_cite` items are verse keys (`JHN 3:16`) or labels (`BSB Topical Index: 666`,
`BSB Concordance: grace`, `BSB Book Outline: Romans`). `should_refuse: true` means the right answer
is the insufficient-context refusal. The file has 32 cases: 7 refusals and 3 Indonesian known-gap
questions.

### Ground-truth verification

```powershell
conda run -n beresbirag python ./scripts/evaluate_retrieval.py --verify-cases
```

This scans the four raw files once (about 3 minutes, most of it the 52 MB concordance) and exits 1,
listing case ids, when any expectation cannot be satisfied:

- a verse key does not exist in BSB versification (`BSB_VERSE_COUNTS`);
- a `topic` is absent from the topical index, or a `concordance_word` from the concordance;
- `text_contains` is not in the indexed text of the referenced verses, topic, headword or outline;
- an `unanswerable` query does parse to real verses, or names a topic or word that exists;
- an answer case `must_cite`s a nonexistent verse, topic or headword.

Every committed case must pass. Add cases from the corpus, never from memory; for example, the
topical index has no `grace`, `love` or `forgiveness` topics, but it does have `God's Grace`,
`Loving Others`, `Forgiving Others` and even `Video Games`.

## 2. Metrics

### Retrieval

| Metric | Definition |
| :--- | :--- |
| **Hit rank** | The first rank at which the case is fully satisfied: the first chunk matching every per-chunk criterion and, with `verse_refs_all`, the rank by which the results so far cover every required verse. |
| **Recall@k** | Share of answerable cases with hit rank ≤ k. |
| **MRR** | Mean of 1 / hit rank (a miss counts 0). |
| **Coverage** | For `verse_refs_all` cases: share of required verses in the union of the top-k chunks of the expected source type. Mean over cases that have it. |
| **Pass** | Answerable: hit rank ≤ k (default k = 5). Unanswerable: no result is presented as an exact structural match (the reranker's exact-match test), meaning the system did not invent a match for a nonexistent chapter, topic or word. |
| **p50 / p95 s** | Wall-clock `retriever.retrieve()` latency percentiles (linear interpolation). |

Recall and MRR are computed over answerable cases only. Unanswerable cases appear in the Pass
column only.

### Answers

| Metric | Definition |
| :--- | :--- |
| **Citation precision** | valid / (valid + partial + unsupported) bracket citations, micro-averaged over all answers. Unparsed brackets (`[see above]`) are excluded. |
| **Must-cite recall** | Share of `must_cite` items covered by a valid or partial citation, averaged over cases that have any. A numeric citation `[2]` covers the verses and label of retrieved chunk 2. |
| **Should-refuse recall** | Share of `should_refuse` cases that the pipeline refused. |
| **False-refusal rate** | Share of answerable cases that the pipeline refused. |
| **Refusal** | `answer_mode == insufficient_context`, or the answer contains the standard insufficient-context sentence. |
| **Answer modes** | Count of `llm`, `fast_path` and `insufficient_context` answers. |
| **Latency** | p50/p95 of retrieval (`prepare`) and generation (`stream`) time; the decision rule below uses p95 of their sum. |
| **Tokens in/out** | Summed from LangChain's `get_usage_metadata_callback()` when the provider reports usage. |
| **Peak RAM** | With `--sample-ram`: peak *system* memory in use (`psutil.virtual_memory().used`), sampled every 250 ms. It includes the OS, Qdrant and Ollama, which is what matters on a 4 GB machine. |
| **Judge** | With `--judge`: share of LLM answers an LLM judge marks fully supported by the retrieved context. |

### Citation validation (`CITATION_POLICY`)

Every response carries a `citation_report` that classifies each `[bracket]` in the answer:

| Citation | Valid when |
| :--- | :--- |
| `[3]` | 1 ≤ n ≤ number of retrieved chunks |
| `[BSB Topical Index: X]` | A retrieved topical chunk has topic X (singular/plural tolerated) |
| `[BSB Concordance: X]` | A retrieved concordance chunk has headword X |
| `[BSB Book Outline: X]` / `[Romans Outline (…)]` | A retrieved outline chunk belongs to book X |
| Any exact copy of a source's `Reference:` header | That source was retrieved |
| `[John 3:16]`, `[Gen 1:1-3; Rom 5:8]` | All referenced verses were retrieved; some = **partial**; none = **unsupported** |
| `[Revelation 99:99]`, `[Genesis 51]` | Never: a book name plus numbers that is not a real reference is **unsupported** |
| `[see above]` | **Unparsed** (not a citation) |

`CitationItem.used` marks the chunks that support a valid or partial citation, and `citations`
lists them first.

- `CITATION_POLICY=annotate` (default) only reports.
- `CITATION_POLICY=strict` removes unsupported citations from the answer and appends
  `> ⚠️ Removed n citation(s) not found in the retrieved BSB context: …`. An LLM answer left
  with no valid or partial citation is replaced by the insufficient-context answer.
- Fast-path answers (verbatim BSB text) are never rewritten under either policy.

## 3. Running the retrieval evaluation

```powershell
# Deterministic, offline: hash embeddings, reranker off, in-memory slice
conda run -n beresbirag python ./scripts/evaluate_retrieval.py

# Honest semantic baseline: FastEmbed bge-small + BM25, FlashRank reranker on
conda run -n beresbirag python ./scripts/evaluate_retrieval.py --embeddings real --reranker on

# Full corpus in the configured Qdrant (after scripts/ingest_all.py)
conda run -n beresbirag python ./scripts/evaluate_retrieval.py --use-existing-collection

# The original 15-query benchmark (CI-compatible)
conda run -n beresbirag python ./scripts/evaluate_retrieval.py --legacy
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--cases PATH` | `./eval/retrieval_cases.jsonl` | Case file |
| `--legacy` | off | Run the original 15 in-code queries (`BENCHMARK_QUERIES`) and their report |
| `--embeddings hash\|real` | `hash` | In-memory slice embeddings: deterministic hashes, or the configured FastEmbed/OpenAI models |
| `--reranker on\|off` | on with `real` or an existing collection, else off | FlashRank cross-encoder |
| `--in-memory-slice` / `--use-existing-collection` | slice | Build a slice from the raw files, or query `QDRANT_*` as configured |
| `--categories a,b` | all | Evaluate only these categories |
| `--verify-cases` | off | Ground-truth verification only (also checks `--answer-cases`) |
| `--report-json PATH` | `./eval/reports/retrieval-<timestamp>.json` | Machine-readable report with every case |
| `--top-k N` | 5 | Results per query (at least 5) |
| `--distractor-rate R` | 0.1 | Share of non-target topics and headwords indexed in the slice |
| `--books A,B` | – | Legacy: books in the slice. Case suite: extra books added to the slice |
| `--min-recall-at-5 X` | 0.90 with `--legacy`, otherwise off | Exit 1 when Recall@5 on non-gap answerable cases is below X |

**The in-memory slice** contains all 66 book outlines, the full scripture of every book a case
references (23 books today), every topic and headword a case or its query plan names, and about
10% of the other 3,146 topics and 14,782 headwords as distractors (deterministic CRC32 sampling).
It is much harder than the old 15-topic / 15-headword slice. It is still easier than the full
corpus, so treat `--use-existing-collection` numbers as the real ones.

**Output.** A per-category table (N, R@1, R@5, MRR, coverage, pass rate, p50/p95 latency) with an
OVERALL row, a table of failed cases with what came back instead, and a separate table for
`known-gap` cases.

## 4. Running the answer evaluation

```powershell
# CI smoke: synthetic in-memory corpus, deterministic fake LLM, no network
conda run -n beresbirag python ./scripts/evaluate_answers.py --fake-llm

# Real models against the indexed collection
conda run -n beresbirag python ./scripts/evaluate_answers.py --models ollama:qwen2.5:1.5b,openai:gpt-4o-mini --sample-ram
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--cases PATH` | `./eval/answer_cases.jsonl` | Case file |
| `--models LIST` | configured `LLM_PROVIDER:LLM_MODEL` | Comma list of `provider:model`, split on the **first** colon, so `ollama:qwen2.5:1.5b` works |
| `--fake-llm` | off | Deterministic smoke run for CI |
| `--sample-ram` | off | Peak system RAM, sampled every 250 ms |
| `--judge provider:model` | off | LLM-as-judge faithfulness check returning `{supported, unsupported_claims}` |
| `--citation-policy annotate\|strict` | `CITATION_POLICY` | Policy used while answering |
| `--top-k N` | `RETRIEVAL_TOP_K` | Retrieved chunks per question |
| `--report-json PATH` | `./eval/reports/answers-<timestamp>.json` | Per-model summary plus every answer and citation report |

A model that cannot start (missing API key, Ollama not running) is reported as an error and the
sweep continues. A model that fails 3 cases in a row is skipped.

## 5. Known gaps

`known-gap` cases are run and reported in their own table. They are excluded from the OVERALL
row, from `--min-recall-at-5`, and from model choice. Each one is a measured, documented failure,
not a hidden one. Fixing one means removing its tag.

| Tag | Cases | Cause |
| :--- | :--- | :--- |
| `id-lang` | 3 retrieval, 3 answer | Indonesian book names (`Yohanes`, `Kejadian`) are not reference aliases, and the embeddings are English-only. |
| `psalm-superscription` | `G01` `Psalm 23:1`, `G02` `Psalm 23` | Ingestion drops verse 1 of psalms that have a superscription: no chunk contains `PSA 23:1` ("The LORD is my shepherd"), and the first `PSA 23` chunk starts at verse 2. The bug is in `ingestion/usj_parser.py`. |

## 6. Choosing the offline model (dual-core / 4 GB)

Run this on the target machine itself, with Qdrant and Ollama running, the collection ingested,
`RESOURCE_PROFILE=low` and `OFFLINE_MODE=true`, and nothing else open:

```powershell
foreach ($m in "qwen2.5:0.5b","qwen2.5:1.5b","llama3.2:1b","gemma3:1b","qwen3:1.7b") { ollama pull $m }
conda run -n beresbirag python ./scripts/evaluate_answers.py --models ollama:qwen2.5:0.5b,ollama:qwen2.5:1.5b,ollama:llama3.2:1b,ollama:gemma3:1b,ollama:qwen3:1.7b --sample-ram
```

**Decision rule.** Pick the model with the highest citation precision, subject to:

- **p95 total latency ≤ 90 s** (retrieval + generation), and
- **peak system RAM ≤ 3.6 GB**.

Then set `DEFAULT_MODELS["ollama"]` in `src/beresbirag/llm/factory.py` to the winner, and
record the run below with its date, the machine, and the report file.

| Model | Cite prec | Must-cite | Refuse rec | False ref | p95 total s | Peak RAM GB | Eligible |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `qwen2.5:0.5b` | TODO (run on target hardware) | | | | | | |
| `qwen2.5:1.5b` *(current default)* | TODO (run on target hardware) | | | | | | |
| `llama3.2:1b` | TODO (run on target hardware) | | | | | | |
| `gemma3:1b` | TODO (run on target hardware) | | | | | | |
| `qwen3:1.7b` | TODO (run on target hardware) | | | | | | |

`DEFAULT_MODELS["ollama"]` stays `qwen2.5:1.5b` until this table holds a real run. The
comparison has not yet been run on dual-core / 4 GB hardware; the machine that produced the
baseline below has 4 cores and 17 GB.

## 7. Baseline results

Run on 2026-10-29 on branch `feat/quality-eval`, on a 4-core / 17 GB Windows machine (not the
dual-core target):

```powershell
conda run -n beresbirag python ./scripts/evaluate_retrieval.py --embeddings real --reranker on
```

In-memory slice with FastEmbed `BAAI/bge-small-en-v1.5` + `Qdrant/bm25`, the FlashRank
`ms-marco-MiniLM-L-12-v2` reranker, and top-k 5. The slice held 4,570 chunks (23 books, all
outlines, 10% distractor topics and headwords) and took 1,668 s to index.

| Category | N | R@1 | R@5 | MRR | Coverage | Pass | p50 s | p95 s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| range_completeness | 8 | 87.5% | 100.0% | 0.938 | 100.0% | 100.0% | 6.69 | 7.30 |
| exact | 11 | 90.9% | 100.0% | 0.939 | – | 100.0% | 6.53 | 6.99 |
| footnote_crossref | 6 | 100.0% | 100.0% | 1.000 | – | 100.0% | 6.60 | 7.06 |
| outline | 5 | 40.0% | 100.0% | 0.600 | – | 100.0% | 6.97 | 8.18 |
| topical_explicit | 6 | 100.0% | 100.0% | 1.000 | – | 100.0% | 6.86 | 7.02 |
| concordance | 6 | 100.0% | 100.0% | 1.000 | – | 100.0% | 6.67 | 7.49 |
| topical_implicit_paraphrase | 11 | 100.0% | 100.0% | 1.000 | – | 100.0% | 6.61 | 7.16 |
| mixed_intent | 6 | 66.7% | 100.0% | 0.733 | – | 100.0% | 7.60 | 8.18 |
| book_scoped_semantic | 7 | 57.1% | 100.0% | 0.750 | – | 100.0% | 7.22 | 7.67 |
| unanswerable | 6 | – | – | – | – | 100.0% | 6.91 | 7.28 |
| **OVERALL** | **72** | **84.8%** | **100.0%** | **0.901** | **100.0%** | **100.0%** | **6.78** | **7.70** |
| *Known gaps* | *5* | *0.0%* | *0.0%* | *0.000* | *83.3%* | *0.0%* | *5.85* | *6.13* |

How to read it:

- **Recall@5 is 100%, but Recall@1 is 84.8%.** The right chunk is always in the context, but it
  is often not first. The weakest first-rank categories are `outline` (40%), where scripture
  chunks outrank the outline, and `book_scoped_semantic` (57%).
- **Before the per-intent coverage step, `mixed_intent` Recall@5 was 66.7%.** M03 ("'beast' …
  Revelation 13:18") and M04 ("Outline of Romans and … Romans 8:28") lost their secondary match
  to several overlapping scripture windows. The retriever now keeps one exact match per detected
  intent in the top-k.
- **Retrieval latency is about 7 s per query on 4 cores**, almost all of it the L-12 cross-encoder
  scoring around 25 long candidates. `RESOURCE_PROFILE=low` switches to `ms-marco-TinyBERT-L-2-v2`;
  measure it on the target machine before relying on the 90 s budget.
- **This is an easier corpus than production.** The full collection has 66 books and every
  topic and headword. Run `--use-existing-collection` against an ingested Qdrant for the real number.
- Every known-gap case misses, as documented in [Known gaps](#5-known-gaps).

The deterministic default (`--embeddings hash`, reranker off) and `--legacy` remain the CI
checks; `--legacy` still scores 100% / 100% / 1.000 on its 15 queries.
