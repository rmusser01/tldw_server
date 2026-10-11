# Backend Performance Remediation — Batch 6: Ingestion Throughput Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13518 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Fix the ingestion paths that are quadratic or multiplied: five `+=` string builders, the double MediaWiki dump pass, five DOM parses per HTML upload, per-page ChromaDB manager construction, sequential per-chunk LLM contextualization, and unbatched embedding HTTP calls.

**Architecture:** Pure throughput work — outputs must be byte-identical where the builder changes (golden-fixture tests). No new dependencies (no `lxml` unless already installed — check before considering).

**Tech Stack:** list-append + `"".join`, `mwxml` streaming, `asyncio.Semaphore` + `run_in_executor`, OpenAI array-input embeddings, existing `_batched` helper (`ChromaDB_Library.py:486`).

## Global constraints

- Backend only; activate venv; one commit per stage referencing TASK-13518.
- **Byte-identical outputs** for every Stage 3 builder change — golden fixtures recorded from pre-change code first.
- External-service behavior (LLM providers, embedding APIs) mocked in tests; no network in CI.
- Bandit: `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing tldw_Server_API/app/core/Web_Scraping tldw_Server_API/app/core/Embeddings tldw_Server_API/app/core/Utils/Utils.py tldw_Server_API/app/core/LLM_Calls/chat_calls.py -f json -o /tmp/bandit_perf_b6.json`.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: MediaWiki single pass + hoisted vector-store writer

**Goal:** Stop decompressing/parsing dumps (up to 10GB) twice; stop building a `ChromaDBManager` per page.

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py` (`count_pages` pass ~1001-1007; parse loop ~1117-1144; `_store_mediawiki_chunks_in_vector_db` ~642-665 called per page from `process_single_item` ~810-821)
- Test: `tldw_Server_API/tests/MediaWiki/test_single_pass_import.py` (new)

**Change:**
1. Delete the pre-count pass; count pages during the processing loop (progress callback starts indeterminate, updates total as it streams). If the progress UI contract *requires* a pre-total, keep an optional count pass behind a flag default-off and note it.
2. Hoist one `ChromaDBManager` above the page loop (pattern already used for `shared_media_writer` ~1014-1019); buffer chunk texts and flush to embedding+Chroma every 256 chunks (reuse `_batched`), flushing the remainder at the end.

**Tests:**
- [ ] `test_import_parses_dump_once` — mock decompress/parse entry points with counters over a small fixture dump → each == 1.
- [ ] `test_manager_constructed_once` — multi-page fixture → `ChromaDBManager.__init__` count == 1 (mock).
- [ ] `test_chunk_flush_batching` — 600 chunks → embedding calls ≤ 3, order-independent content equality of stored chunks vs per-page reference.

**Status:** Not Started

## Stage 2: HTML single-parse reuse

**Goal:** One HTML upload = one DOM build (currently up to five).

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Plaintext/Plaintext_Files.py` (double parse ~240 and ~257)
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py` (sanitize triple-parse ~1095-1149; rewrite ~1237-1243)
- Test: `tldw_Server_API/tests/Ingestion/test_html_single_parse.py` (new)

**Change:**
1. `Plaintext_Files`: read `<title>`/`<meta name=author>` from the **first** soup before decomposing script/style; drop the second `BeautifulSoup(html_content, ...)`.
2. `Upload_Sink.sanitize_html_content`: perform cleanup on the first soup; extract title/author anchors from it; serialize once; only re-parse if bleach's cleaner must operate on a string (then reuse its output tree — do not build a third).
3. Forward the sanitized text downstream (avoid the disk round-trip write+read when the caller is in-process).

**Tests:**
- [ ] `test_html_parsed_once` — counter on `BeautifulSoup.__init__` == 1 for the full convert path.
- [ ] `test_title_author_extraction_unchanged` — golden outputs for 3 fixture HTML files.
- [ ] `test_sanitized_output_identical` — byte-compare sanitized HTML pre/post on fixtures.

**Status:** Not Started

## Stage 3: Quadratic string builders → list + join (5 sites)

**Goal:** Remove O(n²) accumulation over whole documents.

**Files + golden fixtures (one sub-commit per site):**
1. `Ingestion_Media_Processing/PDF/PDF_Processing_Lib.py` (~317-364): per-page `page_parts` list; `markdown_text = "".join(parts)` at end.
2. `Ingestion_Media_Processing/Books/Book_Processing_Lib.py` `epub_to_markdown` (~205-246): collect chapter fragments in a list.
3. `Books/Book_Processing_Lib.py` `xml_to_markdown` (~514-540): pass an accumulator list through the recursion; join once at top. (Leave the full-DOM `ET.parse` as-is this batch; streaming is out of scope.)
4. `Plaintext/Plaintext_Files.py` `_xml_to_text_simple` (~135-143): same accumulator treatment.
5. `Web_Scraping/Article_Extractor_Lib.py` `convert_to_markdown` (~929-937): parts list + join.

**Tests (per site):**
- [ ] Golden byte-equality test — generate output from a fixture input with the *old* implementation captured into the test (or a checked-in golden), assert the new implementation matches exactly.
- [ ] `test_large_input_linear_time` — synthetic ~2MB input completes under a generous fixed budget (regression tripwire, not a benchmark): e.g., 200k-char document < 5s.

**Status:** Not Started

## Stage 4: `safe_read_file` — bounded detection, sampled printability

**Goal:** The generic fallback reader stops doing full-file chardet + up to 7 full decodes + per-character Python loops.

**Files:**
- Modify: `tldw_Server_API/app/core/Utils/Utils.py` (`safe_read_file` ~534-574; `is_valid_url` per-call `re.compile` ~627 — hoist to module-level compiled pattern while here)
- Test: `tldw_Server_API/tests/Utils/test_safe_read_file.py` (new)

**Change:** try `utf-8` decode first (return immediately on success); `chardet.detect` on the first 64KB only; printability ratio sampled on the first 8KB (regex `[^\\s\\P{C}]`-style count or `str.isprintable` on a slice); keep the encoding-candidate order and final fallback semantics; preserve behavior for the utf-8-success path (must remain byte-identical).

**Tests:**
- [ ] `test_utf8_short_circuit` — utf-8 file → 1 decode, 0 chardet scans (mock counters).
- [ ] `test_detection_capped` — 5MB non-utf8 fixture → chardet input ≤ 64KB, decode attempts bounded.
- [ ] `test_outputs_match_reference` — fixture corpus (utf-8, latin-1, utf-16) decoded identically to the old implementation.

**Status:** Not Started

## Stage 5: Contextual chunking — bounded concurrency

**Goal:** The per-chunk `situate_context` LLM loop stops being strictly sequential (500 chunks ≈ 8 minutes today).

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py` (~1015-1064 loop; `situate_context` ~652-670)
- Test: `tldw_Server_API/tests/Embeddings/test_contextual_chunking_concurrency.py` (new)

**Change:** fan out `situate_context` calls with a thread-pool executor + `asyncio.Semaphore` (or `Semaphore`-guarded `run_in_executor` since the call is sync) — concurrency default 8, config knob `EMBEDDINGS_CONTEXTUAL_CONCURRENCY` (respect existing provider rate limits; the semaphore guards per-provider storming); **results reassembled in original chunk order**; per-call failures keep current per-chunk fallback semantics.

**Tests:**
- [ ] `test_order_preserved_under_concurrency` — mocked LLM with random per-call latency; output chunk order == input order.
- [ ] `test_concurrency_capped` — mocked LLM tracking in-flight count; max observed ≤ 8.
- [ ] `test_failure_semantics_unchanged` — one failing call → same fallback content as sequential reference.

**Status:** Not Started

## Stage 6: Embedding request batching (async + sync paths)

**Goal:** Async "batch" stops being n HTTP requests; sync path stops failing hard at provider array limits; both reuse pooled sessions.

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/async_embeddings.py` (~233-245, payload at ~338-341)
- Modify: `tldw_Server_API/app/core/LLM_Calls/chat_calls.py` (~303-323 embedding path)
- Modify: `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py` (~1576-1583 HF path; ~2231-2242 local/TODO site)
- Test: `tldw_Server_API/tests/Embeddings/test_embedding_request_batching.py` (new)

**Change:**
1. Async: group input texts into ≤100-item array-input requests (`{"input": [...], "model": ...}`); single rate-limiter charge per request group where the limiter keys on requests; preserve result ordering (index-mapped).
2. Sync OpenAI: split input at min(provider array cap, 2048 items / token budget ~300k) using the `_batched` helper; concatenate results in order.
3. HF/local: mini-batch the forward pass (64) to bound GPU/memory.
4. Module-level `requests.Session` with the existing retry helper (replace per-call `create_session_with_retries(total=1)`); document thread-safety (requests.Session is not strictly thread-safe — if callers are threaded, keep a small session pool keyed by thread id or guard with a lock).

**Tests:**
- [ ] `test_async_batch_groups_requests` — 250 texts, mocked client → 3 array-input calls, order preserved.
- [ ] `test_sync_splits_at_provider_cap` — 5,000 texts → ceil(5000/2048) calls, concatenated order correct.
- [ ] `test_hf_path_mini_batched` — forward-call batch sizes ≤ 64.
- [ ] `test_session_reused` — mocked transport; session creations == 1 across N calls.

**Status:** Not Started

## Stage 7: Scraping-path efficiency (sitemap concurrency + cluster extractor single DOM)

**Goal:** Sitemap scrapes stop fetching URLs strictly sequentially; the cluster extraction strategy stops building a second DOM per page and scanning text once per keyword.

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py` (sitemap loops ~711-724 and ~530: reuse the bounded-concurrency worker pattern from `enhanced_web_scraping.scrape_multiple` ~1933-1950 — `asyncio.gather` + semaphore, default concurrency 4; stream large sitemaps with `iterparse`-style parsing instead of full `fromstring` DOM where the file can be multi-MB; drop the `minidom.parseString(...).toprettyxml()` pretty-print at ~864 — return the parsed structure directly)
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/cluster.py` (second DOM for title ~427 — return the title from the first soup built at ~148; `_tag_cluster_text` keyword scoring ~168-176 and ~294 — one tokenized pass with a `Counter` instead of `text.count(keyword)` per keyword)
- Test: `tldw_Server_API/tests/Web_Scraping/test_sitemap_and_cluster_efficiency.py` (new)

**Tests:**
- [ ] `test_sitemap_scrape_bounded_concurrency` — 20 mocked URLs at 50ms each → wall time < 400ms at concurrency 4; max in-flight ≤ 4.
- [ ] `test_cluster_strategy_single_dom` — `BeautifulSoup.__init__` counter == 1 per article.
- [ ] `test_tag_scores_identical` — Counter-based scores equal per-keyword `count()` reference on fixture text.

**Status:** Not Started

---

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MediaWiki tldw_Server_API/tests/Ingestion tldw_Server_API/tests/Embeddings tldw_Server_API/tests/Utils/test_safe_read_file.py -v -m "not external_api and not local_llm_service"
# golden fixtures regenerate check (should be no-ops after changes):
python -m pytest tldw_Server_API/tests -k "golden or byte_identical or single_parse" -v
python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing tldw_Server_API/app/core/Web_Scraping tldw_Server_API/app/core/Embeddings tldw_Server_API/app/core/Utils/Utils.py tldw_Server_API/app/core/LLM_Calls/chat_calls.py -f json -o /tmp/bandit_perf_b6.json
```

Record ingestion benchmarks (a synthetic 100-page wiki fixture + 2MB PDF fixture timing, added to the Batch-0 bench set as `bench_ingest_fixture.py`) in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13518 (notes, touched files, verification, final summary, DOD).
