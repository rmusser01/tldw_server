# Claims extraction

This document describes the claims subsystem — read the split first, it's the most common confusion: **(a) ingestion-time claims** are small numbers of atomic propositions extracted per chunk during vectorization and persisted to the Media DB (`Claims` + `claims_fts`) for search/monitoring/review; **(b) answer-time claims** are the `ClaimsEngine` behind RAG `enable_claims`, extracting claims from a *generated answer* and verifying them, streamed as `claims_overlay`/`final_claims`. Plus the FVA falsification extension, clustering, review queues, alerts, and a Jobs domain.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Engine | `app/core/Claims_Extraction/claims_engine.py` | `ClaimsEngine`, heuristic/LLM extractors, the hybrid verifier, claim-type classification, verification-status rules |
| Ingestion-time | `ingestion_claims.py` | `extract_claims_for_chunks` / `store_claims` (sha256 chunk-hash rows, review-rule assignments) |
| Ingestion hook | `app/core/Embeddings/ChromaDB_Library.py` | behind `ENABLE_INGESTION_CLAIMS` — claims ride the vectorization pipeline (no vectorization ⇒ no claims) |
| Dispatch | `extractor_registry.py`, `extractor_catalog.py` | mode resolution (heuristic/ner/llm/auto; spaCy NER per language); `claimify` maps to the generic LLM JSON extractor (not a distinct implementation) |
| Parsing/alignment | `output_parser.py`, `alignment.py`/`span_alignment.py`, `runtime_config.py` | fenced-JSON strict/lenient parsing; span alignment (fuzzy 0.75 default); LLM config resolution (claims keys > RAG defaults > global) |
| FVA | `fva_pipeline.py` + `anti_context_retriever.py` + `adjudicator.py`, `budget_guard.py` | falsification trigger → counter-evidence retrieval → adjudication; cost-budget guarded |
| Service/API | `claims_service.py` + routes (prefix `/claims`) | search, notifications/digest, watchlist evaluation, settings, monitoring, alerts CRUD+evaluate, rebuild health, review queue/rules/analytics, extractor catalog, clusters (+timeline/evidence), `/verify/fva` |
| Storage | `app/core/DB_Management/media_db/{runtime/claims_write_ops,claims_search_ops,claims_fts_ops}.py` + schema extensions | Media-DB-owned tables; FTS5 MATCH + bm25 (SQLite) / ts_rank (Postgres) with LIKE fallback |
| Jobs | `claims_jobs.py`, `claims_job_contracts.py`, `claims_job_handlers.py` | domain `claims`, queue `default`; rebuild/alert-delivery/analytics-export/review-notification job types with per-type retry envs |
| RAG tie-in | `rag_service/unified_pipeline.py`, `rag_service/claims.py` (a lazy re-export shim), `endpoints/rag_unified.py` | the `enable_claims` gate, pre-claim verification of stored claims, streaming events |

## Ingestion-time (dataflow)

1. Trigger: vectorization with `ENABLE_INGESTION_CLAIMS` (or per-request tri-state overrides on `/media/add`).
2. `extract_claims_for_chunks`: mode resolution (auto → heuristic vs LLM by text/language); strategies: heuristic sentences, spaCy NER, or a strict-JSON LLM prompt (module `ingestion` prompt keys; lenient parse default; alignment modes; multi-pass; timeout 8 s with heuristic fallback on LLM failure).
3. `store_claims`: chunk-hash-keyed rows (default ≤3 per chunk), span coercion, review-rule pre-assignment (+ optional notification jobs), Media-DB upsert with FTS maintenance. Optional claim embeddings into a per-user Chroma collection.
4. Rebuilds: policy/interval/staleness-driven; a Jobs domain handles rebuild and the auxiliary job types.

## Answer-time (dataflow, RAG `enable_claims`)

1. Request keys: extractor (`aps|claimify|auto`), verifier (`nli|llm|hybrid`), top-k (5), confidence (0.7), max (25), concurrency (8). Skipped when the fast groundedness check passes ≥0.9 with `skip_full_claims`.
2. Provider resolution wraps the request's credential runtime; every claims LLM call is accounted (ADR-067).
3. Extraction by strategy (`aps` is the proposition-chunking LLM mode; `claimify` dispatches to the generic LLM JSON extractor); multi-pass dedupe by normalized text/overlapping spans; heuristic fallback.
4. **Pre-claims shortcut**: when retrieved docs carry media ids, stored ingestion-time claims load and verify directly — extraction skipped.
5. `HybridClaimVerifier.verify`: claim-type classification (numeric/date vs general) → evidence retrieval (≤3 snippets, numeric/date-containing docs boosted) → local NLI (`roberta-large-mnli` default; entailment/contradiction/neutral → supported/refuted/nei) with the LLM judge as fallback below threshold (hybrid).
6. Output: aggregate precision/coverage + per-claim payloads in `result.metadata.claims`; streaming emits `claims_overlay` then `final_claims`. The FVA path adds falsification → anti-context → adjudication under its budget guard.

## Boundaries

- The Media DB owns all persisted claims (both backends); answer-time results are ephemeral unless pre-claims verified stored rows.
- Watchlists integration: cluster subscriptions + claims-alert delivery into watchlist digests.
- Review workflows (rules/queue/bulk/analytics) are module features over Media-DB tables with RBAC gates.
- Do not confuse with ADR-052's "MCP admin claims" — a different sense of the word.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| LLM extractor throttle/budget exhausted | Fallback metrics + heuristic extraction (or empty under strict budget) |
| Malformed model JSON | Lenient parser recovers fenced/partial JSON; strict raises loudly |
| NLI model unavailable | Warn once; hybrid degrades to the LLM judge |
| Claim retrieval failure | Falls back to base documents |
| Rebuild job failure | Isolated at the worker boundary with sanitized codes |

## Config keys (selection; exact)

`ENABLE_INGESTION_CLAIMS`, `CLAIM_EXTRACTOR_MODE`, `CLAIMS_MAX_PER_CHUNK` (3), `CLAIMS_EMBED` + model id, the `CLAIMS_LLM_*` family, `CLAIMS_JSON_PARSE_MODE` (lenient), `CLAIMS_ALIGNMENT_MODE`/`_THRESHOLD`, `CLAIMS_CONTEXT_WINDOW_CHARS`, `CLAIMS_EXTRACTION_PASSES`, `CLAIMS_LLM_TIMEOUT_SEC` (8), `CLAIMS_ENABLE_LOCAL_NLI`, `RAG_NLI_MODEL[_PATH]`, the `CLAIMS_JOB_BUDGET_*` family, `CLAIMS_ADAPTIVE_THROTTLE_*`, `CLAIMS_JOBS_*` (enabled/worker/queue/per-type retries), the rebuild family (`_ENABLED/_INTERVAL_SEC/_POLICY/_STALE_DAYS`), monitoring/cluster/alert/review-metrics families. Note: `Env_Vars.md` lists only ~8 claims keys — the module docs carry the rest.

## Verified gotchas

1. `claimify` is schema-accepted but maps to the generic LLM JSON extractor; only `aps` and provider modes are special-cased.
2. `rag_service/claims.py` is a circular-import re-export shim — the engine lives in `Claims_Extraction/`.
3. Ingestion claims ride vectorization: no embeddings run ⇒ no claims even with the flag on.
4. Citation offsets/spans are best-effort and normalization-sensitive — keep span-alignment tests near parser changes.
5. The claims Jobs domain (`claims`/default queue) is distinct from the watchlists Scheduler queue.

## Governing decisions and docs

No claims-specific ADR; related: ADR-053 (RAG fusion context), ADR-067 (accounted LLM calls). Freshness: the module README and `Answer_Time_Claims_Engine.md` are current; `Claims_Extraction.md` is **partly stale** (7 months behind the module's clusters/review/notifications/FVA/analytics breadth); `Claims_API_and_Schema.md` mostly current.

## Related docs

- [rag.md](./rag.md) — the `enable_claims` gate and streaming events
- [watchlists.md](./watchlists.md) — cluster subscriptions and alert digests
- [chunking-embeddings.md](./chunking-embeddings.md) — the pipeline ingestion claims ride
