# Backend Performance Remediation — Coordination Index (2026-10-06)

Umbrella index for the staged execution plans drafted from the 2026-10-06 backend
efficiency review. The review audited six areas (DB layer, API endpoints, RAG/embeddings,
chat/LLM, ingestion, services/scheduler/auth) and confirmed ~45 findings across five
systemic patterns:

1. Expensive resources rebuilt per request instead of cached (PBKDF2, reranker models, Jinja templates, DB DDL).
2. N+1 loops sitting next to batch helpers that already exist in the same module.
3. Synchronous SQLite/model work executing directly on the event loop (workers, exports, destructive endpoints).
4. Python doing what FTS5/SQL indexes already do (LIKE scans, Python GROUP BY, full-table loads).
5. Genuine O(n²) algorithms (string `+=` accumulation, pairwise dedup, signature-slice scans).

Scope: **backend only** (`tldw_Server_API/**`). `apps/**` (WebUI) is explicitly out of
scope — a separate workstream owns it:
[WebUI/extension perf remediation index](2026-10-06-webui-perf-remediation-coordination-index.md)
(TASK-13520–13525). Its **[BE]**-marked stages request endpoints from this program's
Batch 3 (TASK-13515). This mirrors the collision rules of the
[credit workstreams index](2026-10-01-credit-workstreams-coordination-index.md).

## Batch → plan → task map

| Batch | Plan | Scope flavor | Size | Backlog task |
|-------|------|--------------|------|--------------|
| 0 | [Baseline harness](2026-10-06-perf-batch-0-baseline-harness-implementation-plan.md) | Measurement infrastructure | 2-3 days | TASK-13512 |
| 1 | [Per-request fixed costs](2026-10-06-perf-batch-1-request-fixed-costs-implementation-plan.md) | Auth, ChaCha deps, MCP tools/list, config | ~3 days | TASK-13513 |
| 2 | [Chat & character per-turn path](2026-10-06-perf-batch-2-chat-turn-path-implementation-plan.md) | History assembly, templates, world books, streaming | ~1 week | TASK-13514 |
| 3 | [N+1 batching](2026-10-06-perf-batch-3-n-plus-one-batching-implementation-plan.md) | DB + endpoint layer | ~1 week | TASK-13515 |
| 4 | [SQL/FTS pushdown & indexes](2026-10-06-perf-batch-4-sql-fts-pushdown-implementation-plan.md) | Search paths, aggregation, schema index | ~1 week | TASK-13516 |
| 5 | [Event loop, workers, schedulers, MCP memory](2026-10-06-perf-batch-5-event-loop-workers-implementation-plan.md) | Background efficiency + memory hygiene | ~1 week | TASK-13517 |
| 6 | [Ingestion throughput](2026-10-06-perf-batch-6-ingestion-throughput-implementation-plan.md) | MediaWiki, parsers, embeddings batching | ~1 week | TASK-13518 |
| 7 | [RAG pipeline depth](2026-10-06-perf-batch-7-rag-pipeline-implementation-plan.md) | Reranker lifecycle, parallelism, dedup | ~1 week | TASK-13519 |

## Recommended sequencing

1. **Batch 0 first** — the query-counting helper and benchmark scripts are the
   verification mechanism for every later "before/after" claim.
2. **Batches 1 and 2 in parallel** (no file overlap: Batch 1 = AuthNZ/API_Deps/MCP/config;
   Batch 2 = Chat/Character_Chat). These remove the largest per-request fixed costs.
3. **Batch 7 Stage 1 (reranker singleton) can be pulled forward independently** at any
   time — it is the single biggest per-search win and touches only
   `advanced_reranking.py` + `unified_pipeline.py`.
4. **Batches 3 then 4, sequentially** — both touch `Prompts_DB.py` and
   `ChaChaNotes_DB.py`; running them in parallel risks merge conflicts on the same files.
5. **Batches 5 and 6 anytime, in parallel with 3/4** — disjoint files
   (Jobs/services/MCP vs Ingestion/Embeddings).
6. **Coordinate Batch 4 Stage 7 (schema index) with TASK-13403** (ChaChaNotes
   SQLite/Postgres schema parity v68 vs v72) — same migration machinery, avoid
   conflicting version bumps.

### Minimum viable slice

If only a subset ships, in order of leverage:
Batch 0 → Batch 1 Stage 1 (PBKDF2 cache) → Batch 7 Stage 1 (reranker singleton) →
Batch 2 Stages 1-2 (metadata map + Jinja cache) → Batch 4 Stage 1 (FTS retriever) →
Batch 5 Stage 1 (worker tick off-loop).

## Collision rules

- Never modify `apps/**` (WebUI workstream owns it).
- Backend edits must avoid files currently modified in the working tree — run
  `git status` at each stage start and re-check the file list in the plan.
- One Backlog task per batch already exists (TASK-13512-13419); do not create
  additional tasks per stage unless a stage grows beyond its batch — prefer
  appending implementation notes to the batch task.
- Batches 3/4 share `Prompts_DB.py`, `ChaChaNotes_DB.py`, `endpoints/chat.py` — run
  sequentially or split by stage.
- Behavior-preserving unless a stage explicitly says otherwise. No API contract changes.
  Schema change only in Batch 4 Stage 7 (index addition + migration bump, both backends).
  Intentional behavior changes (documented in the batch plans and task notes):
  Batch 5 empty-poll backoff + SLO cadence; Batch 6 late-chunk doc cap (Batch 7 Stage 6)
  and sitemap concurrency; Batch 7 rerank candidate cap.

## Program-level Definition of Done

Per batch (enforced in each plan file):

- [ ] All stages complete; each stage landed with its own commit referencing the task ID.
- [ ] Named tests written first and passing; no existing tests disabled.
- [ ] Query-count benchmarks (where applicable) re-run; delta recorded in
      `Docs/Reviews/PERF_BASELINE_2026_10.md`.
- [ ] Bandit run on touched paths; new findings fixed, not deferred.
- [ ] Backlog task updated: notes, touched files, verification results, final summary.

## Findings → batch coverage

| Review tier / area | Batch |
|--------------------|-------|
| Tier 1 per-request costs (PBKDF2, ChaCha dep, MCP RBAC, config) | 1 |
| Tier 1 chat-path costs (metadata N+1, Jinja, default character) | 2 |
| Tier 2 O(n²): overlap-trim, tool-args `+=` | 2 |
| Tier 2 O(n²): dedup banding, SequenceMatcher, MMR | 7 |
| Tier 2 O(n²): string builders (PDF/EPUB/XML/articles) | 6 |
| Tier 3 N+1 / fetch-all (DB + endpoints, clustering projection) | 3 |
| Tier 3 SQL/FTS pushdown, missing index, LIKE scans | 4 |
| Tier 4 event-loop blocking, polling, schedulers, destructive endpoints | 5 |
| Tier 4/5 MCP metrics cardinality, JWT growth | 5 |
| Tier 5 ingestion (MediaWiki double pass, parse reuse, embedding batching, sitemap/cluster) | 6 |
| Tier 5 RAG depth (reranker, expansion, LRU, semantic cache, late chunking) | 7 |

Dead-code findings from the review (e.g. `chat_orchestrator.approximate_token_count`,
`save_chat_history_to_db_wrapper` full-rewrite path with no production callers) are
**not** scheduled — propose deleting them in a future cleanup batch rather than
optimizing them here.

Deliberately deferred as minor (bounded N, low frequency — revisit only if profiles
implicate them): `endpoints/chat.py:6822` per-tag keyword queries in
`save_chat_knowledge`; `Kanban_DB.get_card_with_details` per-checklist-item queries;
`Chat/chat_dictionary.py` legacy chatdict regex rebuilds.

## Machine-readable baseline (filled by Batch 0)

See `Docs/Reviews/PERF_BASELINE_2026_10.md` after Batch 0 lands. Every batch's
benchmark assertions reference that file's baseline numbers.
