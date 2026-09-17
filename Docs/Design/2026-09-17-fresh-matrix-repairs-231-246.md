# Repairs from the September 17 fresh-install matrix

Parent: TASK13260. User scope: fix every identified issue before another full UAT. The frozen matrix ran SQLite/PostgreSQL × single/multi-user on `8f8774e6c868b304a96d95ab82e28389c129a78b`. All48 outcomes are recorded; sixteen new findings remain open. This document defines bounded corrections to existing flows, not new features.

## Decisions

Use the canonical schema, authorization context, provider identity, query refresh, scheduling result and error handling paths already present. Request-time schema creation, disabled RLS, elevated runtime roles, fail-open quotas, artificial source insertion and substituted Wikipedia content are excluded: they would evade the actual acceptance requirements. A successful unit test alone does not close a native finding.

| Finding / Backlog child | Correction and original-scenario proof |
|---|---|
|231 / .173|Make unconfigured-client console guidance match WebUI/extension surface; fresh setup console no longer names the wrong surface.|
|232 / .174|Preserve sanitized actionable model-unavailable detail in Chat error presentation; native real400 and Retry retain original model, context and one user turn.|
|233 / .175|Deduplicate identical terminal ingestion warnings while retaining distinct warnings and failure state; actual job readback matches one occurrence per warning.|
|234 / .176|Set a finite supported Next rewrite timeout consistent with existing long generation budgets, preserving client abort and upstream errors; actual >30s proxy control and exact Biology five-card workflow must succeed.|
|235 / .177|Use the successful review response in the re-rate snapshot; Due and scheduled Cram display the interval the next rating actually saves. Preserve practice-only and authority guards.|
|236 / .178|Normalize supported llama provider aliases and qualified IDs consistently across readiness/selection. Preserve wrong-provider conflicts and unconfigured rejection; both single-user recovery routes enter TestBot and persist its exact visible answer.|
|237 / .179|Gate protected Media work on usable credentials and fence late callbacks to the current authority/lifetime. Preserve real outage/unexpected-error feedback; disconnected entry must not emit redundant search errors.|
|238 / .180|Bind the trusted queued owner to content authorization and carry it through the actual persistence executor. Real restricted-role PostgreSQL writes must succeed only for the correct owner, with cleanup after success/error/cancellation.|
|239 / .181|Use the supported connection lifecycle for world-book catalogue reads; actual PostgreSQL Character editor catalogue and populated records load without500.|
|240 / .182|Compute lapse/retention from the stored scheduler outcome, with explicit legacy-row behavior. Hard recall is successful; true lapse remains a lapse on both database backends. Review-session correctness must remain semantically consistent.|
|241 / .183|Do not offer a full-content Media handoff until that selected source is available; actual delayed detail selection must carry complete content, with no old-selection leakage.|
|242 / .184|Apply existing singular/plural localization to Due completion without changing counts; one and multiple completed reviews render correctly.|
|243 / .185|Reproduce the real new ordinary conversation transition after account switch and make its displayed mode match the accepted ordinary conversation. Retain intentional empty Character preference and private-state clearing.|
|244 / .186|Reconnect actual wizard completion with the existing owned Media refresh contract, including same-route and before-mount completion. Preserve filters/pagination and reject stale-owner callbacks.|
|245 / .187|Add idempotent storage-quota PostgreSQL DDL to normal AuthNZ bootstrap with org/team partial unique indexes and existing grants. Actual fresh/repeated bootstrap and restricted quota CRUD/admission must pass without weakening fail-closed behavior.|
|246 / .188|Diagnose the observed45s TestBot stream failure across actual transport boundaries before changing behavior. Upstream200 is not completion. A cause-supported repair or proven external constraint, plus the original exact native completion/reload, is required for disposition.|

## Implementation boundaries

- PostgreSQL schema: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`, storage-quota repository/guard integration tests. Keep quota admission and content-worker238 separate.
- Ingest worker/persistence: `app/services/media_ingest_jobs_worker.py`, `app/core/Ingestion_Media_Processing/persistence.py` and existing worker/DB integration tests. Keep generic client labels separate from authorization.
- World books: `app/core/Character_Chat/world_book_manager.py` catalogue method and real DB controls.
- Study: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`, review helpers/localization and `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` analytics. Preserve scheduler rules.
- Media UI: `ViewMediaPage`, `ContentViewer`, `useMediaSearch`, route readiness and Quick Ingest wizard completion; one author owns shared paths.
- Chat/model/error: existing provider normalization, selection/readiness, ordinary conversation transition and stream transport paths; separate diagnostic246 from proven232/236/243 changes.
- Proxy: `apps/tldw-frontend/next.config.mjs` and existing quickstart proxy test harness, using installed Next rather than a duplicate timer implementation.

The source audits in `.tmp/uat-next-matrix-20260916/audits` and the retained matrix packets contain exact source/test hashes and original observations. Implementers read their complete audit before writing a causal regression. No production changes are made inside the frozen source archives.

## Verification and release gate

### Study240 metric compatibility

Daily lapse/retention trusts the persisted `was_lapse` Boolean, including historical rows. The canonical schema has always made this field non-null with a false default; there is no reliable way to distinguish an old omitted/defaulted value from a deliberately recorded non-lapse. Do not invent a rating-based fallback or rewrite history. Historical rows explicitly recording a lapse retain it. Review-session correct recall also excludes Again0: an unsuccessful learning attempt is not yet a lapse of learned material. For current schedulers, other accepted ratings count as recall unless the recorded outcome says lapse; this includes Hard2 and preserves the actual scheduler behavior of API-only ratings1/4. Live increments and reconstruction must agree. Older tests that used rating1 as a presumed failure must use an actual Again0 or explicitly seed a historical lapse outcome.

Verify both schedulers, new/review queues, all accepted ratings, repeated scheduled reviews, stored historical outcomes, session reconstruction, and existing date/deck/workspace/owner and caller-transaction boundaries. This is a metric correction; scheduler transitions remain unchanged.

### Follow-up247: tenant-scoped Media sequence maintenance

TASK13260.189 was created before edits after the actual238 worker test proved owner1 insertion followed by owner2 SQLSTATE23505. Both scoped INSERTs use non-superuser/non-BYPASSRLS roles and admin0. Normal Media initialization rewinds its shared serial to1 from a tenant-filtered MAX; prior147 correctly excludes foreign module sequences but does not prevent this own-table rewind. Preserve allocated sequence high-water marks, explicit-ID migration/import repair, empty-table behavior and unrelated sequences. Diagnose concurrent allocation as well as sequential tenant initialization before selecting a minimal correction; a read/modify/setval race must not silently reintroduce rewind. Do not disable RLS or introduce elevated content writes. Required official PostgreSQL tests and original two-owner queued-upload acceptance apply.

Each unit needs a demonstrated causal RED, minimal GREEN, relevant adjacent controls and independent review. Required PostgreSQL tests use official fixtures and run without skips; never provision a replacement cluster. Run scoped lint/format checks and Bandit for changed Python using `.venv`. Compare the full frontend compiler against the recorded90 existing signatures and investigate new failures.

Retain sanitized evidence with hashes and per-issue repair revision, test result, native original-scenario acceptance and limits. Keep SQLite/PG profiles and official holders for targeted verification; any updated runtime is explicitly a repair profile, not part of the frozen matrix. Use serial real-model inference. Begin a new full fresh48-row run only after all findings, including246, have a reviewed and evidence-backed disposition. External source denial, image capability limits and reused dependency installation remain explicit.

## Follow-up248: retire Character work at the actual account boundary

Independent review proves stale persistence dispatch after actual logout invalidates the captured service-prompt lease; a cross-account database write is not established. Reuse the existing snapshot request scope and effective lifetime in useChatActions Character branches, pass existing requestScopeFields through the client domain stream method, and check ownership around awaited create/save/recovery boundaries. Preserve deliberate caller cancellation versus account invalidation, acknowledged IDs and retry/no-replay behavior. Add actual-auth RED/GREEN and same-owner/partial recovery controls. No new authority framework or timeout-policy change. TASK13260.190 and retained independent finding own this separate repair.

The scoped Character transport additionally requires the existing service-prompt request policy to allow POST to the exact `/api/v1/chats/{id}/complete-v2` path. The real captured-scope adapter regression rejects before fetch without this prerequisite. Limit the third production edit to that route/method alternative and verify wrong methods, sibling paths and traversal remain rejected.

## Follow-up251 and252: MCP setup and malformed-schema fixture

The actual fresh PostgreSQL Save packs request fails in permission-profile lookup because nullable query parameters lack types. Restrict251 to five setup-chain repository methods: profile listing/create readback and assignment listing/get/create readback. Cast nullable filters to their schema types while preserving list-None-as-unfiltered and exact-NULL readback semantics; use Boolean FALSE in organization projections. No shared translator, schema or authorization change. Required real PostgreSQL/SQLite regressions cover positive, null, scoped, owner and caller-transaction controls; repeat native Save packs after review.

252 is a separate baseline test-fixture failure: destructive setup of a malformed temporary SQLite schema goes through the guarded runtime pool. Match established corruption fixtures using an explicitly closed fixture-only connection; retain the missing-table assertion and production guard negatives. A required-PostgreSQL runner does not make that SQLite fixture a PostgreSQL test.

##246: prevent compression buffering of Character SSE

Real native PostgreSQL single/multi TestBot answers now complete but body bytes arrive in late bursts about40seconds after headers. Multi response has gzip and no-cache. Installed Next compression applies to external rewrites, and ordinary writes do not flush zlib. Establish a causal regression that negotiates gzip, writes a small first frame and holds terminal output: first reader data must arrive while the terminal gate remains held. Keep no-transform-absent gzip as an adverse control and identity as a positive control, with deterministic cleanup. Existing early-header/whole-body tests do not establish timely frame delivery.

Use one local `Cache-Control:no-cache,no-transform` plus `X-Accel-Buffering:no` header mapping for all three Character SSE return branches, including legacy and text fallback. Actual endpoint tests must assert returned headers across both wrapper modes and real PostgreSQL/SQLite. Keep global compression, finite timeouts, inference and authorization unchanged. Repeat native exact TestBot/canonical reload with passive body timing on the reviewed revision; do not retroactively attribute the original timeout solely from compression presence.

## Follow-up253: portable MediaFiles parameter binding

Actual native PostgreSQL Media detail fails after successful queued ingestion because the repository sends SQLite `:name` placeholders unchanged to PostgreSQL. Use the existing portable positional `?`/tuple contract within `MediaFilesRepository`; do not broaden the shared SQL translator. Test its actual rich detail and affected lifecycle calls on official PostgreSQL and SQLite, including latest-original ordering, include-deleted behavior, soft/hard deletion, retained references, ownership and caller transactions. Bind existing Boolean columns with Boolean values where required. Preserve errors as errors rather than converting failed lookups into absent files. Original Media1 and its data remain for native acceptance after independent review.
