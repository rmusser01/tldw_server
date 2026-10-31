# PRD: Email Ingestion and Search Modernization (MsgVault-Inspired)

- Title: Email Ingestion and Search Modernization
- Owner: Project owner / maintainer (single owner)
- Status: Draft
- Target Version: v0.2.x
- Last Updated: 2026-09-26

## Release Scope and Evidence (2026-09-26)

Core email functionality is file-based ingestion, persistence, operator/text search,
and message retrieval. Gmail synchronization is an **optional connector** with a
separate validation and enablement gate. Pending live Gmail validation does not
block the core gate. One owner handles implementation, operations and release
approval; no separate backend/SRE/product approvals are required.

Development and validation use synthetic email only. Do not connect the owner's
personal Gmail or send personal email to an LLM. Upload processing is not assumed
to be model-free: `/media/add` defaults `perform_analysis` to true, embeddings and
LLM-assisted chunking have separate switches, and claims extraction can inherit
server settings. The tested offline options and intercepted call boundaries are
recorded in `Docs/Operations/Email_Core_Validation_2026-09-13.md` (TASK-13250).

The milestone checkmarks below are implementation records, not deployment
certification. The 2026-09-13 follow-up fixes the identical-body identity collision
(TASK-13251), preserves parsed email metadata, and adds opt-in cursor pagination
(TASK-13253) while retaining offset clients. Synthetic EML, ZIP and MBOX regressions
now require separate identities. See the follow-up section of the validation record
for fresh results and remaining environment gates.

TASK-13255 additionally validates real API-key authentication and per-user SQLite
isolation, including main-app test-mode request middleware and route registration. See
`Docs/Operations/Email_Authenticated_Validation_2026-09-13.md`. TASK-13256 adds
authenticated synthetic EML upload, API-key scope/role rejection, organization
storage-quota behavior and matching search tenancy in local ASGI/SQLite; see
`Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md`.
Subsequent guarded loopback HTTP probes validate full-app startup/shutdown,
API-key uploads, scoped search/detail/media search and PostgreSQL forced-RLS
isolation. Their reports are `Docs/Operations/Email_Live_SQLite_Validation_2026-09-25.md`
and `Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md`. JWT login is
outside those probes.

TASK-13376 closes the remaining core implementation and validation work against a
local reference deployment: one loopback Uvicorn worker, SQLite first and then
PostgreSQL. Historical 10,000-message search runs and three 100-message archive
batches establish bounded correctness and measurements; they do not certify the
1,000,000-message search target or sustained ingestion throughput. Final scale,
parity, flag-reload and rollback evidence is recorded separately in
`Docs/Operations/Email_Release_Checklist_and_Rollback.md`; all core technical gates
are now verified with retained final artifacts. This local scope does not
claim validation of another deployment topology or a process-restart rollout.

Stage 1 implementation is committed as `b880be1b53`: attachment policy, bounded
metrics and sensitive logging. The combined core suite passed 328 tests with two
optional real-PST fixture skips; the focused logging/pool/access-log suite passed
83 tests. Additional legacy-search diagnostics have six sentinel paths and 24
selected regressions. SQLite's complete million-message certificate passes at
warm aggregate p50 202.23 ms / p95 725.74 ms; sustained HTTP ingestion passes at
130.54 messages/sec with complete parity, rollback and live metrics. Its generated
temporary roots are confirmed absent. Final PostgreSQL certificates at
`880d690a93` pass search at p50 220.92 ms / p95 495.49 ms and sustained HTTP at
61.71 messages/sec over 61.5734 request seconds, with exact parity, retry, RLS,
live metrics and local rollback/restore checks. PostgreSQL's batch diagnostic is
false (3 of 38 below 50; minimum 38.96); sustained acceptance uses the minute-wide
aggregate. All probe-owned cleanup is verified. The evidence record is
`Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`.

PostgreSQL query/archive tuning is committed as `aac02e68e8`, preserving substring
semantics and forced-RLS isolation. The selected per-connection profile uses
`PGOPTIONS='-c work_mem=64 MB'`, expression statistics and a native multicolumn
trigram index, with no global service configuration change. Its short diagnostic
comparison and bounded archive profiles do not certify final PostgreSQL search
or sustained throughput; the full attempt aborted during fixture parity before
timing. Commit `1fee95629a` addresses that local Docker shared-memory limit with
serial verification within its transaction, preserving complete parity/security
checks and restoring two workers / 64 MB before search timing. The closeout record
retains the failed initial run and maintenance provenance. The next full run
passed parity but failed repeated label COUNT; `455fad64fb` adds a
transaction-local custom-plan correction that preserves parameters, predicates
and RLS and restores session settings.
The generic-plan transition is inferred from the prepared-execution boundary.
Its bounded pooled comparison and 22 passing regressions do not close gates
alone. The earlier complete PostgreSQL certificate at frozen `455fad64fb` passed with
all ten populated classes, meaningful negation, unchanged 21 source hashes,
complete fixture parity, forced RLS and zero guard attempts. The fresh HTTP
attempt then failed throughput and legacy/native parity (0 versus 1,700).
Legacy parameter/phrase/rank fixes, bounded SQL rewrite/placeholder caches and
identity round-trip reductions are committed as `ae46f643ab`. The full HTTP
rerun passes with 5,600 exact legacy/native IDs, quoted title/body parity 1/1,
`auto_email`, rollback/restore, zero guards and 22 stable source hashes. Final
combined regressions are 211 passed. The later fresh shared-service run failed
with POSIX shared-memory SQLSTATE 53100 at 64 MiB; that diagnostic and earlier
455/ae46 results are historical. Final probe configuration `880d690a93` uses a
dedicated loopback 5435 PostgreSQL 18.6 service with 256 MiB shared memory, the same
image digest and 64 MB connection budget, preserving shared 5434 and production
queries/RLS. Fresh search and HTTP both pass with 23 stable hashes. Final quality
is Ruff/Bandit clean, 62 compiled files/38 production-probe paths and 91 helper
regressions; all 55 exact roots and five scoped database/role/manifest targets,
plus only the owned container/volume, are cleaned. The 27.87 failed HTTP and
bounded profiling results remain historical; single-owner approval is unrecorded.

Optional live Gmail, staging sync lag and actual OST files remain unverified.
TASK-13377 executes both native PST endpoint tests with a pinned public synthetic
container; see `Docs/Operations/Email_Real_PST_Validation_2026-09-26.md`. Synthetic
adapter tests and deterministic missing-parser errors are separate evidence. None of the optional live Gmail checks blocks core release.

## Summary

`tldw_Server_API` already parses email content and metadata well, but persistence and search are still media-generic. This PRD defines an email-native ingestion and search capability inspired by `msgvault`: first-class message identity, normalized participants/labels, Gmail-style search operators, and incremental mailbox sync. The result should materially improve retrieval quality, dedupe correctness, and ingestion reliability for both uploaded archives and connected inboxes.

## Problem Statement

Current email handling has three high-impact limitations:

1. Rich parsed metadata is not fully persisted, which reduces filter/search fidelity.
2. Dedupe behavior can collapse distinct emails when content is similar.
3. Search is optimized for generic media fields and does not support email-native operators (`from:`, `to:`, `subject:`, `label:`, `before:`, `has:attachment`, etc.).

These gaps make email archives difficult to use at scale and prevent parity with user expectations from mailbox search experiences.

## Goals

1. Persist first-class email identity and metadata (sender, recipients, message-id, thread hints, labels, attachments).
2. Introduce email-native query semantics with Gmail-style operators.
3. Improve dedupe correctness using stable message identity rather than content hash alone.
4. Support incremental sync for connected sources (starting with Gmail) using checkpoint/cursor semantics.
5. Preserve compatibility with existing media ingestion/search APIs while adding dedicated email routes.

## Non-Goals

1. Building a full webmail client (compose, send, threaded conversation UI).
2. Replacing existing generic media search behavior for non-email media types.
3. Full provider coverage in v1 (focus on upload formats plus Gmail connector).
4. Full-text indexing of all attachment binary formats in v1 (metadata-first, selective text extraction).

## Target Users and Primary Use Cases

1. Researchers ingesting large historical mail archives (`.eml`, `.mbox`, `.pst/.ost`) who need precise search filters.
2. Knowledge workers connecting live inboxes and running repeated queries over new mail.
3. Operators triaging ingestion errors or duplicates with source-level observability and sync checkpoints.

## Original Baseline (Historical)

This describes the original problem definition. For the current code audit and
verification limits, use the dated release evidence above.

1. Email parsing is already robust and includes core metadata extraction:
   - `tldw_Server_API/app/core/Ingestion_Media_Processing/Email/Email_Processing_Lib.py`
2. Upload entry points support `eml`, `zip(eml)`, `mbox`, and gated `pst/ost`:
   - `tldw_Server_API/app/api/v1/endpoints/media/process_emails.py`
3. Persistence currently applies a generic metadata allowlist that drops email-specific fields:
   - `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
4. Search currently emphasizes media title/content fields with generic filters:
   - `tldw_Server_API/app/core/DB_Management/Media_DB_v2.py`
5. Connector framework exists and can host mailbox providers:
   - `tldw_Server_API/app/api/v1/endpoints/connectors.py`
   - `tldw_Server_API/app/core/External_Sources/connectors_service.py`

## Product Scope

### In Scope

1. Email-native data model additions and migrations.
2. Ingestion pipeline changes to persist parsed email fields and stable message identity.
3. Email query parser and planner for operator-based search.
4. Dedicated email search endpoint and compatibility bridge from media search.
5. Gmail connector with incremental sync checkpoints and idempotent upserts.

### Out of Scope

1. UI redesign for full mailbox browsing.
2. Support for every IMAP provider in v1.
3. Advanced semantic ranking or LLM answer synthesis on top of email search (can be follow-on).

## Requirements

### Functional Requirements

#### FR-INGEST-001: Preserve Email Identity

System must persist tenant-and-source-scoped message identity fields:

1. `tenant_id` (user/org scope key)
2. `source_id` (internal provider/source record)
3. `source_message_id` (provider-native id when available)
4. RFC `message_id` header
5. normalized `internal_date`/received timestamp

Uniqueness requirement:

1. Primary unique key: `(tenant_id, source_id, source_message_id)` when `source_message_id` exists.
2. Secondary collision control: `(tenant_id, source_id, message_id)` when RFC `message_id` exists.
3. Hash fallback only when both ids are unavailable.
4. All email joins and lookups must include `tenant_id`.

#### FR-INGEST-002: Persist Structured Participants

System must persist `from`, `to`, `cc`, and `bcc` as normalized participant records and message-participant links, not just flattened text blobs.

#### FR-INGEST-003: Persist Labels and Attachments Metadata

System must persist mailbox labels/folders and attachment metadata with message linkage:

1. attachment filename
2. content type
3. byte size
4. content-id/disposition when present
5. optional extracted text availability flag

#### FR-INGEST-004: Archive and Upload Parity

All currently supported upload formats (`.eml`, `.zip`, `.mbox`, and `.pst/.ost` when parser backend is enabled) must write through the same normalized persistence path to guarantee consistent query behavior. When `.pst/.ost` parsing is not available, the API must return deterministic, informative errors and perform no partial normalized writes for those messages.

#### FR-INGEST-005: Attachment Extraction Policy

Attachment metadata is retained even when extraction is disabled. Explicit
`extract_attachments=false` keeps metadata only and overrides the legacy
`ingest_attachments` switch; omission inherits that switch. Extraction supports
nested EML only. Default selection is `message/rfc822` plus the legacy `.eml`
filename fallback; an explicit MIME allowlist uses declared types instead.
Deny rules take precedence. Unsupported binaries and PST/OST attachment payloads
remain metadata-only. Depth, count and size bounds still apply, and skipped child
mail must not enter its parent's body. Descriptor status explains captured or
skipped extraction without implying processing success or binary storage.

The multipart fields, MIME syntax, precedence and status reasons are documented in
`Docs/API-related/Email_Attachment_Policy_13376.md`; implementation rationale is in
`Docs/Design/Email_Attachment_Policy_13376_2026-09-26.md` (TASK-13376.3).

#### FR-SEARCH-001: Gmail-Style Query Operators (v1)

Search must support:

1. `from:`
2. `to:`
3. `cc:`
4. `bcc:`
5. `subject:`
6. `label:`
7. `has:attachment`
8. `before:YYYY-MM-DD`
9. `after:YYYY-MM-DD`
10. `older_than:`
11. `newer_than:`
12. free-text terms (subject/body/participants depending on planner)

#### FR-SEARCH-002: Deterministic Query Planning

Operator queries must compile to deterministic SQL with:

1. explicit joins for participants/labels
2. FTS-backed text conditions
3. timestamp filters using normalized UTC values
4. stable ordering for tie-breakers

#### FR-SEARCH-003: Endpoint Contract

System must provide a dedicated endpoint:

1. `GET /api/v1/email/search`

And a compatibility bridge:

1. `POST /api/v1/media/search` with `media_types=["email"]` and optional `email_query_mode=operators` (current compatibility surface).

#### FR-SEARCH-004: Boolean Semantics and Pagination Contract

Email query behavior must be explicitly defined and deterministic:

1. Default operator is `AND`.
2. `OR` is supported explicitly via `OR`.
3. Unary negation is supported via leading `-` for free-text and fielded terms.
4. Parentheses are not supported in v1; usage returns structured 400 parse errors.
5. Default ordering is `internal_date DESC, email_message_id DESC`.
6. Search endpoint must support cursor pagination aligned to default ordering.

#### FR-SYNC-001: Gmail Connector (v1 Provider)

Connector subsystem must support Gmail as an email source with:

1. initial backfill
2. incremental sync via history/checkpoint cursor
3. idempotent updates and label delta processing

#### FR-SYNC-002: Sync State and Observability

System must persist per-source sync state:

1. last successful cursor/history id
2. last run timestamp
3. error state and retry backoff counters

#### FR-SYNC-003: Cursor Invalidation Recovery

System must recover from invalid/expired provider cursors:

1. Detect invalid cursor/history-id responses and mark source state accordingly.
2. Attempt bounded replay from a configurable window.
3. Escalate to full backfill when bounded replay cannot restore continuity.
4. Emit metrics and operator-visible status for recovery events.

#### FR-GOV-001: Retention and Deletion Guarantees

Email data lifecycle must be enforceable:

1. Support tenant-scoped retention policies for normalized email tables.
2. Support hard-delete workflows for user-requested data deletion.
3. Ensure retention/deletion behavior applies to both legacy and normalized stores during migration.

### Non-Functional Requirements

#### NFR-PERF-001: Search Latency

For an actual 1,000,000-message mailbox benchmark (single tenant), the warm-pass
aggregate over the complete ten-class query mix must meet:

1. p50 <= 250 ms
2. p95 <= 900 ms

The ten classes and warm/cold methodology are fixed in
`Docs/Product/Email_Search_Benchmark_Protocol.md`. Every required case must match
stored messages; per-query latency is diagnostic rather than a separate SLO gate.
The cold pass opens a fresh MediaDatabase handle for each query; backend pools may
reuse physical connections. Host filesystem and PostgreSQL server caches remain
unchanged, so these are handle-reopen diagnostics rather than proof of cache-cold
performance. Cold-start migration windows are outside the warm SLO claim.

#### NFR-PERF-002: Ingestion Throughput

Batch archive ingestion should sustain at least 50 messages/sec on reference dev hardware for metadata-only persistence (no heavy attachment OCR).
Record authenticated HTTP request duration, total wall time, batch/message counts
and retry/isolation checks for each backend. A fixture loader's setup rate or three
short archive batches cannot satisfy this sustained-ingestion gate.

#### NFR-REL-001: Idempotency

Re-running the same sync window must not create duplicates and must remain safe under retries.

#### NFR-SEC-001: Access Control

Email data must remain tenant/user isolated using existing AuthNZ and database scoping patterns.

#### NFR-OBS-001: Metrics

Expose counters and timings for:

1. messages ingested
2. dedupe collisions avoided
3. query parse failures
4. query latency buckets
5. connector sync lag and failures

#### NFR-PERF-003: Reproducible Benchmark Protocol

Performance claims must reference a fixed benchmark protocol:

1. documented hardware profile and DB backend
2. documented dataset shape (message count, attachment ratio, label cardinality)
3. fixed query mix and warm/cold run methodology
4. published benchmark script/fixture location in repo

## Proposed Data Model (v1)

Add email-focused tables in Media DB context (names can be prefixed per conventions):

1. `email_sources`
   - `tenant_id`, provider, source handle, auth linkage, status
2. `email_messages`
   - `tenant_id`, identity fields, subject/body snippets, timestamps, content hash, raw metadata json
3. `email_participants`
   - `tenant_id`, normalized address/display name
4. `email_message_participants`
   - `tenant_id`, role (`from|to|cc|bcc`) relation table
5. `email_labels`
   - `tenant_id`, provider label id + normalized label name
6. `email_message_labels`
   - `tenant_id`, many-to-many mapping
7. `email_attachments`
   - `tenant_id`, per-message attachment metadata
8. `email_sync_state`
   - `tenant_id`, per source cursor/history id and sync health

Tenancy constraint:

1. Every `email_*` table must include `tenant_id`.
2. Every foreign key and join path must remain tenant-scoped.

Required indexes:

1. unique `(tenant_id, source_id, source_message_id)`
2. secondary unique `(tenant_id, source_id, message_id)` with null-aware handling
3. timestamp index on internal date
4. participant lookup indexes by `(tenant_id, normalized address)`
5. label lookup indexes by `(tenant_id, normalized label)`

FTS design:

1. `email_fts` virtual table/index over `subject`, `body_text`, and searchable participant/label projections.

## Proposed API Surface (v1)

1. `POST /api/v1/email/sources`
   - create/connect an email source
2. `GET /api/v1/email/sources`
   - list sources and sync status
3. `POST /api/v1/email/sources/{source_id}/sync`
   - trigger backfill or incremental sync
4. `GET /api/v1/email/search`
   - operator and text query endpoint (cursor pagination)
5. `GET /api/v1/email/messages/{email_message_id}`
   - message detail (metadata, participants, labels, attachment metadata) using internal opaque id (not RFC `message_id`)

Compatibility:

1. existing `/api/v1/media/process-emails` and `/api/v1/media/add` remain supported
2. `/api/v1/media/search` may optionally delegate to email planner when requested

## Query Language (v1)

Supported grammar components:

1. free text tokens with phrase support (`"quoted phrase"`)
2. unary operators (`has:attachment`)
3. key-value operators (`from:alice@example.com`, `label:inbox`)
4. date operators (`before:2026-02-01`, `after:2026-01-01`)
5. relative time operators (`older_than:30d`, `newer_than:7d`)

Behavioral rules:

1. tokens are `AND` by default
2. explicit `OR` is supported
3. unary negation (`-`) is supported for free-text and fielded terms
4. parentheses are not supported in v1 and return structured 400 parse errors
5. repeated field operators are `OR` within field unless explicitly configured otherwise
6. unsupported operators return structured 400 parse errors
7. planner normalizes date filters to UTC boundaries
8. default sort is `internal_date DESC, email_message_id DESC`
9. cursor pagination follows default sort and must be stable across pages

## Migration and Backward Compatibility

1. Add schema migrations for new tables and indexes (SQLite and PostgreSQL paths).
2. During phases 1 and 2, dual-write email ingests into legacy media rows and normalized email tables.
3. Source-of-truth by surface:
   - `/api/v1/email/*` reads from normalized email tables.
   - `/api/v1/media/search` remains legacy by default until cutover flag is enabled.
4. Add reconciliation job/reporting to detect and repair legacy-vs-normalized drift during dual-write.
5. Incrementally backfill legacy email media rows into new tables via migration job.
6. Preserve existing response shapes where routes are unchanged.
7. Provide feature flags for staged rollout:
   - `EMAIL_NATIVE_PERSIST_ENABLED`
   - `EMAIL_OPERATOR_SEARCH_ENABLED`
   - `EMAIL_GMAIL_CONNECTOR_ENABLED`

## Delivery Plan

### Phase 0: Quick-Win Hardening

1. Preserve currently parsed metadata in persistence allowlist.
2. Improve dedupe precedence to prefer message identity.
3. Add ingestion metrics for duplicate suppression and parse failures.

Exit Criteria:

1. Distinct messages with identical bodies no longer collide in normal cases.
2. Existing email ingestion tests pass plus new dedupe regression tests.

### Phase 1: Email-Native Storage and Search

1. Introduce new email tables and indexes.
2. Implement write path from existing email parser outputs.
3. Add `GET /api/v1/email/search` with parser plus planner.
4. Add FTS synchronization for email searchable fields.

Exit Criteria:

1. Operator queries return accurate filters across a seeded test corpus.
2. Performance targets meet baseline on reference fixtures.

### Phase 2: Gmail Connector and Incremental Sync

1. Add Gmail source in connector registry and auth flows.
2. Implement initial backfill and incremental cursor sync.
3. Add sync health/status visibility in source endpoints.
4. Implement cursor invalidation recovery (bounded replay then full backfill).

Exit Criteria:

1. Repeated incremental sync runs are idempotent.
2. Label updates and message state changes propagate without full reindex.
3. Invalid cursor events auto-recover or escalate with explicit operator-visible state.

### Phase 3: Migration Completion and Optimization

1. Backfill legacy persisted emails into normalized tables.
2. Optimize query plans and indexes based on observed workload.
3. Finalize compatibility strategy for `/media/search` delegation.
4. Enforce retention and deletion policy across normalized email tables.

Exit Criteria:

1. Legacy and new ingestion paths converge on identical searchability.
2. Operational metrics remain within SLOs for production workloads.

## Success Metrics

1. Search relevance: >= 95 percent pass rate on curated operator-query test suite.
2. Dedupe correctness: < 0.1 percent false-positive duplicate merges in sampled audits.
3. Sync freshness: incremental sync lag median < 5 minutes for active connected inboxes.
4. Reliability: ingestion/sync job success rate >= 99 percent excluding provider outages.
5. Performance: NFR latency targets met for benchmark corpus.

## Risks and Mitigations

1. Risk: schema complexity increases migration risk.
   - Mitigation: phased feature flags, dual-write period, backfill jobs with validation.
2. Risk: operator parser ambiguity or user confusion.
   - Mitigation: strict parser errors, clear docs, compatibility fallback to free-text search.
3. Risk: connector provider API quotas and transient failures.
   - Mitigation: checkpointed incremental sync, exponential backoff, retry budgets.
4. Risk: large attachment metadata and text extraction cost.
   - Mitigation: metadata-first indexing, optional extraction toggles, size caps.

## Security and Compliance Considerations

1. Enforce tenant and user scoping for all email source/message reads and writes.
2. Avoid logging raw bodies, headers, attachment filenames, credentials or metadata in INFO-and-higher email diagnostics, including echoed dependency failures and traceback locals. Use bounded stage/outcome/error-type fields; preserve error propagation and rollback behavior.
3. Apply existing secret management patterns for provider credentials.
4. Enforce documented retention and hard-delete workflows for email content in both legacy and normalized stores during migration.

## Testing and Validation Plan

1. Unit tests:
   - query parser/operator normalization
   - dedupe key selection
   - label/participant mapping
2. Integration tests:
   - upload archives produce normalized rows
   - email search endpoint correctness with mixed operators
3. Migration tests:
   - SQLite and PostgreSQL migration application and rollback safety
4. Connector tests:
   - incremental sync idempotency and cursor advancement
5. Regression tests:
   - existing media and email ingestion endpoints unchanged where expected

## Documentation Deliverables

1. Update API docs:
   - `Docs/API-related/Email_Processing_API.md`
2. Add operator query guide for users:
   - `Docs/User_Guides/` (new page)
3. Add developer integration notes:
   - `Docs/Code_Documentation/` (new page for email search architecture)

## Open Questions

1. Should v1 include IMAP generic source, or only Gmail connector plus upload archives?
2. Should conversation/thread reconstruction be required for v1 search results, or deferred?
3. What bounded replay window should be used before escalating cursor recovery to full backfill?
4. Binary attachment text extraction is deferred. The v1 nested-EML-only default and MIME policy are resolved by FR-INGEST-005 and `Docs/API-related/Email_Attachment_Policy_13376.md`.

## Implementation Checklist and Milestone Tickets

### Ticket Conventions

1. Ticket IDs use `EMAIL-M{milestone}-{nnn}`.
2. Status tracking uses markdown checkboxes:
   - `[ ]` incomplete implementation, verification or approval
   - `[x]` complete
3. Milestone closure requires all `Must` tickets in the selected rollout scope complete and milestone exit criteria met. M2 and live Gmail portions of M4 apply only to optional Gmail enablement.

### Milestone M0: Quick-Win Hardening (Phase 0)

Milestone Goal: Eliminate highest-risk ingestion/search correctness gaps without schema redesign.

Must Tickets:

- [x] `EMAIL-M0-001` Persist parsed email metadata fields in safe persistence path.
  Depends On: None.
  Deliverables: Update metadata allowlist and persistence tests so `from/to/cc/bcc/subject/date/message_id/headers_map/attachments` survive ingest.
  Acceptance: Existing email ingest tests pass; new assertions confirm persisted metadata completeness.

- [x] `EMAIL-M0-002` Implement dedupe precedence for message identity.
  Depends On: `EMAIL-M0-001`.
  Deliverables: Dedupe order `(source_id, source_message_id)` -> `(source_id, message_id)` -> hash fallback.
  Acceptance: Regression tests demonstrate distinct messages with same body are no longer merged.

- [x] `EMAIL-M0-003` Add ingestion metrics for dedupe and parse outcomes.
  Depends On: None.
  Deliverables: Counters/timers for parsed messages, dedupe matches, parse failures, persistence failures.
  Acceptance: Metrics visible in existing monitoring path and covered by unit/integration checks.
  Closeout: TASK-13376.1 adds bounded parse/dedupe/persistence observations; registry tests and both guarded sustained HTTP reports retain actual parse/Media/native successes, dedupe matches and native-search duration/result observations in the core release record.

- [x] `EMAIL-M0-004` Add feature flags for staged rollout.
  Depends On: None.
  Deliverables: `EMAIL_NATIVE_PERSIST_ENABLED`, `EMAIL_OPERATOR_SEARCH_ENABLED`, `EMAIL_GMAIL_CONNECTOR_ENABLED` wired into config.
  Acceptance: Flags default safely off/on per rollout plan and are validated in config tests.

Should Tickets:

- [x] `EMAIL-M0-005` Update `Docs/API-related/Email_Processing_API.md` with metadata persistence behavior.
  Depends On: `EMAIL-M0-001`.
  Acceptance: Docs reflect actual persisted fields and fallback behavior.

- [x] `EMAIL-M0-006` Add audit logging guardrails for sensitive fields.
  Depends On: None.
  Acceptance: No raw message body, sensitive headers, filenames, credentials or metadata in INFO-and-higher diagnostics across the verified upload/storage/search/access-log failure paths.
  Validation (2026-09-26): TASK-13376.2 is implemented in `b880be1b53`; the focused logging/pool/access-log suite passed 83 tests, including real lower-backend schema failures with preserved exception causes. Additional legacy-search success/fallback/failure sentinel paths and 24 selected regressions passed; Ruff/Bandit reported no findings in the touched scope. Audit: `Docs/Operations/Email_Sensitive_Logging_Audit_13376_2_2026-09-26.md`.

Milestone Exit Gate:

- [ ] M0 Gate approved by the owner with ingestion correctness, metrics and logging criteria validated for the selected rollout scope.

### Milestone M1: Email-Native Storage and Operator Search (Phase 1)

Milestone Goal: Deliver first-class email storage model and usable Gmail-style query endpoint.

Must Tickets:

- [x] `EMAIL-M1-001` Create schema migration for email-native tables.
  Depends On: `EMAIL-M0-004`.
  Deliverables: `email_sources`, `email_messages`, `email_participants`, `email_message_participants`, `email_labels`, `email_message_labels`, `email_attachments`, `email_sync_state` with mandatory `tenant_id`.
  Acceptance: SQLite and PostgreSQL migration tests pass.

- [x] `EMAIL-M1-002` Add identity and query indexes.
  Depends On: `EMAIL-M1-001`.
  Deliverables: Tenant-scoped unique constraints and lookup indexes defined in PRD.
  Acceptance: Schema introspection tests validate expected indexes.

- [x] `EMAIL-M1-003` Implement ingestion write path into normalized email tables.
  Depends On: `EMAIL-M1-001`, `EMAIL-M0-001`, `EMAIL-M0-002`.
  Deliverables: Parser outputs mapped to normalized tables for all upload formats.
  Acceptance: Integration tests confirm `.eml`, `.zip`, `.mbox` parity and `.pst/.ost` parity when parser backend is enabled, with deterministic degraded-mode errors otherwise.

- [x] `EMAIL-M1-004` Implement email FTS structures and sync logic.
  Depends On: `EMAIL-M1-001`, `EMAIL-M1-003`.
  Deliverables: `email_fts` creation and update hooks for insert/update/delete.
  Acceptance: FTS search fixtures return expected rows and ranking stability.

- [x] `EMAIL-M1-005` Implement operator query parser (v1 grammar).
  Depends On: None.
  Deliverables: Parser for `from/to/cc/bcc/subject/label/has:attachment/before/after/older_than/newer_than` plus free text, explicit `OR`, unary negation, and no-parentheses v1 enforcement.
  Acceptance: Unit tests cover valid/invalid query cases with structured errors.

- [x] `EMAIL-M1-006` Implement deterministic query planner (AST -> SQL).
  Depends On: `EMAIL-M1-005`, `EMAIL-M1-001`, `EMAIL-M1-004`.
  Deliverables: SQL planner with participant/label joins, FTS terms, UTC date normalization, and stable sort (`internal_date DESC, email_message_id DESC`) for cursor pagination.
  Acceptance: Planner golden tests pass and produce stable SQL signatures.

- [x] `EMAIL-M1-007` Add `GET /api/v1/email/search`.
  Depends On: `EMAIL-M1-005`, `EMAIL-M1-006`.
  Deliverables: Endpoint, schemas, auth checks, cursor pagination, sorting.
  Acceptance: API integration tests pass and return expected filtered results.

- [x] `EMAIL-M1-008` Add compatibility bridge in `/api/v1/media/search` for email operator mode.
  Depends On: `EMAIL-M1-007`.
  Deliverables: Optional delegation path with backward-compatible defaults.
  Acceptance: Existing media search tests remain green; new bridge tests added.

- [x] `EMAIL-M1-009` Performance benchmark harness for email search.
  Depends On: `EMAIL-M1-007`.
  Deliverables: Repeatable benchmark fixtures/scripts plus documented benchmark protocol (hardware, dataset shape, query mix, warm/cold runs).
  Implemented Artifacts:
  - `Helper_Scripts/benchmarks/email_search_bench.py`
  - `Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc`
  - `Docs/Product/Email_Search_Benchmark_Protocol.md`
  Acceptance: p50/p95 targets from NFR are measured and reported using the documented protocol.

Should Tickets:

- [x] `EMAIL-M1-010` Add `GET /api/v1/email/messages/{email_message_id}` detail endpoint.
  Depends On: `EMAIL-M1-003`.
  Acceptance: Detail API uses opaque internal identifier and returns participant, label, and attachment metadata consistently.

Milestone Exit Gate:

- [ ] M1 Gate approved with operator-query correctness suite >= 95 percent pass rate.

### Milestone M2: Gmail Connector and Incremental Sync (Phase 2)

Milestone Goal: Sync live Gmail sources with reliable incremental updates.

Must Tickets:

- [x] `EMAIL-M2-001` Add Gmail connector provider registration and config plumbing.
  Depends On: `EMAIL-M0-004`, `EMAIL-M1-001`.
  Deliverables: Provider registration, auth config, source creation path.
  Acceptance: Source creation API can persist Gmail source records.

- [x] `EMAIL-M2-002` Implement Gmail initial backfill worker flow.
  Depends On: `EMAIL-M2-001`, `EMAIL-M1-003`.
  Deliverables: Batch fetch and normalized upsert for initial mailbox import.
  Acceptance: Controlled seed mailbox imports completely with idempotent rerun.

- [x] `EMAIL-M2-003` Implement incremental sync cursor/history logic.
  Depends On: `EMAIL-M2-002`.
  Deliverables: `email_sync_state` read/write, history cursor advancement.
  Acceptance: Repeated sync cycles process only deltas and advance cursor safely.

- [x] `EMAIL-M2-004` Implement label delta and message state reconciliation.
  Depends On: `EMAIL-M2-003`.
  Deliverables: Apply add/remove label changes without full message reindex.
  Acceptance: Sync tests validate label updates and removals accurately.

- [x] `EMAIL-M2-005` Add source sync endpoints (`POST /email/sources/{id}/sync`, `GET /email/sources` status).
  Depends On: `EMAIL-M2-003`.
  Deliverables: Trigger and status APIs with auth and tenancy checks.
  Acceptance: Endpoint tests cover successful sync, retry, and failure states.

- [x] `EMAIL-M2-006` Add retry and backoff policy for provider failures/quota limits.
  Depends On: `EMAIL-M2-003`.
  Deliverables: Exponential backoff and bounded retry budget per source.
  Acceptance: Fault-injection tests confirm controlled retries and no duplicate writes.

- [x] `EMAIL-M2-009` Implement cursor invalidation recovery flow.
  Depends On: `EMAIL-M2-003`.
  Deliverables: Detect invalid/expired cursor, run bounded replay, escalate to full backfill when needed, persist operator-visible recovery state.
  Acceptance: Integration tests simulate invalid cursor responses and verify successful recovery or explicit escalation state.

Should Tickets:

- [x] `EMAIL-M2-007` Add sync lag dashboard and alerting hooks.
  Depends On: `EMAIL-M2-003`, `EMAIL-M0-003`.
  Acceptance: Lag and failure rate surfaced in monitoring and alert policy docs.

- [x] `EMAIL-M2-008` Add provider contract tests with mocked Gmail API responses.
  Depends On: `EMAIL-M2-003`.
  Acceptance: Contract suite catches malformed payload and pagination edge cases.

Milestone Exit Gate:

- [ ] M2 Gate approved with idempotent incremental sync, median lag < 5 minutes in staging, and validated cursor invalidation recovery.
  Validation Checklist: `Docs/Product/Email_Ingestion_Search_PRD.md`.

### Milestone M3: Legacy Backfill and Query Optimization (Phase 3)

Milestone Goal: Converge old and new data paths and harden production performance.

Must Tickets:

- [x] `EMAIL-M3-001` Build legacy email backfill job from media rows into normalized tables.
  Depends On: `EMAIL-M1-003`.
  Deliverables: Backfill worker with resumable progress checkpoints.
  Acceptance: Backfill can process large datasets and resume after interruption.

- [x] `EMAIL-M3-002` Add dual-read validation between legacy and normalized search paths.
  Depends On: `EMAIL-M3-001`, `EMAIL-M1-007`.
  Deliverables: Validation tooling comparing result parity for sampled queries (`Helper_Scripts/checks/email_search_dual_read_parity.py`, `tldw_Server_API/tests/Helper_Scripts/test_email_search_dual_read_parity.py`).
  Acceptance: Parity reports meet agreed threshold before cutover.

- [x] `EMAIL-M3-003` Optimize planner/indexes using real workload traces.
  Depends On: `EMAIL-M1-009`, `EMAIL-M3-002`.
  Deliverables: Index tuning and planner optimizations documented and benchmarked (`tldw_Server_API/app/core/DB_Management/Media_DB_v2.py`, `Helper_Scripts/benchmarks/email_search_bench.py`, `Docs/Product/Email_Search_M3_003_Index_Tuning.md`).
  Acceptance: NFR latency targets met on benchmark corpus.
  Historical implementation validation (2026-02-10): `warm_p50_ms=9.51`, `warm_p95_ms=10.89` with trace-driven query mix (`workload_top_n=5`), and query-plan capture reported `queries_with_index_hits=5/5`. This smaller five-query run does not certify the complete ten-class, 1,000,000-message NFR gate. TASK-13376.4 records final SQLite and PostgreSQL scale evidence.
  Final validation (2026-09-26): Complete 1,000,000-message certificates pass at warm aggregate SQLite 202.23/725.74 ms and PostgreSQL 220.92/495.49 ms, with all ten populated cases and meaningful negation. Final PostgreSQL uses `880d690a93` on the dedicated 256 MiB service; earlier455/shared 64 MiB results are historical. See `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`.

- [x] `EMAIL-M3-004` Finalize `/media/search` delegation strategy and default behavior.
  Depends On: `EMAIL-M1-008`, `EMAIL-M3-002`.
  Deliverables: Cutover config and compatibility notes (`EMAIL_MEDIA_SEARCH_DELEGATION_MODE`, `Docs/Product/Email_Search_M3_004_Media_Search_Delegation.md`).
  Acceptance: No breaking API changes; existing clients continue functioning.
  Validation (2026-02-10): Added integration coverage for default legacy behavior, auto-email cutover delegation, explicit legacy override, and auto-mode fallback when operator search is disabled.

- [x] `EMAIL-M3-005` Add data retention and deletion policy enforcement for normalized email tables.
  Depends On: `EMAIL-M3-001`.
  Deliverables: Tenant-scoped retention/hard-delete enforcement methods in `Media_DB_v2` and coverage in `tldw_Server_API/tests/DB_Management/test_email_native_stage1.py`; implementation note: `Docs/Product/Email_Search_M3_005_Retention_Deletion.md`.
  Acceptance: Retention/deletion tests pass with tenant-safe scoping.
  Validation (2026-02-10): `pytest tldw_Server_API/tests/DB_Management/test_email_native_stage1.py -q` passed (`16 passed`), including tenant-scope retention and hard-delete workflow tests.

Should Tickets:

- [x] `EMAIL-M3-006` Add attachment extraction policy toggles and MIME-specific defaults.
  Depends On: `EMAIL-M1-003`.
  Acceptance: Extraction behavior configurable and documented.
  Validation (2026-09-26): TASK-13376.3 is implemented in `b880be1b53`. The FR-INGEST-005 contract, parser policy and persisted nested-child paths are included in the 328-pass core suite; two optional real-PST fixtures are skipped, with fake-adapter metadata-only behavior and missing-parser errors separately tested. See `Docs/API-related/Email_Attachment_Policy_13376.md` and `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`. TASK-13377 subsequently passes both native real-PST endpoint cases; provenance and limits are in `Docs/Operations/Email_Real_PST_Validation_2026-09-26.md`.

Milestone Exit Gate:

- [ ] M3 Gate approved by the owner after the selected local reference scope has full-scale search, sustained ingestion, parity/cutover/rollback and retention/deletion evidence.
  Prior bounded tests and implementation checkmarks are historical evidence, not this release approval. Validation checklist: `Docs/Operations/Email_Release_Checklist_and_Rollback.md` (TASK-13376).

### Milestone M4: Release Readiness and Documentation

Milestone Goal: Complete rollout safeguards, docs, and operational handoff.

Must Tickets:

- [x] `EMAIL-M4-001` Publish operator query user guide.
  Depends On: `EMAIL-M1-007`.
  Deliverables: New guide in `Docs/User_Guides/Server/Email_Operator_Search_Guide.md` (mirrored to `Docs/Published/User_Guides/Server/Email_Operator_Search_Guide.md`) with examples and troubleshooting.
  Acceptance: Guide content published with operator syntax, endpoint usage, and error handling references; owner review pending the selected rollout scope's sign-off.

- [x] `EMAIL-M4-002` Publish email search architecture and developer integration docs.
  Depends On: `EMAIL-M1-006`, `EMAIL-M2-003`.
  Deliverables: New page `Docs/Code_Documentation/Email_Search_Architecture.md`.
  Acceptance: Includes schema, parser/planner flow, and extension guidelines.

- [x] `EMAIL-M4-003` Create production runbook for sync operations and incident response.
  Depends On: `EMAIL-M2-006`, `EMAIL-M2-007`.
  Deliverables: Ops runbook covering core offline validation and optional Gmail retries, cursor repair, and quota incidents (`Docs/Operations/Email_Sync_Operations_Runbook.md`).
  Acceptance: Runbook published with synthetic core validation and connector incident playbooks. Live Gmail execution is deferred and required only for the optional Gmail enablement gate.
  Validation (2026-02-10): Focused endpoint/worker regression slice passed (`10 passed`) covering source status/sync APIs, cursor recovery (bounded replay/full-backfill-required), label/message-state deltas, retry backoff, retry-budget exhaustion, and large Gmail fixture edge cases.

- [x] `EMAIL-M4-004` Final release checklist and rollback plan.
  Depends On: `EMAIL-M3-004`.
  Deliverables: Feature-flag rollout sequence and rollback triggers (`Docs/Operations/Email_Release_Checklist_and_Rollback.md`).
  Acceptance: Separate core and optional Gmail checklists and rollback steps published; one owner records approval for each enabled scope.
  Validation (2026-02-10): Rollout phase checklist, objective rollback triggers, and ordered rollback execution steps documented with explicit ownership and sign-off gates.

Milestone Exit Gate:

- [ ] M4 Core gate approved and file ingestion/search enabled for target rollout scope.
  Current status (2026-09-26): All core technical gates pass for the selected local reference scope: SQLite then dedicated PostgreSQL, complete search/sustained HTTP/metrics/parity/rollback and cleanup. Final PostgreSQL source is `880d690a93`, with aggregate throughput61.71 messages/sec; three slow batches remain diagnostics. Owner sign-off is separate, unrecorded and not inferred from implementation checkmarks.
  Closure criteria: Record the selected scope's final technical artifacts and remaining limits, then record the single owner's release approval in the core release checklist.
- [ ] M4 Optional Gmail enablement gate approved (deferred; not a core dependency).
  Live OAuth, provider behavior, backfill/incremental sync and staging lag remain unverified. Mocked tests and fixture metrics are separate evidence. Any future live work requires an explicitly authorized dedicated synthetic test mailbox and a downstream processing audit. The owner's personal Gmail is excluded.

### Critical Path Summary

1. `EMAIL-M0-001` -> `EMAIL-M0-002` -> `EMAIL-M1-001` -> `EMAIL-M1-003` -> `EMAIL-M1-005` -> `EMAIL-M1-006` -> `EMAIL-M1-007`.
2. `EMAIL-M2-001` -> `EMAIL-M2-002` -> `EMAIL-M2-003` -> `EMAIL-M2-009` -> `EMAIL-M2-005`.
3. `EMAIL-M3-001` -> `EMAIL-M3-002` -> `EMAIL-M3-003` -> `EMAIL-M3-004` -> `EMAIL-M3-005` -> `EMAIL-M4-004`.
