# Email Release Checklist and Rollback

Audience: Owner / maintainer
Status (2026-09-26): Core technical gates complete for the local reference deployment, SQLite then dedicated PostgreSQL (TASK-13376). Owner release sign-off is unrecorded and separate. Optional live Gmail validation is deferred.

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Email_Sync_Operations_Runbook.md`
- `Docs/Operations/Email_Core_Validation_2026-09-13.md`
- `Docs/Operations/Email_Local_Startup_and_Search_Performance_2026-09-25.md`
- `Docs/Operations/Email_Live_SQLite_Validation_2026-09-25.md`
- `Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md`
- `Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md`
- `Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md`
- `Docs/Product/Email_Search_Benchmark_Protocol.md`
- `Docs/API-related/Email_Attachment_Policy_13376.md`
- `Docs/Operations/Email_Sensitive_Logging_Audit_13376_2_2026-09-26.md`
- `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`
- `Docs/Operations/Env_Vars.md`

## Single-Owner Workflow

One maintainer owns implementation, operations and release approval. Record approval
for the scope actually enabled. Gmail is optional; unavailable Gmail access cannot
block file-based ingestion/search. All development/validation uses synthetic mail.
The owner's personal Gmail and personal email are excluded.

The chosen technical validation scope is a local reference deployment with one
loopback Uvicorn worker. Final PostgreSQL uses a dedicated disposable service at
loopback port 5435 with 256 MiB shared memory and scoped role/databases. The shared
port 5434 service is preserved; its earlier results are historical. Rehearsing configuration reloads in this disposable
app does not certify process restarts, multi-worker deployments or another host.
Finish all technical work and preserve reviewable evidence before recording the
single owner's release decision; do not infer sign-off from a completed task.

## Core File Ingestion and Search Gate

- [x] Trace upload parsing, persistence, indexing and optional model calls; record explicit offline options and interception evidence (TASK-13250).
- [x] Validate synthetic EML, ZIP and MBOX messages with distinct bodies, attachment metadata, repeat import, search and detail using temporary SQLite.
- [x] Resolve distinct-message same-body merging (TASK-13251). Strict EML/ZIP/MBOX regressions now require distinct stored/searchable identities; existing corrupted data requires separate recovery.
- [x] Implement FR-SEARCH-004 cursor pagination (TASK-13253); omitted cursor preserves offset behavior. See `Docs/Design/email-search-cursor-pagination.md` for traversal semantics.
- [x] Validate real API-key authentication and per-user SQLite isolation, including main-app route registration/test-mode request middleware (TASK-13255; `Docs/Operations/Email_Authenticated_Validation_2026-09-13.md`).
- [x] Validate authenticated synthetic EML uploads, key/role rejection, expected-user check, org storage quota and owner-scoped search in local ASGI/SQLite (TASK-13256; `Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md`).
- [x] Exercise the main FastAPI lifespan and scoped email search/detail plus media search with synthetic users, temporary SQLite and an outbound/model-call guard (TASK-13361; `Docs/Operations/Email_Local_Startup_and_Search_Performance_2026-09-25.md`).
- [x] Validate full-app Uvicorn startup/shutdown and scoped synthetic email upload/search/detail/media search over loopback HTTP with isolated SQLite (TASK-13362; `Docs/Operations/Email_Live_SQLite_Validation_2026-09-25.md`).
- [x] Register and verify the native email persistence counter in the live synthetic SQLite upload path (TASK-13363; same validation report).
- [x] Validate full-app Uvicorn startup/shutdown and scoped synthetic email upload/search/detail/media search with PostgreSQL AuthNZ and content, including direct forced-RLS isolation checks (TASK-13364; `Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md`).
- [x] Run a scoped 10,000-message PostgreSQL search benchmark and direct cross-user RLS checks (TASK-13369; `Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md`). Disposable resources are confirmed absent after Docker test-store removal; the obsolete private credential manifest was removed.
- [x] Measure synthetic authenticated archive ingestion on SQLite then PostgreSQL, verify 300-message persistence/retry/isolation, and fix PostgreSQL routine-bootstrap sequence rewinding (TASK-13371/TASK-13372; `Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md`).
- [x] Validate native-only archive transactions with shared saved-payload reads, native rollback preserving committed Media rows, changed retry payloads, and guarded SQLite then PostgreSQL probes (TASK-13375; `Docs/Operations/Email_Archive_Native_Transaction_Validation_2026-09-26.md`).
- [x] Verify bounded parse, dedupe and persistence metrics in the real registry and guarded HTTP upload path (TASK-13376.1; EMAIL-M0-003). Final SQLite and PostgreSQL reports retain actual parse/Media/native success, dedupe and search observations.
- [x] Verify INFO-and-higher sensitive logging captures and failure propagation across the audited upload/storage/search/access-log paths (TASK-13376.2; EMAIL-M0-006). Stage 1 commit `b880be1b53`, 83 focused logging/pool/access-log cases, plus six legacy-search sentinel paths and 24 selected legacy regressions; touched Ruff/Bandit clean.
- [x] Verify metadata-only defaults, explicit extraction override, MIME allow/deny precedence and persisted nested EML links (TASK-13376.3; EMAIL-M3-006). Stage 1 commit `b880be1b53`; included in 328 passing core tests with two optional real-PST skips. See `Docs/API-related/Email_Attachment_Policy_13376.md`.
- [x] Execute both optional native PST endpoint tests with a pinned public synthetic fixture (TASK-13377). Ten endpoint/metadata cases pass without skips; actual OST containers remain unverified. See [real PST validation](Email_Real_PST_Validation_2026-09-26.md).
- [x] Certify sustained archive ingestion of at least 50 messages/sec for metadata-only persistence on each backend (TASK-13376.5). SQLite 130.54 and final dedicated PostgreSQL 61.71 messages/sec pass over >= 60 measured request seconds. PostgreSQL batch diagnostic is false (3/38 below 50, minimum 38.96); the gate is minute-wide aggregate. Historical failed/bounded/shared-service results remain separate.
- [x] Record complete 1,000,000-message fixtures, representative shape, all ten populated cases, meaningful negation and aggregate warm p50 <= 250 ms/p95 <= 900 ms (TASK-13376.4). SQLite 202.23/725.74 ms and final dedicated PostgreSQL 220.92/495.49 ms at `880d690a93` pass; cold-handle and per-case diagnostics retained separately.
- [x] Validate the chosen local reference topology and target scale with full archive and million-message certificates (TASK-13376.6). Final PostgreSQL resource profile is dedicated 256 MiB shared memory; shared 64 MiB diagnostics are historical.
- [x] Record complete legacy/native media-search result-set parity for the selected synthetic queries in both backends (TASK-13376.6). SQLite 7,900/1/1; final PostgreSQL 3,800/1/1, each with exact equal IDs.
- [x] Configure and verify core rollout flags in the local reference environment:
  - `EMAIL_NATIVE_PERSIST_ENABLED=true`
  - `EMAIL_OPERATOR_SEARCH_ENABLED=true`
  - `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
  - `EMAIL_GMAIL_CONNECTOR_ENABLED=false`
- [x] Verify `GET /api/v1/email/search`, `GET /api/v1/email/messages/{id}`, and existing `POST /api/v1/media/search` behavior in each local backend.
- [x] Validate `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=auto_email` separately, compare it with explicit operator results, and restore `opt_in`. Exact sets match 7,900 SQLite and 3,800 final PostgreSQL IDs.
- [x] Rehearse native/search flag rollback, verify disabled endpoints and explicit operator requests reject as expected, confirm legacy data remains, restore flags and verify detail retrieval (TASK-13376.6). Both reports retain all expected legacy IDs and 404/404/422 disabled statuses, then detail 200 and baseline flags restored.
- [x] Record cleanup of only probe-owned roots, scoped PostgreSQL databases/roles and manifests, and the owned dedicated container/volume. All 55 exact roots and five generated targets are absent; shared service preserved. Retain credential-free evidence artifacts and receipts.
- [ ] Owner approves the core rollout and records date/environment/evidence below.

Core validation does not require OAuth, a Gmail source or a connectors worker.
For isolated core validation, keep `CONNECTORS_WORKER_ENABLED=false`; this is a
shared worker flag, so do not disable unrelated connectors in an existing deployment.

## Optional Gmail Enablement Gate — Deferred

- [x] Run mocked Gmail parser, provider and worker regression tests; record their limits.
- [ ] Explicitly authorize a dedicated synthetic test mailbox for future live validation. Do not use the owner's personal Gmail. Audit downstream analysis, claims, chunking and embeddings before account access.
- [ ] Configure optional Gmail/worker flags and credentials in the chosen environment.
- [ ] Validate real OAuth and a connected source; a `scaffold=1` URL is not successful authorization.
- [ ] Execute the optional live-source checklist in the sync runbook: initial backfill, idempotent rerun, deltas, cursor recovery, provider failures and status.
- [ ] Measure sync lag from real controlled-source monitoring samples. Fixture checker output is not staging SLO evidence.
- [ ] Owner approves optional Gmail enablement separately from the core release.

A label filter is not an OAuth access boundary. Read-only Gmail access does not
prevent downstream model processing. No live Gmail work was performed for TASK-13250.

## Release Evidence

Current validation: `Docs/Operations/Email_Core_Validation_2026-09-13.md`.
Authenticated-access follow-up: `Docs/Operations/Email_Authenticated_Validation_2026-09-13.md`.
Local startup and bounded performance: `Docs/Operations/Email_Local_Startup_and_Search_Performance_2026-09-25.md`.
Live SQLite HTTP validation: `Docs/Operations/Email_Live_SQLite_Validation_2026-09-25.md`.
Live PostgreSQL HTTP/RLS validation: `Docs/Operations/Email_Live_PostgreSQL_Validation_2026-09-25.md`.
Bounded PostgreSQL performance and storage-recovery limits: `Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md`.
Archive correctness and measured throughput gaps: `Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md`.

### TASK-13376 Final Core Evidence

The following record contains final technical evidence. The only remaining core
checkbox is the single owner's separate release decision; optional Gmail remains deferred.
The main record is `Docs/Operations/Email_Core_Closeout_Validation_2026-09-26.md`.
The earlier SQLite million-message result of warm p50 191.52 ms / p95 714.60 ms
is superseded: its bulk fixture did not maintain canonical legacy Media FTS.
Both full search and sustained HTTP/local release certificates pass. Earlier
PostgreSQL `455fad64fb` search and `ae46f643ab` shared-service HTTP results are
historical/superseded by the final dedicated `880d690a93` certificates.

PostgreSQL query/archive tuning is committed as `aac02e68e8`. Its selected probe
profile is per-connection `PGOPTIONS='-c work_mem=64 MB'`, with observed
`shared_buffers=128 MB`, `hash_mem_multiplier=2` and
`max_parallel_workers_per_gather=2`; no global service configuration changed.
Substring predicates and forced RLS are preserved. The closeout report retains
the failed initial million-message result, later maintenance and bounded profiles.
The subsequent full attempt aborted during fixture parity before timing. Neither
the short tuning comparison nor a bounded archive profile closes the final gates.
Commit `1fee95629a` addresses Docker's 64 MiB POSIX shared-memory limit by setting
parallel workers to zero only in the verification transaction. Complete
parity/security checks remain; timed search restores two workers / 64 MB. The next
full run passed parity but failed repeated label COUNT. Transaction-local custom
planning preserves parameters, predicates and RLS and restores session mode on
success/error; a bounded pooled comparison and 22 regressions pass. The earlier
complete search certificate at `455fad64fb` passed at 248.63/659.39 ms; it is now
superseded for current-source acceptance. The
generic-plan transition is inferred from the preparation boundary, with no
post-failure generic count observed. No global service configuration changed,
and no acceptance is inferred from diagnostic samples.

Legacy/SQL-cache/identity fixes at `ae46f643ab` passed shared-service HTTP. Its fresh
search attempt failed POSIX shared-memory SQLSTATE 53100 at 64 MiB; no pass was
inferred. Final probe source `880d690a93` uses the dedicated 256 MiB profile at
loopback 5435, preserving shared 5434 and production queries/RLS. Both fresh search
and full HTTP pass with 23 unchanged hashes. Quality is Ruff/Bandit clean with 62
compiled files/38 production-probe paths, 211 combined and 91 helper regressions.
Final cleanup is complete; owner approval is separate and unrecorded.

| Evidence | SQLite | PostgreSQL |
| --- | --- | --- |
| Sustained HTTP messages / measured request seconds / total wall seconds / messages per second | **Pass:** 7,900 / 60.5196 / 62.9324; 130.54 by request time, 125.53 by wall time; 79 batches, each >= 50. `Docs/Operations/evidence/email_core_closeout_13376/sustained_sqlite.json` | **Aggregate pass:** 3,800 / 61.5734 / 62.4711; **61.71** by request time / **60.83** by wall time; 38 batches. Batch diagnostic false: 3 below 50, minimum 38.96. `Docs/Operations/evidence/email_core_closeout_13376/sustained_postgres.json` |
| Actual message/attachment counts, labels, sender/recipient pools and date span | 1,000,000 native/Media/version/identity/indexed-body rows; 200,000 attachments (20%); 23 labels; 200/500 participant pools; 365 days. `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite_parity.json` | Same full fresh 1M/200k/23/200/500/365-day shape and complete body/index/identity/version parity. `Docs/Operations/evidence/email_core_closeout_13376/million_postgres_parity.json` |
| Ten-class warm aggregate p50 / p95 and gate outcome | **Pass: 202.23 ms / 725.74 ms**, 200 measured samples; all ten populated, meaningful negation 166,667 -> 165,000. `Docs/Operations/evidence/email_core_closeout_13376/million_sqlite.json` | **Pass: 220.92/495.49 ms**, 200 samples/all ten populated/negation 166,667->165,000; `880d690a93`/23 stable hashes. `Docs/Operations/evidence/email_core_closeout_13376/million_postgres.json` |
| Cold-pass handle-reopen p50 / p95 and per-query diagnostics | 197.33 ms / 694.77 ms; fresh handles may reuse pooled physical connections and caches; per-case diagnostics retained separately | 281.54/502.76 ms; same pool/cache limits. Per-operator diagnostic false; documented aggregate gate true |
| Flag configuration, search/detail/media search, full-set parity, `auto_email`, rollback/restore | Pass: complete 7,900-ID parity; quoted title/body 1 each; `auto_email` 7,900; rollback retains all 7,900 legacy IDs, search/detail 404 and operator bridge 422; restored detail 200 and baseline flags | Exact 3,800/1/1 parity, `auto_email` 3,800 same IDs; all 3,800 legacy IDs retained, disabled 404/404/operator 422, restored detail 200/baseline flags |
| Live parse/dedupe/persistence/search metric observations | 8,000 parse/Media/native successes including retry; 100 dedupe matches; request/result/duration samples retained | 3,900 parse/Media/native successes including retry; 100 dedupe matches; 133 native searches (132 query-present/1 empty), duration/results retained |
| Zero outbound/model attempts, background interception, tenant isolation and cleanup | Zero outbound/model attempts; source unchanged across 19 recorded file hashes; other-user search empty/detail 404; exact generated SQLite roots confirmed absent in `Docs/Operations/evidence/email_core_closeout_13376/email_cleanup_roots_receipt_13376.json` | Both final guards zero/23 stable hashes; other search 0/detail 404/non-super/non-BYPASSRLS/forced RLS. Five generated targets catalog/roles 0/manifests absent; owned container/volume removed, shared service preserved; 55 exact roots absent. Receipts in main record |

Record hardware/RAM, run date, committed revision or modified-source identity,
exact commands, report links, final regression/Ruff/Bandit/review results and
remaining limitations with these values. Bulk fixture setup duration is reported
separately from HTTP ingestion throughput. Cold passes open fresh MediaDatabase
handles; pools may reuse physical connections and OS/server caches remain
unchanged. They do not prove cache-cold performance. Optional Gmail is deferred;
its missing live evidence is not an
open core technical gate. The owner's release decision below remains a separate
human record after the technical evidence is complete.

For an actual rollout record:

- Scope: core / optional Gmail
- Validation date and code revision:
- Environment, DB backend and dataset:
- Core test, parity and benchmark artifacts:
- Open gaps and disposition:
- Optional Gmail only: source ID, job IDs, cursor progression, monitoring window and final source state:

## Rollback Triggers

Core triggers: message loss/identity collisions, incorrect search/detail results,
tenant isolation failures, migration failures or media-search regressions.

Optional Gmail triggers: sustained failed/retrying state, unresolved invalid
cursors, or provider quotas/throttling that prevent reliable sync. A connector
failure calls for connector rollback; it does not automatically invalidate core uploads.

## Rollback Steps

1. Pause affected ingestion/sync activity and preserve evidence. For identity corruption, preserve a DB backup before any repair; feature flags do not restore lost metadata.
2. Return media delegation to `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in` if promotion caused a regression.
3. For a connector incident, set `EMAIL_GMAIL_CONNECTOR_ENABLED=false`. Disable `CONNECTORS_WORKER_ENABLED` only if stopping all connectors is intended.
4. For core failures, disable affected native write/search flags as needed (`EMAIL_NATIVE_PERSIST_ENABLED`, `EMAIL_OPERATOR_SEARCH_ENABLED`). Record that new legacy-only imports then need reconciliation before native search is re-enabled.
5. Restart affected processes and verify the remaining enabled surfaces with synthetic data. Legacy email search also needs correctness checks; it is not presumed safe from the dedupe defect.
6. Record the cause, affected scope, outcome and any required recovery/backfill.

The TASK-13376 local rehearsal reloads settings in its disposable app and verifies
rollback/restore over HTTP. It is evidence for flag behavior and data retention;
an actual deployment still follows the restart step above when its processes
read settings at startup.

## Owner Sign-off

- [ ] Core release approved
- [ ] Optional Gmail enablement approved (deferred)
- Name:
- Date:
- Environment / scope:
- Evidence and notes:
