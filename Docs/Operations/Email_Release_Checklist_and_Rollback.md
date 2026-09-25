# Email Release Checklist and Rollback

Audience: Owner / maintainer
Status (2026-09-25): Local core validation recorded; chosen-deployment release gate open. Optional live Gmail validation deferred.

Related:
- `Docs/Product/Email_Ingestion_Search_PRD.md`
- `Docs/Operations/Email_Sync_Operations_Runbook.md`
- `Docs/Operations/Email_Core_Validation_2026-09-13.md`
- `Docs/Operations/Email_Local_Startup_and_Search_Performance_2026-09-25.md`
- `Docs/Operations/Env_Vars.md`

## Single-Owner Workflow

One maintainer owns implementation, operations and release approval. Record approval
for the scope actually enabled. Gmail is optional; unavailable Gmail access cannot
block file-based ingestion/search. All development/validation uses synthetic mail.
The owner's personal Gmail and personal email are excluded.

## Core File Ingestion and Search Gate

- [x] Trace upload parsing, persistence, indexing and optional model calls; record explicit offline options and interception evidence (TASK-13250).
- [x] Validate synthetic EML, ZIP and MBOX messages with distinct bodies, attachment metadata, repeat import, search and detail using temporary SQLite.
- [x] Resolve distinct-message same-body merging (TASK-13251). Strict EML/ZIP/MBOX regressions now require distinct stored/searchable identities; existing corrupted data requires separate recovery.
- [x] Implement FR-SEARCH-004 cursor pagination (TASK-13253); omitted cursor preserves offset behavior. See `Docs/Design/email-search-cursor-pagination.md` for traversal semantics.
- [x] Validate real API-key authentication and per-user SQLite isolation, including main-app route registration/test-mode request middleware (TASK-13255; `Docs/Operations/Email_Authenticated_Validation_2026-09-13.md`).
- [x] Validate authenticated synthetic EML uploads, key/role rejection, expected-user check, org storage quota and owner-scoped search in local ASGI/SQLite (TASK-13256; `Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md`).
- [x] Exercise the main FastAPI lifespan and scoped email search/detail plus media search with synthetic users, temporary SQLite and an outbound/model-call guard (TASK-13361; `Docs/Operations/Email_Local_Startup_and_Search_Performance_2026-09-25.md`).
- [ ] Validate chosen deployment/startup, production database backend and target scale. Local ASGI tests do not establish live server readiness or PostgreSQL isolation.
- [ ] Record actual performance/parity evidence for the intended cutover scope. The local 10,000-message synthetic SQLite benchmark does not certify the 1M-message target or production parity.
- [ ] Configure and verify core rollout flags in the chosen environment:
  - `EMAIL_NATIVE_PERSIST_ENABLED=true`
  - `EMAIL_OPERATOR_SEARCH_ENABLED=true`
  - `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=opt_in`
  - `EMAIL_GMAIL_CONNECTOR_ENABLED=false`
- [ ] Verify `GET /api/v1/email/search`, `GET /api/v1/email/messages/{id}`, and existing `POST /api/v1/media/search` behavior in that environment.
- [ ] If delegation promotion is in scope, validate `EMAIL_MEDIA_SEARCH_DELEGATION_MODE=auto_email` separately.
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

## Owner Sign-off

- [ ] Core release approved
- [ ] Optional Gmail enablement approved (deferred)
- Name:
- Date:
- Environment / scope:
- Evidence and notes:
