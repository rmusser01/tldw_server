---
id: TASK-13005
title: Synchronize Notes attachments and blob lifecycle
status: Done
assignee: []
created_date: '2026-08-08 20:24'
updated_date: '2026-08-13 20:31'
labels:
  - notes
  - sync-v2
  - parity
  - attachments
dependencies:
  - TASK-13004
references:
  - Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md
documentation:
  - >-
    Docs/superpowers/specs/2026-08-11-notes-attachment-sync-and-blob-lifecycle-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Notes attachment metadata and binary content participate in the same offline and multi-device lifecycle as their owning notes, using the existing attachment reference and resumable blob-transfer contracts rather than a separate Notes-only transport.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities advertise Notes use of the versioned attachment.ref domain and negotiated blob-transfer support.
- [x] #2 Attachment references preserve stable identity note ownership filename media type size digest lifecycle state and optimistic base state across SQLite and PostgreSQL.
- [x] #3 Upload list download insert delete and restore operations capture canonical metadata envelopes while binary transfer verifies content integrity and supports safe resume.
- [x] #4 Missing corrupt oversized unauthorized or quarantined blobs produce explicit recoverable states without losing the owning note or attachment metadata.
- [x] #5 Concurrent attachment rename replace delete restore and note deletion yield idempotent results or reviewable conflicts with garbage collection evidence.
- [x] #6 Attachment APIs and sync paths enforce dataset ownership path safety content limits and secret-safe diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Execute TASK-13005.1 using Docs/superpowers/plans/2026-08-11-notes-attachment-sync-contract-persistence-implementation-plan.md.
2. Execute TASK-13005.2 after TASK-13005.1 using Docs/superpowers/plans/2026-08-11-notes-attachment-sync-mutation-lifecycle-implementation-plan.md.
3. Execute TASK-13005.3 after TASK-13005.1 and TASK-13005.2 using Docs/superpowers/plans/2026-08-11-notes-attachment-sync-legacy-bootstrap-implementation-plan.md.
4. Execute TASK-13005.4 after TASK-13005.1, TASK-13005.2, and TASK-13005.3 using Docs/superpowers/plans/2026-08-11-notes-attachment-sync-restore-operations-implementation-plan.md.
5. Run the aggregate verification and complete the parent only after all four children are Done.

ADR required: yes
ADR path: Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md
Reason: durable schema, Sync versioning, tenancy, API, restore, and blob-deletion authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the ADR-038 work stream through all four atomic child tasks, all now Done.

- TASK-13005.1 established strict attachment.ref v2 negotiation, schema-v59 SQLite/PostgreSQL registry/RLS, immutable revision bindings, versioned cursors/acks, per-dataset namespaces, and default-off capability gating.
- TASK-13005.2 made canonical create/list/content/rename/replace/delete/restore and legacy aliases use one owner-bound Sync mutation authority with idempotency, optimistic ETags, verified blob integrity, and fail-closed gates.
- TASK-13005.3 source-verified and resumably imported legacy Notes attachments without deleting rollback evidence, with bounded privacy-safe bootstrap diagnostics.
- TASK-13005.4 completed dependency-safe restore, monotonic historical binding release, fenced physical blob deletion, read-only lifecycle diagnostics, public contracts, and lifecycle/range e2e coverage.

Architecture: existing ADR-038 governs the full schema, ownership, API, restore, namespace, and garbage-collection contract; no additional ADR was required. Public M1/M2/M3 and ADR documentation now reflect the implemented behavior.

Aggregate evidence recorded by the children includes: live PostgreSQL/affected 178 passed, schema migration/store/RLS 116 passed, broad Sync 571 passed for the foundation; mutation matrix 464 passed with 4 optional PG skips; bootstrap gate 651 passed with 2 optional PG skips; restore/operations affected gate 506 passed with 3 optional PG skips and final boundary gate 32 passed with 3 skips. All server-free PostgreSQL catalog/query contracts and SQLite integrations are green; live PostgreSQL fixtures are committed and skip only when a server is unavailable. Ruff/static, Bandit, py_compile, diff checks, security/authorization, integrity, resume, size-limit, corruption/quarantine, concurrency, retention, and physical-GC scenarios are recorded across the child tasks. No new lessons file was invented: no additional general repository trap remained after applying the existing testing/live-verification guidance.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Notes attachments now participate in the same stable-ID, multi-device Sync v2 lifecycle as their owning notes: negotiated metadata, verified resumable bytes, canonical REST mutations, safe legacy bootstrap, dependency-complete restore, immutable acknowledgment evidence, fenced retention/physical collection, and privacy-safe diagnostics. TASK-13005.1 through TASK-13005.4 are all Done, ADR-038 and public contracts are current, and the aggregate regression/security evidence is recorded with environment-only PostgreSQL skips called out.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Notes attachment and shared blob-transfer suites pass on supported database backends.
- [x] #8 Bandit and static checks pass for touched production files.
- [x] #9 Integrity resume limit authorization and garbage-collection scenarios have automated evidence.
<!-- DOD:END -->
