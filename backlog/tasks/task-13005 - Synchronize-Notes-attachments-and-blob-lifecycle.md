---
id: TASK-13005
title: Synchronize Notes attachments and blob lifecycle
status: In Progress
assignee: []
created_date: '2026-08-08 20:24'
updated_date: '2026-08-11 15:26'
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
- [ ] #1 Capabilities advertise Notes use of the versioned attachment.ref domain and negotiated blob-transfer support.
- [ ] #2 Attachment references preserve stable identity note ownership filename media type size digest lifecycle state and optimistic base state across SQLite and PostgreSQL.
- [ ] #3 Upload list download insert delete and restore operations capture canonical metadata envelopes while binary transfer verifies content integrity and supports safe resume.
- [ ] #4 Missing corrupt oversized unauthorized or quarantined blobs produce explicit recoverable states without losing the owning note or attachment metadata.
- [ ] #5 Concurrent attachment rename replace delete restore and note deletion yield idempotent results or reviewable conflicts with garbage collection evidence.
- [ ] #6 Attachment APIs and sync paths enforce dataset ownership path safety content limits and secret-safe diagnostics.
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes attachment and shared blob-transfer suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Integrity resume limit authorization and garbage-collection scenarios have automated evidence.
<!-- DOD:END -->
