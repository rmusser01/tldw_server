---
id: TASK-13003
title: Synchronize Notes keywords collections and folders
status: Done
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-09 19:43'
labels:
  - notes
  - sync-v2
  - parity
  - organization
dependencies:
  - TASK-13002
references:
  - Docs/ADR/031-notes-capability-sync-domains.md
  - Docs/ADR/032-durable-server-origin-sync-mutation-batches.md
  - >-
    Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md
documentation:
  - Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md
  - Docs/API/Sync_V2_M1.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add first-class Sync v2 ownership for Notes keywords, keyword links and collections, folders, and note-folder membership so organization state survives offline and multi-device use instead of remaining REST-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capabilities advertise versioned notes.keyword, notes.keyword_link, notes.keyword_collection, notes.keyword_collection_link, notes.folder, and notes.folder_link domains with upsert/tombstone operations.
- [x] #2 Domain adapters and materializers preserve stable object identity, hierarchy, membership, optimistic base state, and user ownership on SQLite and PostgreSQL.
- [x] #3 Server-origin keyword, collection, folder, and membership REST mutations capture canonical envelopes when Sync v2 is active.
- [x] #4 Keyword writes no longer return the active-sync unsupported error only when all required organization domains are enabled for the dataset.
- [x] #5 Concurrent rename, hierarchy, membership, merge, and delete/update cases produce deterministic idempotent results or reviewable conflicts.
- [x] #6 Restore preview, repair, and capability documentation include every organization domain.
- [x] #7 Existing personal datasets bootstrap their current organization state before the six-domain group becomes write-ready, and interrupted bootstrap resumes without partial enrollment or data loss.
- [x] #8 Upgrading one device does not deliver unsupported organization domains to legacy devices whose registered requested domains exclude them.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed execution plan: Docs/superpowers/plans/2026-08-08-notes-organization-sync-implementation-plan.md

1. Define the six-domain public contract, strict payload schemas, and deterministic identities.
2. Migrate ChaChaNotes to schema v55 and add stable resource identities plus a focused projection seam, including canonical folder-link suppression that preserves local source provenance.
3. Add atomic mutation-group persistence to the Sync store.
4. Add batch preflight, durable append, ordered materialization, and resume semantics.
5. Implement strict organization adapters and conflict policy.
6. Implement SQLite/PostgreSQL materializers and production factory registration.
7. Bootstrap existing datasets and isolate legacy-device implicit pulls.
8. Route direct organization REST mutations through the coordinator.
9. Make compound note writes, effective folder provenance, and keyword merge lossless.
10. Integrate restore/repair, update docs, and record focused release evidence.

ADR required: yes
ADR paths: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md; Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md
Reason: This task adds durable cross-database mutation groups and a local derived suppression projection so canonical folder-link absence converges without deleting source-ingestion provenance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed first-class Sync v2 ownership for the indivisible six-domain Notes organization group. The implementation uses strict public payloads and deterministic resource/link identities, durable mutation-group append followed by resumable ordered product projection, readiness-gated bootstrap, device-aware pulls, conflict-aware materialization, dependency-ordered restore, and group-aware repair. `Docs/API/Sync_V2_M1.md` now documents the complete public contract, corrected SHA-256 vectors, bootstrap/repair behavior, ownership and provenance rules, and explicit non-goals.

| AC | Evidence |
| --- | --- |
| 1 | `test_sync_v2_models.py`, `test_sync_v2_endpoints.py`, and `Docs/API/Sync_V2_M1.md` verify all six versioned schemas and upsert/tombstone operations. |
| 2 | `test_sync_v2_notes_organization_adapters.py`, `test_sync_v2_notes_organization_materializer.py`, and `test_sync_v2_notes_organization_postgres_contract.py` cover identity, hierarchy, relationships, base state, and ownership for SQLite plus the PostgreSQL contract. |
| 3 | `test_sync_v2_server_origin_batch.py`, `test_sync_v2_server_origin_capture.py`, and `test_notes_organization_sync_api.py` verify durable canonical capture for direct and compound REST mutations. |
| 4 | `test_notes_organization_sync_api.py` and the M1-only compatibility cases in `test_sync_v2_server_origin_capture.py` verify ready-group writes and stable fail-closed behavior when the group is absent. |
| 5 | Adapter and API tests cover rename, hierarchy, membership, merge, delete/update, idempotent replay, and reviewable conflict paths. |
| 6 | `test_sync_v2_restore_preview.py`, `test_sync_v2_replay_repair.py`, and `Docs/API/Sync_V2_M1.md` cover all six domains, ordered restore, safe resumable group repair, and exact-post-state bookkeeping. |
| 7 | `test_sync_v2_profile_bootstrap.py` and `test_sync_v2_notes_organization_bootstrap.py` verify all-or-none enrollment, source snapshot capture, interruption recovery, and readiness gates. |
| 8 | `test_sync_v2_service.py` verifies device-requested implicit pulls and excludes unsupported organization domains from legacy devices. |

Release-risk review found no accidental partial enrollment, product projection before durable group append, client-controlled bootstrap bypass, integer canonical identity, trusted remote origin provenance, flashcard movement, cascade soft-delete cleanup, ownership bypass, or organization-domain leakage to legacy devices. Cross-database behavior is documented as durable Sync append plus resumable product materialization, not atomicity across databases; folder source/manual effective membership follows ADR-033 suppression without deleting source provenance.

Verification used the eleven focused plan commands. Supported SQLite and mock-contract suites passed; the optional live PostgreSQL folder integration was explicitly skipped because no live DSN was configured (`21 passed, 1 skipped` in that command). Bandit passed with existing `nosec` annotations. The exact broad Ruff scope reported 20 pre-existing findings in unrelated legacy files; a supplemental Ruff check over every Task 10 touched code/test file passed. `git diff --check` passed. Public examples are synthetic and private-safe.

ADR required: yes
ADR path: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md
Reason: This task implements the existing durable cross-database mutation-group contract, domain ownership boundaries, bootstrap policy, and long-lived Sync semantics. ADR-033 additionally governs canonical folder-link suppression while preserving local source provenance.
Related ADR path: Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md

No lesson was added: execution found routine stale test expectations and sandbox-only SQLite/cache permissions, with no new repository-general incident beyond the existing testing-evidence guidance.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Notes organization and Sync v2 suites pass on supported database backends.
- [x] #8 Bandit and static checks pass for touched production files.
- [x] #9 Public schemas and examples contain no note content or secret-bearing fixtures.
<!-- DOD:END -->
