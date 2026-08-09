---
id: TASK-13003
title: Synchronize Notes keywords collections and folders
status: Done
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-09 21:30'
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
Completed first-class Sync v2 ownership for the indivisible six-domain Notes organization group. The implementation uses strict public payloads and deterministic resource/link identities, durable mutation-group append followed by resumable ordered product projection, readiness-gated bootstrap, device-aware pulls, conflict-aware materialization, dependency-ordered restore, and group-aware repair. Restore loads and validates each complete persisted group, rejects explicit filters that would split it, retains historical grouped siblings before superseding heads, and fails closed for incomplete or unsatisfiable dependency plans. For a live relationship dependency, restore uses a valid earlier provider inside the immutable group; otherwise it chooses only the latest eligible earlier external revision or, when none exists, the earliest eligible later revision. Compatible tombstones follow applicable live/link work, while an immutable historical tombstone group remains before later exact restores of a shared identity. Repair uses the same canonical full-envelope group validator as server-origin batch resume, resumes pending suffixes and singletons, and reports corrupt or blocked work only through stable safe fields. `Docs/API/Sync_V2_M1.md` documents the complete public contract, corrected SHA-256 vectors, bootstrap/repair behavior, ownership and provenance rules, and explicit non-goals.

| AC | Evidence |
| --- | --- |
| 1 | `test_sync_v2_models.py`, `test_sync_v2_endpoints.py`, and `Docs/API/Sync_V2_M1.md` verify all six versioned schemas and upsert/tombstone operations. |
| 2 | `test_sync_v2_notes_organization_adapters.py`, `test_sync_v2_notes_organization_materializer.py`, and `test_sync_v2_notes_organization_postgres_contract.py` cover identity, hierarchy, relationships, base state, and ownership for SQLite plus the PostgreSQL contract. |
| 3 | `test_sync_v2_server_origin_batch.py`, `test_sync_v2_server_origin_capture.py`, and `test_notes_organization_sync_api.py` verify durable canonical capture for direct and compound REST mutations. |
| 4 | `test_notes_organization_sync_api.py` and the M1-only compatibility cases in `test_sync_v2_server_origin_capture.py` verify ready-group writes and stable fail-closed behavior when the group is absent. |
| 5 | Adapter and API tests cover rename, hierarchy, membership, merge, delete/update, idempotent replay, and reviewable conflict paths. |
| 6 | SQLite-backed service/store tests in `test_sync_v2_restore_preview.py` cover all six domains, immutable groups, internal/earlier/later resource providers, multiple historical revisions, compatible tombstone-last ordering, historical tombstone groups before later exact restores, filters, missing dependencies, and genuine contradictions; `test_sync_v2_replay_repair.py` covers canonical stored-plan validation, pending/failed resume, safe blocked results, and exact-post-state bookkeeping. `Docs/API/Sync_V2_M1.md` records the matching public behavior. |
| 7 | `test_sync_v2_profile_bootstrap.py` and `test_sync_v2_notes_organization_bootstrap.py` verify all-or-none enrollment, source snapshot capture, interruption recovery, and readiness gates. |
| 8 | `test_sync_v2_service.py` verifies device-requested implicit pulls and excludes unsupported organization domains from legacy devices. |

Release-risk review found no accidental partial enrollment, product projection before durable group append, client-controlled bootstrap bypass, integer canonical identity, trusted remote origin provenance, flashcard movement, cascade soft-delete cleanup, ownership bypass, or organization-domain leakage to legacy devices. Cross-database behavior is documented as durable Sync append plus resumable product materialization, not atomicity across databases; folder source/manual effective membership follows ADR-033 suppression without deleting source provenance.

Verification used the eleven focused plan commands. Supported SQLite and mock-contract suites passed; the optional live PostgreSQL folder integration was explicitly skipped because no live DSN was configured (`21 passed, 1 skipped` in that command). Bandit passed with existing `nosec` annotations. The exact broad Ruff scope reported 20 pre-existing findings in unrelated legacy files; a supplemental Ruff check over every Task 10 touched code/test file passed. `git diff --check` passed. Public examples are synthetic and private-safe.

| Step 4 command | Exact result |
| --- | --- |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py` | 108 passed, 3 warnings. |
| `pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py` | Exit 0, 41 collected, summary truncated; cache warning observed. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_store.py -k "mutation_group or envelope"` | 28 passed, 37 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py` | Fresh review-fix authorized rerun: 48 passed, 7 warnings. Restricted precursor had 8 SQLite-permission failures plus 10 genuine validator-message compatibility failures; the latter were fixed and independently verified 10 passed/10 deselected. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py` | 69 passed, 4 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py` | 26 passed, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "notes_organization or implicit_pull or legacy_device"` | Fresh review-fix rerun: 2 passed, 92 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py` | 60 passed, 2 warnings on authorized rerun. |
| `pytest -q tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "note or keyword or collection or folder"` | 51 passed, 5 warnings on authorized rerun. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py -k notes_organization` | Fresh review-fix rerun: 4 passed, 29 deselected, 3 warnings. Full two-file compatibility gate: 33 passed, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py` | 21 passed, 1 optional live-PostgreSQL skip, 3 warnings. |

| Static/security command | Exact result |
| --- | --- |
| `ruff check tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/app/api/v1/schemas/notes_schemas.py` | Exit 1: 20 legacy findings, all outside review-fix touched lines; no unrelated cleanup. |
| `ruff check` over the seven review-fix production/test Python files | All checks passed. |
| `bandit -q -r tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/app/api/v1/endpoints/notes.py` | Exit 0; informational existing B608 `nosec` notices only. |
| `git diff --check` | Exit 0, no output. |

| Step 8 representative command | Exact result |
| --- | --- |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py` | 89 passed, 4 warnings. |
| `pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py` | 74 passed, 2 warnings on authorized SQLite rerun. |
| `git diff --check` (fresh, immediately before the Round 2 commit) | Exit 0, no output. |
| `git status --short` (fresh, immediately before the Round 2 commit) | Four scoped staged paths: this task file, `Docs/API/Sync_V2_M1.md`, `restore.py`, and `test_sync_v2_restore_preview.py`; no unrelated path. |

| Round 2 review-fix command | Exact result |
| --- | --- |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py -k provider_selection` | RED: 2 failed, 2 passed, 17 deselected, 4 warnings. GREEN: 4 passed, 17 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py` | 37 passed, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "notes_organization or implicit_pull or legacy_device"` | 2 passed, 92 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py` | Restricted run hit the eight known SQLite path-permission failures; exact authorized rerun: 48 passed, 7 warnings. |
| Touched-file Ruff | Restricted cache write failed; exact authorized rerun passed. |
| `bandit -q tldw_Server_API/app/core/Sync/v2/restore.py` | Exit 0, no output. |

Round 2 changes only the tested dependency-provider choice inside the existing restore ordering contract. It does not add a restore payload, schema, domain, authorization rule, dependency, or ADR. The repair documentation now states that `failed_only=true` includes pending-only groups and single envelopes, and that skipped or unmaterializable pending work keeps aggregate status at `repair_needed`.

| Round 3 review-fix command | Exact result |
| --- | --- |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py -k tombstone_graph` | RED: 2 failed, 3 passed, 21 deselected, 4 warnings. GREEN: 5 passed, 21 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py` | 42 passed, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "notes_organization or implicit_pull or legacy_device"` | 2 passed, 92 deselected, 3 warnings. |
| `pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py` | Authorized SQLite run: 48 passed, 7 warnings. |
| Step 8 representative command 1 | 89 passed, 4 warnings. |
| Step 8 representative command 2 | Authorized SQLite run: 74 passed, 2 warnings. |
| Touched-file Ruff | All checks passed. |
| `bandit -q tldw_Server_API/app/core/Sync/v2/restore.py` | Exit 0, no output. |
| Step 8 `git diff --check` immediately before the Round 3 commit | Exit 0, no output. |
| Step 8 `git status --short` immediately before the Round 3 commit | Four scoped staged paths: this task file, `Docs/API/Sync_V2_M1.md`, `restore.py`, and `test_sync_v2_restore_preview.py`; no unrelated path. |

Round 3 makes tombstone-last a compatible graph preference instead of a blanket reverse edge. The bounded unit graph retains immutable group and per-identity history precedence; a candidate live-to-tombstone edge is added only when it cannot close a tombstone-to-live path. This preserves unrelated and latest tombstone ordering without hiding genuine dependency cycles. No new schema, payload, domain, authorization rule, dependency, ADR, or support file was added.

ADR required: yes
ADR path: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md
Reason: This task implements the existing durable cross-database mutation-group contract, domain ownership boundaries, bootstrap policy, and long-lived Sync semantics. ADR-033 additionally governs canonical folder-link suppression while preserving local source provenance.
Related ADR path: Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md

No lesson was added: execution found routine stale test expectations and sandbox-only SQLite/cache permissions, with no new repository-general incident beyond the existing testing-evidence guidance.

Spec Fix Round 4/5 adds the public content-free ordered_actions restore plan while retaining safe_applies, object_conflicts, tombstones, counts, and existing aliases as compatibility category summaries. Each ordered row exposes only plan index, action kind, domain/object identity, operation, server cursor, optional immutable mutation-group ID/step/size, and optional stable conflict code. Restore preview now classifies canonical actions sequentially against an in-memory simulated inventory, so a historical tombstone followed by an exact live restore is apply rather than an unsafe noop; preview does not mutate product state. Real service/store and FastAPI coverage verifies historical grouped tombstones then restore, ordinary live/link work before compatible tombstones, stable plan/group metadata, safe noop/conflict rows, final simulated state, and the public field allow-list. Docs/API/Sync_V2_M1.md identifies ordered_actions as executable and legacy arrays as summaries, and includes a content-neutral synthetic example. RED: 2 failed, 26 deselected, 2 warnings. Final focused: 3 passed, 25 deselected, 2 warnings; restore/repair: 44 passed, 2 warnings; endpoint selectors: 4 passed, 53 deselected, 2 warnings; service selector: 2 passed, 92 deselected, 2 warnings; authorized capture/batch: 48 passed, 7 warnings; Step 8: 89 passed, 2 warnings and 74 passed, 2 warnings; touched Ruff passed; Bandit exited 0. No new ADR or general lesson required; ADR-032 and ADR-033 remain governing. Known non-blocking concerns remain the documented 20-finding broad Ruff legacy baseline and optional live PostgreSQL skip.
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
