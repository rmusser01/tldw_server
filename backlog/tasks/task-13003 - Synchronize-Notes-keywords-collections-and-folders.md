---
id: TASK-13003
title: Synchronize Notes keywords collections and folders
status: Done
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-10 18:26'
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
  - Docs/ADR/034-web-clipper-external-identity-mapping.md
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
2. Migrate ChaChaNotes and add stable resource identities plus a focused projection seam.
3. Add atomic mutation-group persistence to the Sync store.
4. Add batch preflight, durable append, ordered materialization, and resume semantics.
5. Implement strict organization adapters and conflict policy.
6. Implement SQLite/PostgreSQL materializers and production factory registration.
7. Bootstrap existing datasets and isolate legacy-device implicit pulls.
8. Route every reachable Notes organization mutation surface through the coordinator.
9. Make compound note writes, folder provenance, keyword merge, and conflict recovery lossless.
10. Integrate restore/repair, bound concurrency, complete tenant-safe schema migrations, update docs, and record release evidence.

ADR required: yes
ADR paths: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md; Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md; Docs/ADR/034-web-clipper-external-identity-mapping.md
Reason: This task defines durable mutation-group concurrency, Notes organization ownership/RLS, folder-source suppression, and the WebClipper external-to-canonical identity boundary.
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

Spec Fix Round 5/5 makes the public global ordered_actions plan unambiguous across datasets by requiring safe dataset_id on every core and API action, sourced directly from each stored envelope. Restore preview now performs a dataset-scoped bounded final-head preflight before sequential simulation: when original local inventory exactly matches the final planned state for an identity, earlier historical live revisions and their immutable dependent group siblings remain executable applies and the final head is reapplied; a divergent local fingerprint remains a conflict. Compatibility arrays, counts, aliases, group order, and non-mutating preview behavior are preserved. Real service/store and FastAPI tests cover two datasets sharing the same domain/object ID, exact public allow-list/privacy, a historical keyword-plus-link group before a later keyword update, final simulated state, group adjacency/metadata, and retained genuine conflicts. Docs/API/Sync_V2_M1.md documents dataset disambiguation and matching-final-head sequential semantics. RED: 2 failed, 28 deselected, 2 warnings. GREEN: 2 passed, 28 deselected, 2 warnings; full restore/repair: 46 passed, 2 warnings; endpoint selectors: 6 passed, 53 deselected, 2 warnings; service selector: 2 passed, 92 deselected, 2 warnings; authorized capture/batch: 48 passed, 7 warnings; Step 8: 89 passed, 2 warnings and 74 passed, 2 warnings; touched Ruff passed; Bandit exited 0. No new ADR or general lesson required; ADR-032 and ADR-033 remain governing. Known non-blocking concerns remain the documented 20-finding broad Ruff legacy baseline and optional live PostgreSQL skip.

Code Quality Fix Round 1 makes restore preview fail closed before simulated state advances: a divergent tombstone is a blocking conflict, while a tombstone matching its explicit live base remains executable; stored accepted `apply_status=conflict` envelopes and their complete groups remain visible through the safe `sync_restore_stored_apply_conflict` ordered action instead of falling back to an older head. Client timestamps are normalized to canonical UTC at model, database-row, idempotency, and immutable group-hash boundaries so PostgreSQL-native aware datetimes preserve fingerprints. Restore ordering now uses a bounded indegree topological sort with hard chronology/dependency/group edges and deterministic compatible-live ready priority. Global ceilings are 50,000 preview candidates, 10,000 actions, and 1,000 group members with safe HTTP 413 codes. Repair's `limit` is documented and tested as a complete-unit soft limit, with the 1,000-member group ceiling and a cursor covering every admitted member. Focused RED/GREEN: I1 2 failed then 2 passed; I2 1 failed then 1 passed; I3 2 failed then 2 passed; I4 preview 3 failed then 3 passed and repair 2 failed then 2 passed. Final gates: restore/repair 55 passed; store 66 passed; endpoint selectors 12 passed; service selector 2 passed; capture/batch 48 passed; PostgreSQL contracts 21 passed with 1 optional live-DSN skip; Step 8 89 passed and 74 passed; exact touched Ruff passed; exact touched-production Bandit exited 0 with existing informational `nosec` notices only. `Docs/API/Sync_V2_M1.md` records the public behavior and limits. Compatibility fields, ordered-action privacy, group adjacency, and preview's no-product-mutation guarantee remain unchanged. No new ADR or general lesson was required; ADR-032 and ADR-033 remain governing. Known non-blocking concerns remain the documented broad Ruff legacy baseline and optional live PostgreSQL skip.

Code Quality Fix Round 2 preserves genuine pre-normalization immutable group hashes without weakening any non-timestamp fingerprint field. Validation accepts the canonical hash, the exact SQLite-stored timestamp spelling, and the exact historical UTC `Z` spelling reconstructible from a PostgreSQL-native UTC timestamp; arbitrary non-UTC PostgreSQL lexemes remain unrecoverable after `TIMESTAMPTZ` normalization and are not guessed. One shared 1,000-member constant now rejects oversized atomic appends before writes and bounds group reads to 1,001 rows before model construction, with the existing safe 413 code preserved through restore and repair. Dependency-provider selection pre-sorts live occurrences per identity and uses cursor bisect instead of repeated full scans. Public restore-preview requests cap dataset/domain lists at 100 and selected object/attachment/local-inventory lists at 10,000; duplicate dataset IDs are deduplicated in first-seen order. RED/GREEN: timestamp compatibility 2 failed then 4 passed after the PostgreSQL `Z` regression; group ceiling 2 failed then 2 passed; provider operation count 1 failed then 1 passed; schema/dedup 6 failed then 6 passed. Final gates: restore/repair 61 passed; store 71 passed; endpoint selectors 12 passed; service selector 2 passed; capture/batch 48 passed; PostgreSQL contracts 21 passed with 1 optional live-DSN skip; Step 8 89 passed and 74 passed; exact Ruff over all nine touched Python files passed; Bandit over all six touched production Python files exited 0 with the existing informational `nosec` notice only. `Docs/API/Sync_V2_M1.md` records compatibility scope and bounds. Compatibility response fields, ordered-action privacy, group adjacency, and preview's no-product-mutation guarantee remain unchanged. The predecessor review's “10-file” statement was checked against git and the Round 1 commit actually changed 12 tracked files (two documentation/task files, seven production Python files, and three tests). No new ADR or general lesson was required; ADR-032 and ADR-033 remain governing. Known non-blocking concerns remain the documented broad Ruff legacy baseline and optional live PostgreSQL skip.

Final merge review reopened TASK-13003: concurrent server-origin groups can preflight the same head and append/materialize out of order, object-state advancement is non-monotonic, and additional public Notes write paths remain uncaptured. AC3/AC5 and the affected Definition of Done gates stay open until the final fix wave is implemented and verified.

Final merge-review wave completed AC3 and AC5. Every audited public/background Notes organization writer now resolves an explicit owner and routes active Sync writes through the readiness-gated coordinator; inactive behavior remains compatible and active incomplete/error states fail closed before product writes. Dataset-serialized append CAS, cursor-ordered projection, durable conflict bookkeeping, bounded repair/rebase handling, canonical restore actions, and superseded terminal state make concurrent mutations deterministic or reviewable. ChaChaNotes schema v56 maps owner-scoped WebClipper clip IDs to private UUID note identities with transactional rollback, endpoint-aware RLS, exact-retry recovery, and fail-closed history/payload checks. Schema v57 scopes Notes organization uniqueness per PostgreSQL tenant, validates cross-owner graphs under approved transaction-held locks, temporarily suspends and restores only verified FORCE-RLS tables, and reinstalls the complete policy set after schema ensure. ADR-032, ADR-033, and ADR-034 govern these contracts.

Final evidence: authoritative Sync selection collected 555; restricted execution had 547 passes plus 8 SQLite-path permission failures, and the exact authorized eight reran 8/8. Organization/bootstrap/API/materializer gate passed 275. Final combined v56/v57/RLS and WebClipper gate passed 156; independent final migration/WebClipper review passed 138 with one optional live-PostgreSQL tenancy skip because no DSN was configured. Bandit over every changed production Python file exited 0. Focused Ruff and focused Ruff formatting passed; ChaChaNotes passed with its documented legacy-code exclusions, while unfiltered statistics remain 16 unrelated baseline findings. git diff --check passed. Independent final review found no remaining Critical, Important, or Minor issue. No new general lesson was added: the incidents were specific to the new migrations and are captured in ADR-033/034 plus regression tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and hardened first-class Sync v2 ownership for Notes organization data across REST/background capture, atomic mutation groups, concurrent projection/conflict recovery, bootstrap, restore/repair, PostgreSQL tenancy, and WebClipper identity migration. SQLite and server-free PostgreSQL contract suites are green; the only environment limitation is one optional live-PostgreSQL tenancy test skipped without a configured DSN.
<!-- SECTION:FINAL_SUMMARY:END -->

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
