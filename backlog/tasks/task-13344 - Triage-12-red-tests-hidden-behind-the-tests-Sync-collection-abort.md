---
id: TASK-13344
title: Triage 12 red tests hidden behind the tests/Sync collection abort
status: To Do
assignee: []
created_date: '2026-09-22 07:22'
updated_date: '2026-09-22 17:24'
labels:
  - bug
  - sync
  - tests
dependencies: []
references:
  - tldw_Server_API/tests/Sync/test_sync_v2_personal_context_exchange_gate.py
  - 'tldw_Server_API/app/core/Sync/v2/personal_context_conflicts.py:203'
  - 'tldw_Server_API/tests/Sync/test_sync_v2_store.py:3036'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A full tests/Sync run (3h 01m, after the collection abort was fixed under TASK-13306) reports 12 failed, 2908 passed, 38 skipped. Five names recovered; all five reproduce in isolation, so they are deterministic.

HIGHEST VALUE - likely a real product defect, not fixture rot:
test_sync_v2_personal_context_exchange_gate.py, 3 cases (test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order, and both parametrisations of test_mixed_exact_proof_preserves_native_notes_resolution_actions). All assert ["mixed-exact-note"] == ["mixed-exact-note","mixed-exact-personal"]: a MIXED notes/personal-context batch returns only the notes item, silently dropping the personal-context resolution. EVERY failing case is mixed_*; the pure personal-context cases pass (95 passed). Reproduces in 7s.

Real cause, now visible because TASK-13306 made the swallow log it:
  SyncStoreError: "Personal Context conflict candidate is unavailable"
  raised at core/Sync/v2/personal_context_conflicts.py:203 when the source/remote envelope
  lookup returns None or one of the identity checks at :194-202 mismatches.
DISPROVED already, do not re-derive: a connection-threading cause. Both store.get_envelope_by_server_cursor (store.py:1407) and store.get_envelope_by_client_id (store.py:1413) DO pass connection=self._connection. The identity checks at :194-202 are the remaining candidate.

OTHERS RECOVERED:
- test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert - link_state fixture drift, red since 2026-09-03 (8c97f181e5).
- test_sync_v2_server_origin_capture.py::test_workspace_chat_api_write_stays_direct_when_sync_active - assert 404 == 201, last touched 2026-08-10.

The remaining seven names were lost to the output buffer; re-run with --junit-xml or -rf to recover them rather than another 3-hour blind run.

RUNTIME IS PART OF THE PROBLEM: 3h 01m is why nobody runs this directory. Any CI remedy must be a scoped subset or a nightly, not the full command in a PR gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three mixed-batch failures classified as product defect or fixture drift, with the verdict recorded
- [ ] #2 All 12 names enumerated via --junit-xml, not a blind re-run
- [ ] #3 Each of the 12 fixed or filed with a reason
- [ ] #4 A gate-able tests/Sync subset identified with its runtime measured
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#2 SATISFIED - all 12 names recovered via --junit-xml (2960 tests, 12 failures, 0 ERRORS, 40 skipped, 10772s = 2.99h). Zero errors confirms the importorskip fix from TASK-13306 worked: the directory now runs end to end.

COMPLETE LIST, grouped by file. Every one REPRODUCES IN ISOLATION, so all 12 are deterministic, not ordering artifacts:

test_sync_v2_domain_adapters (3 failed / 48 passed in 0.96s) - a cluster, cheapest to start with:
  test_default_attachment_ref_adapter_conflicts_divergent_stable_payload_hash
  test_default_attachment_ref_adapter_rejects_invalid_parent_domain
  test_default_sync_v2_registry_advertises_personal_and_workspace_metadata_domains

test_sync_v2_personal_context_exchange_gate (3 failed / 95 passed in 7s) - diagnosed, see below:
  test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order
  test_mixed_exact_proof_preserves_native_notes_resolution_actions[overwrite]
  test_mixed_exact_proof_preserves_native_notes_resolution_actions[duplicate_rename]

test_sync_v2_personal_context_certification (2 failed / 24 passed in 71s in ISOLATION, but only 1 of them failed in the full run - order-dependent in the OPPOSITE direction, worth a look on its own):
  test_bootstrap_reuses_existing_nondefault_authority_without_creating_default

test_sync_v2_chat_materializer (1 failed / 11 passed in 8.9s):
  test_message_metadata_write_failure_is_replayable_without_duplicate_rows

test_sync_v2_notes_attachment_bootstrap (1 failed / 30 passed in 10.7s):
  test_cleanup_candidate_schema_rejects_path_hash_identity_drift

test_sync_v2_notes_organization_postgres_contract:
  test_postgres_predecessor_selector_uses_dataset_cursor_and_nonapplied_status

test_sync_v2_server_origin_capture:
  test_workspace_chat_api_write_stays_direct_when_sync_active  (assert 404 == 201)

test_sync_v2_store:
  test_postgres_personal_context_receipt_locks_binding_before_upsert  (link_state fixture drift, red since 2026-09-03)

SUGGESTED ORDER: domain_adapters first - 3 of the 12 in one file that runs in under a second.

The exchange-gate three already have their cause: SyncStoreError "Personal Context conflict candidate is unavailable" from personal_context_conflicts.py:203, visible because TASK-13306 made the swallow log it. Connection-threading was hypothesised and DISPROVED (both lookups thread connection=self._connection); the identity checks at :194-202 remain the candidate.

AC#4 INPUT: the full directory is 2.99h, so it cannot be a PR gate. The four files above total ~92s and cover 7 of the 12 - a plausible seed for a gate-able subset.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
