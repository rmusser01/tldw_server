---
id: TASK-13344
title: Triage 12 red tests hidden behind the tests/Sync collection abort
status: Done
assignee: []
created_date: '2026-09-22 07:22'
updated_date: '2026-09-23 00:56'
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
- [x] #1 The three mixed-batch failures classified as product defect or fixture drift, with the verdict recorded
- [x] #2 All 12 names enumerated via --junit-xml, not a blind re-run
- [x] #3 Each of the 12 fixed or filed with a reason
- [x] #4 A gate-able tests/Sync subset identified with its runtime measured
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TRIAGE COMPLETE. All four acceptance criteria met. Follow-ups: TASK-13349, 13350, 13351, 13352.

AC2 -- ENUMERATION (and a correction). Run in a detached git worktree at HEAD so my working-tree edits could not disturb it, with --junit-xml. The count is 18, not 12: the original figure came from a run where 7 of the names were lost to the output buffer, so it was never a full enumeration. All 18 were then re-run SERIALLY and all 18 reproduce, in 42 seconds -- none is a parallelism artifact.

AC1 -- THE THREE MIXED-BATCH FAILURES: fixture drift, NOT a product defect. This overturns the task's hypothesis. Filed with the full evidence as TASK-13349. Summary: instrumenting the seven identity checks at personal_context_conflicts.py:194-202 shows six match exactly and only `remote is None` trips. _attach_candidate (:133-155) is the only path that creates a personal-context conflict, and it INSERTS the candidate envelope if absent before inserting the conflict -- so the product invariant is that such a conflict always has a stored remote candidate. The gate fixtures call store.insert_conflict directly with a fabricated remote_envelope_id that nothing ever creates, i.e. a state the product cannot produce. All five personal-context conflicts in that file hit `remote is None`; two are in tests that expect rejection, which is why only three fail. Timeline confirms it: fixtures 2026-09-04 (291ddc7ba6, 66945a5a9d), the requirement 2026-09-05 (c02afda9a7), which added a new 588-line test file but never updated this one. The connection-threading theory stays disproved -- the source lookup on the same connection succeeds.

AC3 -- ALL 18 CLASSIFIED:

  Fixture drift, product cannot produce the state (3) -> TASK-13349
    exchange_gate: test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order
    exchange_gate: test_mixed_exact_proof_preserves_native_notes_resolution_actions[overwrite]
    exchange_gate: test_mixed_exact_proof_preserves_native_notes_resolution_actions[duplicate_rename]

  Genuine product inconsistency the test caught (1) -> TASK-13350
    domain_adapters: test_default_sync_v2_registry_advertises_personal_and_workspace_metadata_domains
    factory.py:113-114 registers NotesTaskDomainAdapter and NotesTaskActivityDomainAdapter
    UNCONDITIONALLY, while SYNC_V2_SUPPORTED_DOMAINS (models.py:189) deliberately excludes them --
    they sit in SYNC_V2_KNOWN_DOMAINS instead. Compare attachment.ref, which IS env-gated in the
    same factory. So the registry advertises two domains the service's supported list omits.
    Deliberately NOT "fixed" by relaxing the assertion: that would silence the inconsistency.

  Needs Postgres, hard-fails instead of skipping (9) -> TASK-13351
    certification: test_postgres_two_connections_choose_exactly_one_existing_authority (PoolTimeout,
      and it burns 30s of suite runtime waiting for a pool that never arrives)
    certification: test_bootstrap_reuses_existing_nondefault_authority_without_creating_default
    conflicts: test_postgres_candidate_attachment_replay_and_retention[skip|overwrite|duplicate_rename]
    conflicts: test_stale_purge_after_failed_ingress_requires_refresh_and_reconfirmation[postgres-insertion|postgres-preflight]
    store: test_postgres_personal_context_receipt_locks_binding_before_upsert
    notes_organization_postgres_contract: test_postgres_predecessor_selector_uses_dataset_cursor_and_nonapplied_status
      (ran against SQLite, then compared Postgres dialect SQL)
    CLAUDE.md requires these to skip when the fixture reports Postgres unavailable. They do not.

  Individual assertion triage (5) -> TASK-13352
    domain_adapters: test_default_attachment_ref_adapter_rejects_invalid_parent_domain
    domain_adapters: test_default_attachment_ref_adapter_conflicts_divergent_stable_payload_hash
      (these two are probably one cause: the immutability error fires before parent-domain
       validation, and a divergent payload hash is REJECTED where a CONFLICT is expected --
       materially different, since rejection is terminal and conflict is reviewable)
    chat_materializer: test_message_metadata_write_failure_is_replayable_without_duplicate_rows
    server_origin_capture: test_workspace_chat_api_write_stays_direct_when_sync_active (404 vs 201)
    notes_attachment_bootstrap: test_cleanup_candidate_schema_rejects_path_hash_identity_drift
      (already known pre-existing; 'CHECK constraint failed' regex no longer matches)

AC4 -- RUNTIME. The premise that a CI remedy "must be a scoped subset or a nightly" turns out not to hold. The bottleneck is serial execution, not the tests:

    recorded serial baseline          3h 01m
    full directory, -n 4 --dist loadfile   5m 09s   (18 failed, 2972 passed)
    green subset, same flags           1m 08s   (2415 passed, 1 skipped, 0 failed)

  The green subset is the whole directory minus the nine files holding the 18 failures, i.e. 65 of
  74 files and 2415 tests, gate-able today at 69 seconds.

  --dist loadfile matters: it keeps each file on one worker, preserving within-file ordering. Two
  cross-checks that parallelism is not distorting the result: every one of the 18 reproduces
  serially, and the parallel run found MORE failures than the recorded serial run (18 vs 12), not
  fewer.

  NOT APPLIED, deliberately: shards run plain `pytest` with no -n (ci.yml:668-672), so adding
  xdist is a CI-wide change across every shard, and this repo already has one shard that exists
  precisely because its tree is parallel-unsafe (platform-mcp-inapp, cross-test state pollution).
  Recommending it per-shard is the owner's call.

  Also worth noting: tests/Sync is ALREADY in CI, inside the gap-verified-2 shard (ci.yml:703,
  2218) bundled with six other directories. So the directory is not uncovered -- the task's
  premise that "nobody runs this directory" is about local runs.

METHOD NOTE: the enumeration ran in a detached worktree at HEAD, removed afterwards, and the
source instrumentation used to diagnose AC1 was restored from a file copy, not git stash --
stashes are per-repository and this checkout has five worktrees.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
18 failures enumerated (not 12 -- the original count was truncated), all deterministic, all classified across four follow-up tasks. The three mixed-batch failures are fixture drift, not the product defect the task hypothesised: the fixtures build a conflict state _attach_candidate makes unreachable. One failure turned out to be a genuine product inconsistency the test was correctly catching. The 3h runtime is a serial-execution artifact: -n 4 --dist loadfile runs the whole directory in 5m09s and the green subset in 69s.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
