---
id: TASK-13352
title: Four tests/Sync assertion failures needing individual triage
status: Done
assignee: []
created_date: '2026-09-23 00:53'
updated_date: '2026-09-23 01:35'
labels:
  - tests
  - sync
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enumerated by junit-xml under TASK-13344. All deterministic, all reproduce serially, none need Postgres. Each needs a test-versus-product judgement.

1. test_sync_v2_domain_adapters.py::test_default_attachment_ref_adapter_rejects_invalid_parent_domain
     assert 'attachment_ref_v1_immutable' == 'attachment_ref_parent_domain_invalid'
     The adapter returns the immutability error before it validates the parent domain, so the
     more specific diagnosis is masked. Decide whether the ordering or the expectation is wrong.

2. test_sync_v2_domain_adapters.py::test_default_attachment_ref_adapter_conflicts_divergent_stable_payload_hash
     isinstance(AdapterRejected(error_code='attachment_ref_v1_immutable'), AdapterConflict) is False
     A divergent payload hash is being REJECTED where the test expects a CONFLICT. Those are
     materially different outcomes for sync -- a rejection is terminal, a conflict is reviewable.
     Same immutability path as (1), so the two are probably one cause.

3. test_sync_v2_chat_materializer.py::test_message_metadata_write_failure_is_replayable_without_duplicate_rows
     get_message_by_id('msg-1') returns None, so the replay assertion has nothing to check.
     Either the materializer no longer writes the row on this path, or the fixture id changed.

4. test_sync_v2_server_origin_capture.py::test_workspace_chat_api_write_stays_direct_when_sync_active
     assert 404 == 201. TASK-13344 notes the file was last touched 2026-08-10. A 404 suggests the
     route moved or the workspace fixture no longer resolves, not that the write was refused.

ALSO, already known and separately tracked as pre-existing: test_sync_v2_notes_attachment_bootstrap.py::test_cleanup_candidate_schema_rejects_path_hash_identity_drift expects a 'CHECK constraint failed' regex that no longer matches.

Source: TASK-13344 AC3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four is classified test-drift or product defect, and fixed or filed
- [x] #2 (1) and (2) are checked together -- both are the attachment.ref immutability path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE in 31795b4d50 for four of the five; the fifth is deliberately left red. tests/Sync goes 18 failures to 14 (re-measured across the whole directory).

Every one was test drift -- the product was right in all four cases.

1+2. attachment.ref, and they were indeed one cause as this task suspected.
   Both asserted v1-era outcomes: a dedicated attachment_ref_parent_domain_invalid code,
   and an AdapterConflict on a divergent payload hash. 41c097bbcc (2026-08-11) made
   adapter version 1 immutable, and that check is the FIRST thing evaluate_envelope does
   -- three months after these tests were written (0e35487726, 2026-05-23). The test
   helper _attachment_ref_envelope still defaults to adapter_version 1, so a v1 write is
   refused whatever the payload says and neither outcome is reachable.
   Rewritten to pin the ordering itself, which is the deliberate design. The v2
   equivalents already have coverage -- 40 tests in test_sync_v2_attachment_refs.py --
   so nothing was lost: parent_domain is enforced by the v2 schema as
   Literal["notes.note"], and a divergent hash is REJECTED against the canonical object
   hash rather than raised as a reviewable conflict. That difference is material (a
   rejection is terminal) and is now recorded in the test rather than silently changed.

3. chat materializer. Asserted the message row SURVIVED a failed metadata write, which
   was true when row and metadata were two separate writes. They are now one
   transaction: the failure logs "Transaction (outermost) failed, rolling back on
   thread ... Failed to persist Sync v2 metadata for message msg-1". Confirmed
   empirically by probing the row rather than reasoning about it -- with that single
   assertion relaxed, every other assertion in the test passed, including
   count_messages_for_conversation == 1 on retry and the correct metadata.
   The test now asserts the ROLLBACK, which is strictly stronger than what it was
   defending: there is no half-written row for the replay to reconcile at all.

4. server_origin_capture, 404 vs 201. The cause was not a moved route -- the route
   exists and sibling tests use the same path. The body said "Workspace not found":
   workspace-scoped creation now resolves Workspace Persona defaults first
   (Workspaces/assistant_defaults.py:19-21) and 404s on a workspace that does not exist.
   The test posted workspace_id="workspace-1" without ever creating it.
   Fixing that peeled two further layers, each a fake that had outlived its subject:
     - add_conversation had gained a threaded `conn`, so the fake raised TypeError
     - then a FOREIGN KEY failed, because the workspace-scoped path now takes the direct
       branch through create_character_conversation -- which is exactly what "stays
       direct" MEANS and what this test asserts -- and that branch writes participant
       rows keyed on the conversation, which existed only in a dict.
   Resolution: drop the add_conversation/get_conversation_by_id/upsert_conversation_settings
   fakes and let the real write happen. Simpler, and a truer test of the branch under
   test. Stubbing get_workspace alone was tried first and is NOT sufficient -- it clears
   the 404 and then fails the foreign key, because conversations.workspace_id references
   workspaces(id). A real row is required.

5. NOT FIXED, deliberately: test_cleanup_candidate_schema_rejects_path_hash_identity_drift
   was listed here as already-known pre-existing. It is still red and still out of scope
   for this task.

ALSO STILL RED BY DESIGN: test_default_sync_v2_registry_advertises_personal_and_workspace_
metadata_domains, in one of the files touched here. That test is doing its job -- the
registry advertises notes.task domains that SYNC_V2_SUPPORTED_DOMAINS omits -- and
relaxing the assertion would silence a real product inconsistency. TASK-13350 owns it.

Verification: the three touched files run 95 passed / 1 failed (that one being TASK-13350).
Whole directory re-run with -n 4 --dist loadfile: 14 failed / 2976 passed, down from 18,
and the four that disappeared are exactly these.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Four of the listed failures fixed; all four were test drift with the product correct. The two attachment.ref ones shared a cause as suspected: v1 immutability short-circuits checks they predate. The chat materializer assertion described a pre-atomic write. The 404 was a new workspace-existence check, behind which sat two more stale fakes. tests/Sync: 18 failures to 14.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
