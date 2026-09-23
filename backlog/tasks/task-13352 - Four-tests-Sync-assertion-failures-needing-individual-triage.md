---
id: TASK-13352
title: Four tests/Sync assertion failures needing individual triage
status: To Do
assignee: []
created_date: '2026-09-23 00:53'
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
- [ ] #1 Each of the four is classified test-drift or product defect, and fixed or filed
- [ ] #2 (1) and (2) are checked together -- both are the attachment.ref immutability path
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
