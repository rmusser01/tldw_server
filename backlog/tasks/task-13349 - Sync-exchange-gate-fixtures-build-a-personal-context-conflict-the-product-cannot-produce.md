---
id: TASK-13349
title: >-
  Sync exchange-gate fixtures build a personal-context conflict the product
  cannot produce
status: To Do
assignee: []
created_date: '2026-09-23 00:51'
updated_date: '2026-09-23 02:33'
labels:
  - bug
  - sync
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three tests fail because their fixtures fabricate a conflict state that no product path can create. VERDICT: fixture drift, NOT a product defect. The task that surfaced these (TASK-13344) hypothesised a product defect; this disproves it.

FAILING (all deterministic, reproduce in ~2s each):
  tests/Sync/test_sync_v2_personal_context_exchange_gate.py
    test_mixed_selected_conflicts_with_exact_proof_resolve_in_request_order
    test_mixed_exact_proof_preserves_native_notes_resolution_actions[overwrite]
    test_mixed_exact_proof_preserves_native_notes_resolution_actions[duplicate_rename]

EVIDENCE. Instrumenting the seven identity checks at core/Sync/v2/personal_context_conflicts.py:194-202 shows six match exactly and only one trips:

  {'source_is_none': False, 'remote_is_none': True,
   'dataset_id': ('dataset-a','dataset-a'), 'device_id': ('device-a','device-a'),
   'client_envelope_id': ('client-envelope-mixed-exact-personal', <same>),
   'object_id': ('record-mixed-exact-personal', <same>),
   'domain': ('personal_context.record', <same>),
   'expected_remote_envelope_id': 'remote-envelope-mixed-exact-personal'}

So get_envelope_by_client_id(dataset, 'remote-envelope-<id>') returns None. The connection-threading theory was already disproved in TASK-13344 and is confirmed disproved here: the source lookup on the same connection succeeds.

ROOT CAUSE. _attach_candidate (personal_context_conflicts.py:133-155) is the only path that creates a personal-context conflict. It reconstructs the candidate envelope from the canonical journal, INSERTS IT IF ABSENT (line 140), and only then inserts the conflict with remote_envelope_id = candidate.client_envelope_id. The product invariant is therefore: a personal-context conflict always has a stored remote candidate envelope.

The gate fixtures (_insert_conflict_with_source, _insert_mixed_conflicts_with_sources, _insert_conflict_set_with_sources) call service.store.insert_conflict directly with a fabricated remote_envelope_id = f'remote-envelope-{conflict_id}' and never insert that envelope. All five personal-context conflicts in the file hit 'remote is None'; two of the five are in tests that expect rejection, which is why only three fail.

TIMELINE. The fixtures date from 2026-09-04 (291ddc7ba6, 66945a5a9d). The remote-envelope lookup and identity check arrived on 2026-09-05 in c02afda9a7 'Implement journaled Personal Context conflict resolution', which added a NEW test file (test_sync_v2_personal_context_conflicts.py, 588 lines) but did not update this one. The new file builds conflicts correctly, by driving a real push through _attach_candidate and reading journal['remote_envelope_id'].

FIX DIRECTION. Build the personal-context conflicts through the product path rather than by hand: capture the journal via PersonalContextService.capture_sync_conflict(profile_id, conflict_id, dataset_id, device_id, local_envelope_id, domain, object_id, local_payload, local_envelope_digest, purge_generation, exchange) and then let _attach_candidate create the candidate envelope and the conflict. Note _validate_candidate (line 105) compares the stored candidate field-by-field against the journal reconstruction, so inserting an envelope alone is NOT sufficient -- the journal must exist and match.

Not attempted here because it is a focused rewrite on a security-sensitive conflict path and deserves its own change, not a tail-end edit during triage.

Source: TASK-13344 AC1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three mixed-batch tests build personal-context conflicts through the product path and pass
- [ ] #2 The fixtures no longer call store.insert_conflict with a fabricated remote_envelope_id
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
INVESTIGATED FURTHER (no code change). The obvious repair -- teach the exchange-gate fixtures to build a valid conflict -- is NOT the right shape, and this records why so the next attempt does not rediscover it.

The gate tests run on the transport double: _RecordingService / _RecordingRepository in test_sync_v2_personal_context_transport.py, which by its own docstring supplies 'only the canonical proof needed by isolated transport scenarios'. It has sync_integrity_key, apply_sync_object and apply_sync_ingress, and nothing else.

Making a personal-context resolution succeed through that double requires adding, at minimum:
  - capture_sync_conflict or an equivalent journal writer
  - get_sync_conflict, returning a journal with FOURTEEN keys that _candidate_envelope
    and _finalize read: profile_id, dataset_id, purge_generation, integrity_key_id,
    candidate, candidate_version_id, domain, candidate_object_id, candidate_created_at,
    remote_envelope_id, conflict_id, authority, local_digest, local_envelope_digest
    (plus key_slot, read by _attach_candidate)
  - resolve_sync_conflict, returning a receipt that the rest of _finalize consumes

and the journal's values must agree exactly with the stored candidate envelope, because
_validate_candidate compares _restore_personal_context_from_storage(dataset, candidate)
field-by-field against the reconstruction, and _finalize separately recomputes
local_digest as sha256 over canonical_json_bytes(local.payload) and local_envelope_digest
via _local_envelope_digest(local).

That is a canonical conflict journal reimplemented inside a test double -- a second source
of truth for conflict semantics, which is the failure mode this whole review is about. It
would pass today and drift exactly as the current fixtures did.

WHAT IS TRACTABLE, and confirmed while checking: the dataset the gate tests enroll ALREADY
carries the metadata _candidate_envelope requires -- metadata['personal_context'] with
profile_id, integrity_key_id, purge_generation 0 and link_state 'complete'
(test_sync_v2_personal_context_transport.py:269-274). So the blocker is purely the missing
canonical journal, not the dataset shape.

RECOMMENDED SHAPE, unchanged but now better motivated: port these three tests to the
runtime used by test_sync_v2_personal_context_conflicts.py, whose _conflict() helper drives
a REAL push through _attach_candidate against a real canonical service. The gate tests then
assert what they are actually about -- mixed-batch ordering and the exchange gate -- over a
conflict the product itself produced.

Also confirmed: simply inserting an envelope whose client_envelope_id is the fabricated
remote-envelope-<id> is NOT sufficient. It clears the 'remote is None' check and then fails
in _validate_candidate, because no journal exists to reconstruct the expected candidate
from.
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
