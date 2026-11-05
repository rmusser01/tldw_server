---
id: TASK-13938
title: Repair inherited Jobs encryption fixture skips
status: To Do
created_date: 2026-10-26 02:41
labels:
- Jobs
- tests
- follow-up
references:
- TASK-13937
- tldw_Server_API/tests/Jobs/test_jobs_manager.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Separate follow-up from strict Jobs completion foundation TASK-13937. Two unchanged test_jobs_manager.py cases silently skip with a misleading crypto-backend-unavailable message because their fixture Base64 keys decode to 33 bytes rather than required32. Crypto backend is available. Repair fixtures and verify encryption acquisition/filtered key rotation in a separate PR; do not change production encryption semantics or fold this into strict extraction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Use deterministic valid32-byte encryption keys in the two inherited fixtures, including the replacement key.
- [ ] #2 With crypto backend available, both test_acquire_decrypts_payload and test_rotate_encryption_keys_respects_filters_sqlite execute their real encryption assertions rather than skip.
- [ ] #3 Preserve legitimate unavailable-backend skips and existing production encryption/acquisition/rotation behavior.
- [ ] #4 Verify focused encryption plus affected Jobs tests, Ruff and baseline-compared test Bandit without new suppressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Durable diagnosis (originally validated during the 2026-10-20 foundation qualification): the crypto backend is available. The fixture literals used by test_acquire_decrypts_payload and test_rotate_encryption_keys_respects_filters_sqlite in test_jobs_manager.py are truncated to 44 Base64 characters, which decode to 33 bytes rather than the required 32. Their broad setup exception handler reports 'Crypto backend unavailable' and skips, hiding this invalid fixture input.

The 2026-11-03 full non-PG Jobs run again reports exactly those two skips; it passes 2,232 other cases. Required real PostgreSQL completion/lifecycle/admin/RLS/parity verification passes 257 cases without skips. These are current results recorded in TASK-13937 and the committed foundation implementation plan; historical /tmp paths are optional local diagnostics, not the only evidence.

Correct these fixtures and narrow their skip diagnosis in a separate follow-up, retaining all success-path encryption assertions. The strict extraction PR does not modify encryption behavior or these fixtures. Historical task TASK-13879 is now archived as TASK-13938.1; TASK-13938 is the active follow-up. Governing extraction decisions are ADR-058 and ADR-082; the archive's old ADR-078 number is historical. No new architecture decision is required for a test-fixture correction.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
