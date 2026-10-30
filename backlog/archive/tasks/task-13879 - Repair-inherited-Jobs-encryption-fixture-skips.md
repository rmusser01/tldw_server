---
id: TASK-13879
title: Repair inherited Jobs encryption fixture skips
status: To Do
created_date: 2026-10-20 18:11
labels:
- Jobs
- tests
- follow-up
references:
- TASK-13878
- tldw_Server_API/tests/Jobs/test_jobs_manager.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Separate follow-up from strict Jobs completion foundation TASK-13878. Two unchanged test_jobs_manager.py cases silently skip with a misleading crypto-backend-unavailable message because their fixture Base64 keys decode to 33 bytes rather than required32. Crypto backend is available. Repair fixtures and verify encryption acquisition/filtered key rotation in a separate PR; do not change production encryption semantics or fold this into strict extraction.
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
Validated during foundation local qualification on source0f2b649cb8688abd7132d8fad7ec7dd293baaf9b; test_jobs_manager.py byte-identical to execution basebe07b6618217743938ea395bc756cf15fb0426aa. Existing literals at lines355,1842,1843 are sliced to44 Base64 chars and decode to33 bytes. /tmp/task13878_skip_diagnosis.log proves crypto backend available and invalid fixture lengths; full non-PG original JUnit /tmp/task13878_sqlite.xml records these2 skips. Foundation leaves these fixtures/production untouched as mandated strict extraction. Local task search and exact test-name search found no existing specific follow-up; live deve5216051/published maximum13877 unchanged, local foundation13878, allocate13879. ADR assessment: no new durable architecture decision for a fixture-only correction; ADR-058/078 foundation decisions unchanged.
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
