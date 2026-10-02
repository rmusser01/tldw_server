---
id: TASK-13420
title: Embeddings test_tenant_quota_429 fails standalone (pre-existing)
status: Done
assignee: []
created_date: '2026-10-02 03:14'
updated_date: '2026-10-03 01:10'
labels:
  - tests
  - embeddings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py::test_tenant_quota_429 fails alone with 'assert None is not None' on current dev, and also at 607431154c, before the RG safety-net PRs.

Observed cause: the test does monkeypatch.setenv('AUTH_MODE', 'multi_user') and expects the second call to _check_backpressure_and_quotas() to be tenant-rate-limited. _check_backpressure_and_quotas -> _should_enforce_tenant_rps -> _is_multi_user_runtime -> _is_single_user_profile -> is_single_user_profile_mode() -> get_profile(), which reads the AuthNZ settings singleton (get_settings()), not os.getenv('AUTH_MODE') directly. The test never calls reset_settings() after the monkeypatch, so the cached singleton (initialized earlier in the session to AUTH_MODE=single_user / profile=local-single-user, per the project's test default) is still in effect: is_single_user_profile_mode() returns True, _is_multi_user_runtime() returns False, _should_enforce_tenant_rps() returns False, and the tenant-RPS increment/limit check at embeddings_v5_production_enhanced.py around line 655 is skipped entirely for both calls -> both return None -> the second assert fails. Likely fix: have the test call reset_settings() (or an equivalent settings-cache invalidation) after monkeypatching AUTH_MODE, the same way tests/Resource_Governance/test_e2e_tokens_daily_cap.py's _init_authnz_sqlite helper does; or make _is_single_user_profile() read the live env var instead of (or in addition to) the cached singleton.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_tenant_quota_429 passes, or is rewritten to assert the current tenant-quota contract
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed as described: _is_single_user_profile() reads the cached AuthNZ settings singleton, so monkeypatching AUTH_MODE alone left the test in single-user mode and the tenant-RPS check never ran. Fix (test-only): reset_settings() after the setenv, and again in a finally so later tests rebuild settings from the restored env. Verified: test file 7 passed; tests/Embeddings -n 4: 682 passed, 18 skipped. Bandit: not applicable (test-only change). No docs affected.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
test_tenant_quota_429 now exercises the real multi-user path: it resets the AuthNZ settings singleton after setting AUTH_MODE=multi_user and restores it afterwards. The test passes alone and in the parallel Embeddings suite; no production code changed.
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
