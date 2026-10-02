---
id: TASK-13420
title: Embeddings test_tenant_quota_429 fails standalone (pre-existing)
status: To Do
assignee: []
created_date: '2026-10-02 03:14'
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
- [ ] #1 test_tenant_quota_429 passes, or is rewritten to assert the current tenant-quota contract
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
