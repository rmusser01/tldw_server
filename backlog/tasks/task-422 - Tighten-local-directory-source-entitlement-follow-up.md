---
id: TASK-422
title: Tighten local-directory source entitlement follow-up
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 19:02'
labels:
  - review
  - ingestion-sources
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address post-merge code-review feedback for local-directory ingestion source entitlements. Multi-user deployments must not enable local-directory source creation via a global flag, and malformed persisted rollout percentages should fail closed through the real feature-flag service path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-directory ingestion source creation is not authorized by a global feature flag in multi-user mode.
- [x] #2 User/org-scoped feature flags still authorize eligible users according to existing targeting rules.
- [x] #3 Malformed persisted rollout_percent values fail closed through the feature-flag service path used by the access policy.
- [x] #4 Focused backend tests cover the changed entitlement behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented post-review entitlement hardening. Local-directory ingestion source access no longer accepts global feature flags in multi-user mode. Persisted malformed or out-of-range rollout_percent values now normalize to 0 for fail-closed behavior in feature-flag reads. Verification: focused Ingestion Sources access-policy tests, admin system-ops feature-flag test, git diff --check, and Bandit on touched backend files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Local-directory source entitlement review fixes are implemented and verified. User/org-scoped flags still authorize access; global flags now report false; malformed persisted rollout_percent values fail closed before the access policy can authorize.
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
