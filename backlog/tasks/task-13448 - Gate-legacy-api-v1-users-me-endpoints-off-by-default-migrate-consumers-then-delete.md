---
id: TASK-13448
title: Gate legacy /api/v1/users/me endpoints off by default, migrate consumers, then
  delete
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-13444 Step 6 decision gate. Evidence 2026-10-02: ENABLE_LEGACY_USER_ME_ENDPOINTS (users.py:121) defaults true; no sunset registered in the deprecation registry; >=5 test files consume the enabled path (test_chat_research_runs_endpoint _current_user_id uses json id; test_admin_e2e_support_api test_single_user_api_key_can_read_users_me; test_auth_comprehensive 3 spots); successor /me/profile returns sectioned UserProfileResponse - not field-compatible with DeprecatedUserResponse. Plan: map each consumer to /me/profile or /me/capabilities, flip default to 410-gone, then delete routes+gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered 2026-10-04 from TASK-13410 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
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
