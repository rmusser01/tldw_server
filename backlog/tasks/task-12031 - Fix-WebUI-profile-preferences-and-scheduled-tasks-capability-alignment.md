---
id: TASK-12031
title: Fix WebUI profile preferences and scheduled-tasks capability alignment
status: Done
assignee: []
created_date: '2026-06-25 21:58'
updated_date: '2026-06-25 22:02'
labels:
  - webui
  - backend
  - auth
  - settings
  - scheduled-tasks
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the WebUI audit remediation roadmap: profile preferences must not return a backend 500 during optional WebUI bootstrap, and scheduled-tasks route state must match backend capability availability instead of surfacing raw 404/server-unreachable failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/users/me/profile?sections=preferences returns 200 preferences or a typed partial-section response for supported auth principals.
- [x] #2 Optional profile bootstrap failures do not cause the WebUI chat/global shell to report the whole backend as unreachable.
- [x] #3 Scheduled tasks route either reaches a registered backend contract or renders a capability-unavailable state for deployments where the router is absent.
- [x] #4 Focused backend/frontend tests cover the selected profile and scheduled-tasks behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 task created after Stage 1 auth persistence commit 636853baea. Planned areas: tldw_Server_API/app/api/v1/endpoints/users.py, tldw_Server_API/app/core/UserProfiles/service.py, relevant UserProfile tests, scheduled-tasks router groups/services/routes/tests after investigation.

Implemented as regression coverage after investigation showed current runtime behavior already satisfies the audited Stage 2 contracts. Added a direct profile preferences endpoint regression and a backend router-group regression for scheduled-tasks control-plane availability. Existing ScheduledTasksPage recovery coverage confirms that a missing OpenAPI path renders the capability-unavailable state without calling the list endpoint.

Verification: `python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_read.py::test_user_profile_preferences_section_returns_success tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py::test_scheduled_tasks_router_groups_expose_control_plane_route -q` passed 2 tests. `bun run test:run ../packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx -t "shows an unsupported-state message"` passed 1 focused test with 47 skipped in the file. `python -m compileall -q tldw_Server_API/tests/UserProfile/test_user_profile_read.py tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py` passed. `git diff --check` passed. Bandit was run on the touched Python test files; new Stage 2 assertions are marked `# nosec B101`, while the scanner still reports 46 pre-existing B101 assert warnings in `test_user_profile_read.py`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 is covered without runtime changes. The profile preferences request now has an explicit regression ensuring it returns a preferences object without a section error. The scheduled-tasks backend router groups now have a regression ensuring the WebUI-facing control-plane route is present, and existing frontend recovery coverage confirms unsupported deployments render capability-unavailable UI instead of a raw failure.
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
