# WebUI Stage 2 Contract Alignment Plan

## Stage 1: Profile Preferences Contract
**Goal**: Preserve the audited profile preferences endpoint contract.
**Success Criteria**: `/api/v1/users/me/profile?sections=preferences` returns a successful profile payload with a `preferences` object and no preferences section error.
**Tests**: `python -m pytest tldw_Server_API/tests/UserProfile/test_user_profile_read.py::test_user_profile_preferences_section_returns_success -q`
**Status**: Complete

## Stage 2: Scheduled Tasks Capability Contract
**Goal**: Preserve backend route registration and frontend capability recovery for scheduled tasks.
**Success Criteria**: Backend router groups used by WebUI include the scheduled-tasks control plane, and the Scheduled Tasks page renders the existing capability-unavailable state without calling the list endpoint when OpenAPI does not advertise the route.
**Tests**: `python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -k router_group -q`; `bun run test:run ../packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
**Status**: Complete

## Stage 3: Verification And Task Closure
**Goal**: Record evidence and close `TASK-12031`.
**Success Criteria**: Focused backend/frontend tests, lint for touched frontend files, Bandit on touched Python scope, and diff checks are recorded.
**Tests**: Focused commands from Stages 1-2 plus lint/Bandit/diff checks.
**Status**: Complete
