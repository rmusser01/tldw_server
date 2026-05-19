---
id: TASK-435
title: Add native WebUI first-run readiness screen
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-19 01:39
labels:
- implementation
- setup
- frontend
- webui
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx
- apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx
- apps/packages/ui/src/routes/option-setup.tsx
- apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/app/core/Setup/readiness_profiles.py
- tldw_Server_API/app/core/Setup/readiness_service.py
- tldw_Server_API/app/core/Setup/setup_manager.py
- tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py
- tldw_Server_API/tests/Setup/test_setup_readiness_api.py
- tldw_Server_API/tests/Setup/test_setup_readiness_preview.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 7 from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: a native setup readiness screen backed by useSetupReadiness, with profile/lane review, explicit secondary Provision Now action, Verify action, and backend /setup fallback for first-run guard failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native `/setup` readiness screen renders readiness profiles, Chat, Embeddings/RAG, and Speech lanes.
- [x] #2 TTS remains visible but secondary inside the Speech lane.
- [x] #3 `Provision now` remains a separate secondary action and is not called by profile selection.
- [x] #4 Backend `/setup` fallback link remains visible, including remote first-run guard states.
- [x] #5 `/setup` keeps connection onboarding when server configuration is still missing or invalid.
- [x] #6 `/setup` switches the same readiness screen to admin mode after first-run completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing setup route and component patterns.
2. Add failing screen tests for profile/lane rendering, fallback link, and explicit Provision Now behavior.
3. Implement ReadinessSetupScreen using useSetupReadiness without hidden provisioning.
4. Wire the native setup route to render readiness when available with fallback to legacy /setup.
5. Run focused frontend tests and update the plan/backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Live walkthrough follow-up completed. First-run WebUI testing exposed and this branch fixes: profile-only readiness payloads now expand through the backend profile contract instead of previewing/provisioning as a no-op, speech verification uses the install manager keyword-only resource_profile argument, setup config rewriting preserves line breaks for adjacent key updates, and the WebUI immediately reflects preview lane state with empty install plans shown as no work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native WebUI setup readiness screen is implemented and walkthrough-tested against a live backend/WebUI pair. Verified profile selection, explicit Preview selection, separate Provision now, Verify readiness, visible backend /setup fallback, and profile-only API behavior. Fresh verification: setup pytest slice 33 passed; WebUI Vitest slice 2 files/9 tests passed; git diff --check passed; Bandit on touched backend production files returned 0 findings.
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
