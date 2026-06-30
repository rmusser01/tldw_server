---
id: TASK-499
title: Verify unified first-run onboarding end-to-end
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 16:56'
labels: []
dependencies: []
references:
  - TASK-489
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-10-end-to-end-verification-security-and-release-gate
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add final Playwright coverage and run/reconcile the backend, frontend, E2E, Bandit, and whitespace release gate for the unified first-run onboarding implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playwright covers focused setup shell until skip/completion.
- [x] #2 Playwright covers completed setup showing the first-source milestone without backend setup mutation on dismiss.
- [x] #3 Playwright covers first-chat completion only after a successful backend response.
- [x] #4 Frontend setup hook avoids duplicate initial state loads across StrictMode/remounts while preserving backend-authoritative state.
- [x] #5 Focused backend, frontend, E2E, Bandit, and whitespace verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a mocked-backend Playwright workflow for the unified first-run onboarding journey, including focused shell, completed-state milestone, and first-chat completion gate.

Moved Next root shell registration into an effect and made nested shell overrides retry once when the root shell setter is registered after nested layout mount.

Coalesced initial setup state/metadata loads in useSetupOnboarding so remounts do not discard completed backend state while metadata is still in flight. Added hook regression tests for metadata-in-flight duplicate loading and remount continuation.

Attempted the broader tldw_Server_API/tests/Setup release command; unrelated audio health/audio pack/installer failures were observed outside this onboarding slice before the run was stopped.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified the unified first-run onboarding flow end to end with a new Playwright workflow, fixed shell override timing needed by the focused setup shell, and hardened the setup onboarding hook against remount-driven duplicate initial loads. Focused backend setup/config tests passed, focused frontend tests passed, the Playwright workflow passed, Bandit reported no findings for the setup backend scope, and git diff whitespace validation passed.

Follow-up verification on 2026-05-31: docs and Makefile onboarding gate passed with 90 tests, 5 warnings: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py -v.

Follow-up verification on 2026-05-31: setup audio release-gate failures were resolved under TASK-500. Full setup/config gate passed with 324 tests, 4 warnings: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v. Bandit for touched setup/audio cleanup code reported 0 findings at /tmp/bandit_setup_audio_release_gate.json, and git diff --check passed.
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
