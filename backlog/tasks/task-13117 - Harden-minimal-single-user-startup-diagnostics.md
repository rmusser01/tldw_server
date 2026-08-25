---
id: TASK-13117
title: Harden minimal single-user startup diagnostics
status: In Progress
created_date: 2026-08-25 01:12
labels:
- authnz
- deployment
- documentation
priority: High
documentation:
- Docs/Deployment/minimal-deploy.md
- Docs/Getting_Started/Profile_Local_Single_User.md
- Docs/Design/plans/IMPLEMENTATION_PLAN_minimal_startup_diagnostics_TASK_13117.md
modified_files:
- tldw_Server_API/app/core/AuthNZ/initialize.py
- tldw_Server_API/app/core/startup_logging.py
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py
- tldw_Server_API/tests/Config/test_startup_api_key_logging.py
- tldw_Server_API/tests/Logging/test_main_log_level.py
- tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py
- Docs/Deployment/minimal-deploy.md
- Docs/Published/Deployment/minimal-deploy.md
updated_date: 2026-08-25 05:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the latest-dev minimal deployment path so closed stdin does not crash AuthNZ prompts, application startup logging honors LOG_LEVEL instead of forcing DEBUG, and the minimal deployment guide uses the maintained local and Docker flows. Preserve fail-closed single-user database invariants and document reversible recovery rather than mutating user data automatically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 AuthNZ yes/no prompts use their declared default when stdin reaches EOF and surface a concise non-interactive notice.
- [ ] #2 The main application Loguru sink honors a valid LOG_LEVEL and falls back safely for missing or invalid values.
- [ ] #3 The minimal deployment guide uses the maintained virtualenv/wizard/Makefile and supported Docker Compose paths.
- [ ] #4 Troubleshooting documents log capture plus reversible handling for single-user invariant conflicts without automatic database cleanup.
- [ ] #5 Focused regression tests, relevant suites, documentation checks, and Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create the repository-required implementation plan, then use TDD for prompt EOF behavior and log-level resolution, update deployment documentation, run focused and broader verification, record results, commit, and open a draft PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-08-24: Created isolated worktree from origin/dev at 4091735b6f9e. Focused baseline: 33 passed, 0 failed. Temporary implementation plan created per repository workflow.

2026-08-24: TDD complete for closed-stdin prompt defaults and LOG_LEVEL normalization/wiring. Focused suite: 43 passed.

2026-08-24: Disposable end-to-end AuthNZ initializer run with stdin closed exited 0, selected yes/no defaults, and completed bootstrap; disposable database, env, and generated secrets were removed.

2026-08-24: Broader relevant suite: 112 passed. Minimal environment Uvicorn smoke returned HTTP 200. Bandit scanned initialize.py, startup_logging.py, and main.py with 0 findings.
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
