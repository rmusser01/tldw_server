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
modified_files:
- tldw_Server_API/app/core/AuthNZ/initialize.py
- tldw_Server_API/app/core/startup_logging.py
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py
- tldw_Server_API/tests/Config/test_startup_api_key_logging.py
- tldw_Server_API/tests/Logging/test_main_log_level.py
- tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py
- tldw_Server_API/tests/Admin_Webhooks/test_legacy_import_postgres.py
- tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py
- tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py
- Docs/Deployment/minimal-deploy.md
- Docs/Published/Deployment/minimal-deploy.md
updated_date: 2026-08-30 18:38
references:
- https://github.com/rmusser01/tldw_server/pull/2820
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the latest-dev minimal deployment path so closed stdin does not crash AuthNZ prompts, application startup logging honors LOG_LEVEL instead of forcing DEBUG, and the minimal deployment guide uses the maintained local and Docker flows. Preserve fail-closed single-user database invariants and document reversible recovery rather than mutating user data automatically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AuthNZ yes/no prompts use their declared default when stdin reaches EOF and surface a concise non-interactive notice.
- [x] #2 The main application Loguru sink honors a valid LOG_LEVEL and falls back safely for missing or invalid values.
- [x] #3 The minimal deployment guide uses the maintained virtualenv/wizard/Makefile and supported Docker Compose paths.
- [x] #4 Troubleshooting documents log capture plus reversible handling for single-user invariant conflicts without automatic database cleanup.
- [x] #5 Focused regression tests, relevant suites, documentation checks, and Bandit pass.
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

2026-08-24: After rebasing onto origin/dev b1d0aed671, broader relevant suite passed 113 tests. Minimal environment Uvicorn smoke returned HTTP 200. Bandit scanned initialize.py, startup_logging.py, and main.py with 0 findings. git diff --check passed.

2026-08-24: Opened draft PR #2820 against dev. Merge readiness is intentionally pending the repository-required human-authored Change summary.
2026-08-30: Reopened for requested PR #2820 rebase, Dodo review follow-up, fresh verification, and merge. Rebased cleanly onto origin/dev f676e23549.
2026-08-30: Latest dev introduced three Admin Webhooks tests that directly registered the prohibited AuthNZ conftest plugin. Replaced those references with the canonical authnz_full_fixtures bridge. Isolation guard: 6 passed; affected Postgres suites: 24 tests collected. Relevant regression suite: 113 passed. One chained smoke immediately after pytest saw a closed SQLite connection; the failure did not reproduce in a direct launch or five consecutive standalone scrubbed smokes, so no speculative database change was made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the minimal single-user launch path without weakening database safety. Closed stdin now selects each AuthNZ prompt default instead of raising EOFError; startup logging honors recognized LOG_LEVEL values and safely falls back to INFO; and the source plus published deployment guides now use maintained Make, wizard, and Compose flows with observable log capture and reversible, backup-first invariant recovery. Added unit, subprocess wiring, documentation contract, and published-copy synchronization coverage. Draft PR: https://github.com/rmusser01/tldw_server/pull/2820. The only remaining merge gate is the requester-authored Change summary required by project policy.
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
