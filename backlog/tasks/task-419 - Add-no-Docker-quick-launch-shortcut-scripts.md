---
id: TASK-419
title: Add no-Docker quick-launch shortcut scripts
status: Done
priority: Medium
documentation:
- Docs/Design/2026-05-17-quick-launch-scripts.md
- Docs/Getting_Started/Profile_Local_Single_User.md
- README.md
modified_files:
- quick-launch.sh
- quick-launch.command
- quick-launch.ps1
- README.md
- Docs/Getting_Started/README.md
- Docs/Getting_Started/Profile_Local_Single_User.md
- Docs/Published/Getting_Started/README.md
- Docs/Published/Getting_Started/Profile_Local_Single_User.md
- Docs/Design/2026-05-17-quick-launch-scripts.md
- Docs/superpowers/plans/2026-05-17-quick-launch-scripts-implementation-plan.md
- tldw_Server_API/tests/Utils/test_quick_launch_scripts.py
- tldw_Server_API/tests/Docs/test_onboarding_entrypoints.py
references:
- https://github.com/rmusser01/tldw_server/pull/1817
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add OS-native quick-launch scripts for local single-user self-hosters who are not using Docker or Make. Keep scripts as thin wrappers around the existing local-single setup/start contract, document them, and add contract tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repo-root quick-launch scripts exist for Linux/macOS shell, macOS Finder, and Windows PowerShell.
- [x] #2 Scripts reuse the existing local-single wizard/profile and start uvicorn without Docker or Make.
- [x] #3 Scripts avoid deprecated `summarize.py`/`-gui` entrypoints and do not print API keys by default.
- [x] #4 README and Getting Started local profile docs mention the no-Make/no-Docker shortcuts.
- [x] #5 Focused tests, shell syntax check, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-quick-launch-scripts-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `quick-launch.sh`, `quick-launch.command`, and `quick-launch.ps1` as thin wrappers around the existing `local-single` setup wizard and uvicorn start command.
- Added contract tests for script presence, local-single wiring, no Docker/Make/legacy Gradio usage, no default API-key printing, and documentation discoverability.
- Ran focused pytest, Bash syntax checks, `git diff --check`, and Bandit for touched Python tests. PowerShell syntax execution was skipped because neither `pwsh` nor `powershell` is available on this host.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Linux/macOS, macOS Finder, and Windows PowerShell quick-launch scripts for the local single-user no-Docker path. The scripts create/update .venv, run the existing local-single setup wizard, start uvicorn at 127.0.0.1:8000, avoid printing API keys by default, and are documented in README plus Getting Started local profile docs. Verification: focused onboarding/script pytest suite passed; bash syntax check passed; Bandit found no issues in touched Python tests. PowerShell syntax execution was not run because pwsh/powershell is unavailable on this host. Draft PR: https://github.com/rmusser01/tldw_server/pull/1817
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
