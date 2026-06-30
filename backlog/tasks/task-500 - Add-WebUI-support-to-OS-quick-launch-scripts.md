---
id: TASK-500
title: Add WebUI support to OS quick-launch scripts
status: Done
assignee: []
created_date: '2026-05-24T05:55:00Z'
updated_date: 2026-05-24 05:55
labels:
- quickstart
- webui
- installer
dependencies: []
priority: medium
modified_files:
- quick-launch.sh
- quick-launch.command
- quick-launch.ps1
- Helper_Scripts/Installer_Scripts/MacOS_Run_tldw.sh
- Helper_Scripts/Installer_Scripts/Linux_Run_tldw.sh
- Helper_Scripts/Installer_Scripts/Windows_Run_tldw.bat
- Helper_Scripts/Installer_Scripts/MacOS_Install_Update.sh
- Helper_Scripts/Installer_Scripts/Linux_Install_Update.sh
- Helper_Scripts/Installer_Scripts/Windows_Install_Update.bat
- tldw_Server_API/tests/Utils/test_quick_launch_scripts.py
- README.md
- Docs/Getting_Started/Profile_Local_Single_User.md
- Docs/superpowers/plans/2026-05-24-quick-launch-webui-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate WebUI-capable quick launch behavior around the repo-root quick-launch scripts. Root macOS/Linux shell and Windows PowerShell launchers should support api, webui, and all modes, while installer run scripts become compatibility wrappers that delegate to the root launchers for installed checkouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repo-root quick-launch.sh and quick-launch.ps1 support api, webui, and all modes with all as the default.
- [x] #2 Root quick-launch scripts keep local-single setup for API startup and launch the Next.js WebUI from apps/tldw-frontend via Bun.
- [x] #3 Installer macOS/Linux/Windows run scripts delegate to the root launchers instead of duplicating API/WebUI launch logic.
- [x] #4 README and local single-user docs describe the consolidated quick-launch modes and default API + WebUI behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-24-quick-launch-webui-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red-green coverage recorded in `tldw_Server_API/tests/Utils/test_quick_launch_scripts.py`: the new consolidation assertions first failed against API-only root launchers and duplicated installer run scripts, then passed after consolidation.
- Verification passed: focused launcher and onboarding docs suites reported 34 passed; Bash syntax checks passed for touched shell launchers and installer scripts; root and wrapper help output returned successfully; Bandit reported no findings for the touched Python test file; `git diff --check` passed.
- Follow-up PR review fixes: populated the task creation timestamp, corrected user-facing `macOS` casing, validated PowerShell `TLDW_API_START_DELAY`, used the current PowerShell edition for the child API process, and avoided `0.0.0.0` as the default browser-facing WebUI API URL.
- Follow-up pre-merge fixes: added the missing quick-launch test helper docstring and expanded the PR description to match the repository template sections.
- Qodo review coverage: added explicit regression assertions for documented launcher tests, direct Uvicorn PID capture in shell all-mode cleanup, and removal of the brittle legacy Windows exit-count assertion.
- PowerShell parser execution was skipped because neither `pwsh` nor `powershell` is installed on this host.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated quick-launch behavior around the repo-root launchers. quick-launch.sh and quick-launch.ps1 now support api, webui, and all modes, default to all, preserve local-single API setup, launch the Next.js WebUI from apps/tldw-frontend on port 8080, and set NEXT_PUBLIC_API_URL by default. The installer macOS/Linux/Windows run scripts are now compatibility wrappers that delegate to the root launchers with legacy venv defaults. Documentation now describes the default API + WebUI behavior and mode-specific startup commands. Verification passed with 34 focused launcher/onboarding tests, Bash syntax checks, launcher help checks, Bandit with no findings, and diff whitespace checks.
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
