---
id: TASK-12072
title: Update Local Single-User Setup guide with WebUI
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-30 06:24
labels:
- docs
- getting-started
- webui
dependencies: []
modified_files:
- Docs/Getting_Started/Profile_Local_Single_User.md
- Docs/superpowers/specs/2026-06-30-local-single-user-webui-guide-design.md
- Docs/superpowers/plans/2026-06-30-local-single-user-webui-guide-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Docs/Getting_Started/Profile_Local_Single_User.md self-contained for running the local API plus the Next.js WebUI, including prerequisites, start steps, verification, and troubleshooting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local single-user guide documents WebUI setup without requiring the README add-on section.
- [x] #2 Guide covers API/WebUI URLs, .env.local values, Bun/npm start path, verification, and common troubleshooting.
- [x] #3 Related docs links remain accurate and avoid conflicting localhost/127.0.0.1 guidance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-30-local-single-user-webui-guide-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in commits 435f9eb9543ed3a9488da242c743ae951b9eb47f and ea1fc7fef9cbb3475ccc29c98229c4015bbeb5dd.

Verification:
- Reviewed Docs/Getting_Started/Profile_Local_Single_User.md command order with sed -n 1,230p.
- git diff --check -- Docs/Getting_Started/Profile_Local_Single_User.md passed.
- Referenced files exist: apps/DEVELOPMENT.md, apps/tldw-frontend/README.md, and README.md.
- README anchor exists: ### Run the Web UI (WIP).
- Stale README-only wording check found no matches.
- source .venv/bin/activate and python Helper_Scripts/docs/check_onboarding_command_boundaries.py passed.
- Bandit skipped: documentation-only Markdown changes; no Python code touched.

Subagent reviews:
- Task 1 spec review passed; Task 1 quality review passed.
- Task 2 spec review passed; Task 2 quality review passed.

Final review fixes applied: clarified that `.env.local` must be edited to use the shown local API values, marked execution plan checklist items complete, and updated the design spec status to Implemented.

Final review minor cleanup: aligned the implementation plan WebUI `.env.local` wording and env fence with the final guide.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the local single-user setup guide so it is self-contained for API plus WebUI setup. The guide now includes WebUI prerequisites, `.env.local` values, Bun and npm start paths, WebUI verification, first-value guidance, troubleshooting for common local WebUI issues, and links to deeper WebUI and advanced networking docs.
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
