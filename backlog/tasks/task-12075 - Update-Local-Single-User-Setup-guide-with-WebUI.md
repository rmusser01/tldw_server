---
id: TASK-12075
title: Update Local Single-User Setup guide with WebUI
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 15:12'
labels:
  - docs
  - getting-started
  - webui
dependencies: []
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
Implemented on a clean PR branch based on origin/dev. The current base already included a compact WebUI block under ## Start, so the final guide extends that block with explicit .env.local values, browser-visible API-key guidance, Bun/npm commands, and a WebUI response check instead of adding a duplicate startup section.

Verification:
- Reviewed Docs/Getting_Started/Profile_Local_Single_User.md command order with sed.
- git diff --check -- Docs/Getting_Started/Profile_Local_Single_User.md Docs/superpowers/specs/2026-06-30-local-single-user-webui-guide-design.md Docs/superpowers/plans/2026-06-30-local-single-user-webui-guide-implementation-plan.md passed.
- Referenced files exist: apps/DEVELOPMENT.md, apps/tldw-frontend/README.md, and README.md.
- README anchor exists: ### Run the Web UI (WIP).
- Stale README-only wording check against the guide found no matches.
- Helper_Scripts/docs/check_onboarding_command_boundaries.py passed.
- Bandit skipped: documentation-only Markdown changes; no Python code touched.

Subagent reviews:
- Task 1 spec review passed; Task 1 quality review passed.
- Task 2 spec review passed; Task 2 quality review passed.

Final review fixes applied: clarified that .env.local must be edited to use the shown local API values, marked execution plan checklist items complete, updated the design spec status to Implemented, aligned plan wording with the final guide, and rebased the task record to TASK-12075 to avoid the existing TASK-12072 on origin/dev.

Review follow-up: corrected the design-spec risk note so it reflects that apps/tldw-frontend/.env.local.example already uses 127.0.0.1; the documented risk is now users mixing localhost and 127.0.0.1 across API/WebUI URLs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the local single-user setup guide so it is self-contained for API plus WebUI setup. The guide now includes WebUI prerequisites, .env.local values, Bun and npm start paths, WebUI verification, first-value guidance, troubleshooting for common local WebUI issues, and links to deeper WebUI and advanced networking docs.
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
