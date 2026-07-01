---
id: TASK-12078
title: Document frontend AUTH_MODE runtime config variable
status: Done
labels:
- documentation
- frontend
modified_files:
- apps/tldw-frontend/README.md
- apps/tldw-frontend/.env.local.example
- backlog/tasks/task-12078 - Document-frontend-AUTH-MODE-runtime-config-variable.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the frontend README so local setup key variables explicitly mention the server-side AUTH_MODE setting used by /api/_tldw-webui/runtime-config, including that it does not use a NEXT_PUBLIC_ prefix. This addresses PR review feedback for rmusser01/tldw_server#2561.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify PR branch context and existing env template coverage.
2. Update the README key variables list to explicitly document AUTH_MODE as a server-side runtime-config variable without a NEXT_PUBLIC_ prefix.
3. Verify the docs diff and whitespace.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated apps/tldw-frontend/README.md so the Key variables list explicitly documents AUTH_MODE as the server-side runtime auth mode used by /api/_tldw-webui/runtime-config, including that it has no NEXT_PUBLIC_ prefix. The .env.local.example already documents AUTH_MODE=single_user on this PR branch, so no env-template change was needed. Verification: git diff --check passed for the touched README and Backlog task. Bandit skipped because the change is documentation-only and does not touch Python code.
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
