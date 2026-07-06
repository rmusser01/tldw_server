---
id: TASK-12811
title: Improve user-facing API module documentation
status: Done
assignee: []
created_date: 2026-07-04 22:25
updated_date: 2026-07-05 03:07
labels:
- docs
- api
- openapi
dependencies: []
priority: medium
modified_files:
- Docs/API-related/API_Tags_Index.md
- Docs/superpowers/plans/2026-07-04-api-module-documentation.md
- Docs/superpowers/specs/2026-07-04-api-module-documentation-design.md
- backlog/tasks/task-12027 - Improve-user-facing-API-module-documentation.md
- tldw_Server_API/app/main.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the user-facing documentation that explains what each API module can do. Scope covers a practical OpenAPI tag/module capability guide and aligned OpenAPI tag metadata so /docs and /redoc are easier to browse. Keep endpoint behavior unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API tag/module guide explains major API capabilities in user-facing language and links to existing detailed docs where available.
- [x] #2 OpenAPI tag descriptions and ReDoc tag groups align with the module guide for stable/common API surfaces.
- [x] #3 Experimental, admin-only, or low-level/internal surfaces are clearly labeled instead of presented as primary user workflows.
- [x] #4 Verification records OpenAPI/schema build checks, markdown/link sanity checks, and Bandit for touched Python files if Python is changed.
- [x] #5 Generated Published docs are left unchanged; source docs are ready for the docs publishing process to regenerate Published.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a concise design spec capturing the reviewed scope and risk controls. 2. Inventory router tags against existing OpenAPI metadata and API docs. 3. Replace the short source API_Tags_Index with a grouped capability guide while leaving generated Published docs unchanged. 4. Update OpenAPI tag metadata/grouping in main.py without changing routes or endpoint behavior. 5. Verify OpenAPI generation, markdown/link sanity, generated Published no-diff status, and security scan for touched Python.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design review before implementation identified two scope controls: keep the guide comprehensive at the module/tag level rather than documenting every endpoint, and avoid letting main.py become a full API manual. OpenAPI metadata should summarize discoverability, while detailed usage stays in markdown docs. Verification update: Docs/Published/API-related/API_Tags_Index.md is generated output and is intentionally left unchanged; fresh verification confirmed no Published branch diff, source markdown links resolve, main.py py_compile passes, OpenAPI schema smoke reports openapi tags=170 groups=10, and Bandit reports errors=0 results=0 in /tmp/bandit_api_module_docs.json.

PR: https://github.com/rmusser01/tldw_server/pull/2637
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded the source API tag index into a grouped module capability guide and aligned OpenAPI tag descriptions/ReDoc groups so users can browse by goal. Generated Published docs were left unchanged for the publishing process. Endpoint behavior, routes, schemas, and security settings were unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reopened for PR follow-up: rebase PR #2637 on latest dev and evaluate/address review comments and checks before pushing an updated branch.
PR follow-up completed: rebased PR #2637 onto latest origin/dev (39e9d1d1c8250d62ee22732f85c17f541b735760), removing unrelated Chat, MCP, UserProfiles, Web_Scraping, and test files from the PR diff. Current diff is limited to Docs/API-related/API_Tags_Index.md, Docs/superpowers API-docs plan/spec files, tldw_Server_API/app/main.py, and this Backlog task record. Gemini's Query import and Qodo code comments are obsolete after rebase because those referenced files are no longer in the PR diff; the source-doc links in API_Tags_Index.md were verified to resolve against source docs. Verification after rebase: py_compile passes for tldw_Server_API/app/main.py; API_Tags_Index local markdown links resolve; Docs/Published is unchanged against origin/dev; git diff --check origin/dev...HEAD exits cleanly; no old code-surface files remain in the PR diff; OpenAPI smoke passes with a local dummy SINGLE_USER_API_KEY and reports tags=174 groups=10; Bandit on tldw_Server_API/app/main.py reports errors=0 results=0 in /tmp/bandit_api_module_docs_rebase.json.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
