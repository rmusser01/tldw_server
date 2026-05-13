---
id: TASK-304.14
title: Address PR 1616 Research Studio review comments
status: Done
assignee: []
created_date: '2026-05-13 04:12'
updated_date: '2026-05-13 04:38'
labels:
  - implementation
  - research-studio
  - pr-review
  - webui
  - backend
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/1616'
parent_task_id: TASK-304
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address still-actionable review threads and check issues on PR #1616 after live PR review sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All current actionable GitHub review threads for PR #1616 are either fixed or explicitly resolved with evidence.
- [x] #2 CodeRabbit hosted-route, portable screenshot, Backlog readability, capability composition, source-browse database, and ChatPane refresh findings are addressed where still valid.
- [x] #3 Gemini StudioPane no-source artifact visibility, SharedWithMe client navigation, and mobile tab persistence findings are addressed where still valid.
- [x] #4 Qodo route test brittleness and extension Navigate shim findings are addressed where still valid.
- [x] #5 Focused backend/frontend tests and diff hygiene are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PR review fixes: hosted /research-studio allowlist, portable manual screenshot path, Backlog description split, database-aware source browsing, combined dependency block scanning, stale ChatPane capability refresh, SharedWithMe client navigation, mobile tab persistence, behavioral route alias tests, extension string redirect destination, and no-source Studio regression coverage for existing artifacts plus Quick Notes expansion.

Verification: UI focused Vitest 53 passed; frontend route/network Vitest 16 passed; backend Research Studio capability pytest 15 passed; Bandit touched backend scope reported 0 findings; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed and resolved all live PR #1616 review threads. Fixed valid comments across Research Studio routing, SharedWithMe navigation, mobile tab persistence, no-source Studio regression coverage, extension alias redirects, hosted route visibility, capability derivation, ChatPane stale health refresh, portable manual screenshots, and Backlog readability. Posted evidence replies on review threads plus a PR summary for the outside-diff ChatPane finding. Verification recorded: focused UI Vitest 53 passed, frontend route/network Vitest 16 passed, backend Research Studio pytest 15 passed, Bandit touched backend scope 0 findings, and git diff --check clean. Remaining state: GitHub Actions are still pending after the force-push.
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
