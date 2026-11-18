---
id: TASK-15559
title: Analyze and save ingested media by reference
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/3503
- https://github.com/rmusser01/tldw-hosted/pull/296
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The public media modal sends stored source text twice. An incident-sized book exceeded the hosted JSON proxy limit. Use the existing chat and media version paths with authenticated stored media references. ADR required: no. This extends existing chat and media access patterns without changing their ownership or architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generation and save use small requests and preserve saved content.
- [x] #2 Oversized input or model context returns an actionable error.
- [x] #3 Public and hosted regression coverage includes the incident-sized payload and owner scoping.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: send a saved media ID and prompt from the browser; use the normal request-scoped Media database and chat provider/accounting pipelines. Reject inputs above 2 MiB and those above the selected model’s known context. Bind version saves to the analyzed active text with a SHA-256 precondition. Public changes are based on dev; hosted TASK-30 backports them to the current pinned artifact.
Independent review identified owner filtering, endpoint scope separation, and stale source races. The implementation uses an explicit owner predicate and a dedicated registered token endpoint ID. The active saved text takes precedence over transcript and historical versions. A changed source returns 409.
Verified 32 focused backend tests and 282 AnalysisModal/request-scope tests. The full 1,405,967-byte synthetic source reached the mocked provider for both SSE and JSON. A smaller simulated context returned a typed 413 before provider dispatch. OpenAPI fingerprint and path drift checks passed, patched hosted suites passed, and Bandit reported zero findings.
Merge review narrowed the lookup to the active text shown by Media and put the source digest check and version insert under the same write transaction. A restricted PostgreSQL 18 test confirmed a second connection cannot acquire a competing row lock until the version commits. The empty media path returned 422. First-frame stream cancellation now releases both the provider iterator and credentials. The updated focused suites passed in both source trees.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The browser sends a media reference and prompts. The backend reads only the current owner’s saved text and ties version persistence to the exact text analyzed. Tests cover incident-sized, oversized, changed and foreign sources.
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
