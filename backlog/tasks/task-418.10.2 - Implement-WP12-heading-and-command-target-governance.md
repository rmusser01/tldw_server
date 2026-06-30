---
id: TASK-418.10.2
title: Implement WP12 heading and command target governance
status: Done
labels:
- wp12
- webui
- route-governance
priority: High
parent_task_id: TASK-418.10
references:
- TASK-418.10
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/1953
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 2 from the WebUI route governance QA plan: add metadata-backed heading governance and command palette route-target governance without page-level redesign or backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Route heading governance covers active smoke inventory routes with metadata-backed exceptions for aliases, redirects, hosted-only, sidepanel-only, and internal/debug routes.
- [x] #2 Command palette route-target tests validate route labels, targets, duplicate labels, and visibility against route metadata.
- [x] #3 Any required metadata additions stay scoped to governance fields and preserve existing route intent.
- [x] #4 Focused Vitest and Playwright route-governance checks pass, with unrelated baseline failures documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented route metadata helpers for command palette labels and h1 policy.
- Added command target governance Vitest coverage for labels, targets, duplicate labels, and command-palette visibility.
- Added frontend-owned route heading governance Vitest coverage and browser coverage for primary self-hosted smoke routes.
- Kept Notes out of strict one-h1 enforcement with a metadata-backed exception because user-authored note content can contain document h1s.
- Aligned Knowledge command copy and English locale strings with route metadata.
- Browser QA initially found `/media` and `/notes` heading failures. `/media` was a test harness issue resolved with the same media API mocks used by existing smoke coverage; `/notes` is now excluded from strict one-h1 browser enforcement through a documented metadata exception.
- Addressed PR review by moving the metadata-only heading test out of `@tldw/ui`, replacing the command label CSS selector with metadata-backed label lookup, and adding direct coverage for the `/settings/health` command-label override.
- Addressed follow-up review by adding a non-empty browser route guard and enforcing dedicated `h1ExceptionReason` values for explicit h1 opt-outs.
- Bandit is not applicable to this frontend-only TypeScript/markdown slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
WP12 Task 2 route heading and command target governance implemented. Focused Vitest governance/command/notes accessibility tests passed. Browser heading governance passed for 11 primary self-hosted smoke routes. Broad Notes test directory run from apps/tldw-frontend showed unrelated/baseline failures in AI title, contrast CSS path, backlink labels, and inline snapshot setup; not caused by this slice and not used as a gate.
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
