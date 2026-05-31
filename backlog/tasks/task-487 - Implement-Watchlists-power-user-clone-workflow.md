---
id: TASK-487
title: Implement Watchlists power-user clone workflow
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 00:13'
labels:
  - watchlists
  - power-user
  - frontend
  - ux
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-22-watchlists-power-user-clone-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Watchlists power-user follow-up: clone monitor and clone source-rule workflows with preservation smoke coverage, while preserving existing news/OSINT/CTI advanced flows and avoiding new backend batch APIs unless required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source clone action is available from /watchlists Sources and opens the existing source form with copied settings/tags/group/extraction/dedupe-style configuration, while resetting runtime/status/seen state and inactive-by-default safety.
- [x] #2 Focused preservation tests cover clone behavior and critical existing advanced watchlists workflows are not regressed.
- [x] #3 No backend batch APIs are added for this slice unless frontend composition is unsafe or impossible.
- [x] #4 Focused frontend tests and relevant verification are run and recorded.
- [x] #5 Monitor clone action remains available from /watchlists Monitors as an inactive/paused copy, with preservation covered by existing clone utility tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Adjusted after code inspection: monitor clone and source clone actions already exist on origin/dev. Monitor clone creates a paused copy and will stay as-is unless tests reveal regression. The unsafe gap is source clone immediately POSTing a duplicate URL, which can fail before the user can edit it. This slice changes source clone to open a prefilled create form, preserves copied source rules/group assignment on save, and covers the behavior with focused frontend tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation changed source clone from immediate create API call to a create-mode SourceFormModal draft. The cloned draft preserves name suffix, URL/type/tags/settings/group_ids/watchlist_id and applies active:false on save. Existing monitor clone behavior remains direct paused-copy creation and is covered by clone utility tests. No backend APIs were added.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed source clone review-form flow and preservation coverage. Addressed code review by rebasing onto latest dev to remove unrelated route-governance/design-system files from the PR diff and by adding real SourceFormModal clone-draft create-mode submit coverage. Verification: focused Watchlists Vitest suite passed; git diff --check passed. Full UI tsc was attempted with an increased heap and failed on existing repo-wide baseline errors outside this Watchlists slice. Bandit skipped because only frontend/markdown/backlog files were touched.
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
