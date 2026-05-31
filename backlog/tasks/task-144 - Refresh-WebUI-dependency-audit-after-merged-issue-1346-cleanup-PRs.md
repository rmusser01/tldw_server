---
id: TASK-144
title: Refresh WebUI dependency audit after merged issue 1346 cleanup PRs
status: Done
created_date: '2026-05-09 03:34'
updated_date: '2026-05-09 03:40'
labels:
  - webui
  - dependencies
  - cleanup
  - docs
dependencies:
  - TASK-104
  - TASK-134
  - TASK-141
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1357'
  - 'https://github.com/rmusser01/tldw_server/pull/1359'
  - 'https://github.com/rmusser01/tldw_server/pull/1365'
  - 'https://github.com/rmusser01/tldw_server/pull/1368'
  - 'https://github.com/rmusser01/tldw_server/pull/1375'
  - 'https://github.com/rmusser01/tldw_server/pull/1385'
  - 'https://github.com/rmusser01/tldw_server/pull/1390'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the WebUI dependency audit for GitHub issue #1346 with current origin/dev after the already-merged cleanup PRs for pubsub-js, browser polyfills, hook-form, clsx, axios, lockfile-only declarations, and tooling declarations. The task should keep the audit actionable by removing stale quick-cleanup recommendations, marking completed slices with PR references, and identifying the next safe candidate category from current manifests rather than stale rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The audit completed-follow-ups section lists the merged issue #1346 PRs now present on origin/dev and their package-level outcomes.
- [x] #2 Rows and follow-up queues no longer recommend already-removed direct declarations as active quick-cleanup work.
- [x] #3 Current manifest and active-code evidence is rechecked for remaining platform-native replacement candidates before any next recommendation is recorded.
- [x] #4 Remaining dayjs/date-formatting work is classified with rationale that distinguishes simple display formatting from Ant Design DatePicker/Dayjs value contracts.
- [x] #5 Documentation-only verification is recorded, including git diff checks and a focused package-reference scan; Bandit skip rationale is recorded if no Python files change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed Docs/Design/WebUI_Dependency_Audit.md after confirming PR #1390 merged into origin/dev at 9ae5726bc01db212376316f6954e7eed589ec830. The audit now marks PR #1357, PR #1359, PR #1365, PR #1368, PR #1375, PR #1385, and PR #1390 as completed issue #1346 cleanup work, updates the stale rows for pubsub-js, @types/pubsub-js, buffer, stream-browserify, @hookform/resolvers, react-hook-form, axios, and clsx to removed, and removes those packages from the active quick-cleanup/replacement queue.

Current manifest evidence across apps/tldw-frontend/package.json, apps/packages/ui/package.json, and apps/extension/package.json found no direct declarations for pubsub-js, @types/pubsub-js, buffer, stream-browserify, @hookform/resolvers, react-hook-form, axios, or clsx. Exact active-code import scans found no package imports for those names. dayjs remains directly declared and imported only from shared UI source/tests, including simple display-formatting surfaces and Ant Design DatePicker/DateRangePicker Dayjs value contracts.

Verification: git diff --check exited 0; stale quick-cleanup rg for already removed package names exited 1 with no matches; the Node manifest check printed "removed issue-1346 package declarations absent"; the exact active-code package-import scan for removed names exited 1 with no matches; the dayjs scan listed the remaining shared UI imports. Bandit skipped because no Python files were modified.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the WebUI dependency audit after the merged issue #1346 cleanup PRs so it no longer recommends already-removed packages as active work. The audit now identifies dayjs/date-time formatting as the next compatibility/design target and records why direct dependency removal is blocked by Ant Design Dayjs value contracts.
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
