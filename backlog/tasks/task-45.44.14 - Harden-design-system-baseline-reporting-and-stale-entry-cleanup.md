---
id: TASK-45.44.14
title: Harden design-system baseline reporting and stale-entry cleanup
status: Done
assignee: []
created_date: 2026-05-14 03:20
updated_date: 2026-05-22 20:09
labels:
- design-system
- webui
- extension
- governance
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1671
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- Docs/Design/tldw_web_design_system_baseline_reporting.md
- https://github.com/rmusser01/tldw_server/pull/1948
documentation:
- Docs/Design/tldw_web_design_system_baseline_reporting.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns public status.
- [x] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Address Qodo media ownership by adding a failing guard test for `src/components/Media/...` product-area mapping, then update the ownership map.
2. Address Gemini reporter maintainability comments by removing the redundant fallback map entry, extracting shared grouping lines, and avoiding per-comparison normalization for constant ownership paths.
3. Address Qodo artifact discoverability by linking the baseline reporting doc from the parent Backlog epic and the governance task.
4. Re-run focused guard tests, live verifier, baseline JSON parse, and git diff checks; then update PR/GitHub threads.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added product-area grouping to the design-system product-state verifier using the tracker path ownership map, and added a stale-baseline cleanup summary so resolved baseline rows are visible without becoming blocking failures. Added focused guard tests for product-area totals and stale cleanup reporting, plus Docs/Design/tldw_web_design_system_baseline_reporting.md to document the migration PR count refresh and stale-row removal workflow. PR: https://github.com/rmusser01/tldw_server/pull/1948. Review follow-up: mapped `src/components/Media/...` into the Ingestion, Library, and media area with regression coverage, removed the redundant explicit fallback area entry, extracted shared baseline breakdown formatting, avoided per-comparison normalization for constant ownership paths, linked the baseline-reporting doc from the parent Backlog epic, and updated GitHub epic #1655 to link the doc and PR. Verification: red focused guard test failed on missing product-area and stale cleanup sections; red media ownership test failed while `src/components/Media/...` still mapped to fallback; green focused guard test passed 54/54; bun run verify:design-system-state passed with 303 baseline exceptions and no stale cleanup section; baseline JSON parse passed; git diff --check passed. Bandit skipped because this slice touches frontend JavaScript/TypeScript tests, markdown, and Backlog metadata only.
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
