---
id: TASK-427
title: Address PR 1838 review comments
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/1838
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257061627
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257061636
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257064174
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257064185
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257064189
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257064194
- https://github.com/rmusser01/tldw_server/pull/1838#discussion_r3257064204
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx
- backlog/tasks/task-424 - Create-phased-PRD-for-Watchlists-digest-and-audio-briefing-workflow.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable Qodo and CodeRabbit review feedback on PR #1838 for the Watchlists digest/audio PR1 branch. Scope is limited to nested scrape_rules preservation, interval scheduling consistency/constants, localized diagnostics labels, and minor task grammar cleanup. Do not broaden the PR beyond current Watchlists review findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Preserve unknown nested `settings.scrape_rules` keys while allowing UI-owned scrape rule fields to be updated or cleared.
- [x] Keep variable cadence preset parsing consistent with the 5-minute minimum and shared interval bounds.
- [x] Localize source diagnostics labels and address the minor task grammar comment.
- [x] Resolve stale/non-actionable and remediated PR review threads after verification and push.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed Qodo/CodeRabbit findings on PR #1838. `buildSourceSettingsPayload` now removes only UI-owned scrape rule keys before merging current form values back into existing nested rules, so backend/advanced keys like pagination survive edits. Schedule interval bounds are exported from `schedule-utils` and reused by the picker and cron builder/parser; sub-5-minute step crons now fall back to advanced cron instead of being silently clamped by presets. Source diagnostics labels now go through the existing `t` helper. The stale Gemini thread targeted a pre-rebase file that is no longer in the PR diff and was resolved without code churn.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1838 review feedback with focused frontend/test/task updates. Verification: focused Watchlists review-fix Vitest slice passed, 7 files and 28 tests; existing SourcesTab regression slice passed, 4 files and 14 tests; `git diff --check` passed. Bandit skipped because no Python files were touched in this review-fix pass.
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
