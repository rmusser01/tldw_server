---
id: TASK-468
title: Address Watchlists PR 1907 follow-up review findings
status: Done
labels:
- watchlists
- review-fix
- demo-readiness
priority: high
modified_files:
- Docs/superpowers/plans/2026-05-21-watchlists-pr1907-followup-review-fixes.md
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- apps/packages/ui/src/services/watchlists-overview.ts
- apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts
- apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.health.test.tsx
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Watchlists/pipeline.py
- tldw_Server_API/tests/Watchlists/test_run_detail_filters_totals.py
- tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
- backlog/tasks/task-440 - Design-staged-remediation-plans-for-Watchlists-demo-readiness-issues.md
- backlog/tasks/task-442 - Implement-Watchlists-demo-rescue-slice.md
- backlog/tasks/task-468 - Address-Watchlists-PR-1907-follow-up-review-findings.md
references:
- https://github.com/rmusser01/tldw_server/pull/1907
- https://github.com/rmusser01/tldw_server/pull/1906
- https://github.com/rmusser01/tldw_server/pull/1915
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix still-valid Watchlists demo rescue review findings after PR #1906 landed: audio status precedence, status-only audio failure detection, run detail filter counter preservation, JSON-style secret redaction, health test fixture typing, and Backlog metadata formatting. Leave already-fixed retry/raw-exception findings unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio status summaries prefer audio-specific fields and normalize mixed-case statuses before display.
- [x] #2 Watchlists overview counts failed audio outputs even when legacy metadata lacks explicit request flags.
- [x] #3 Run detail stats preserve and coerce valid filter counters while hiding filter_tallies unless requested.
- [x] #4 Source error redaction removes common quoted, JSON-style, and bearer-token secrets before storage/logging.
- [x] #5 Targeted frontend, backend, diff, and Bandit verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-21-watchlists-pr1907-followup-review-fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #1915 review fixes applied locally: normalized mixed-case audio statuses, coerced run-detail filter action counts from ints/floats/numeric strings while skipping bools, strengthened quoted/JSON/bearer secret redaction, populated acceptance criteria, and aligned the RunDetailDrawer stream lifecycle expectation with the normalized queued audio-task label. Local verification passed for Watchlists a11y gate, focused Vitest, focused pytest, git diff --check, and Bandit on touched backend files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the still-valid Watchlists PR #1907 follow-up findings on a clean dev-based branch. The patch keeps audio status truthful, preserves legacy filter counters in run detail stats, expands source-error redaction for JSON-style secrets, repairs the typed health fixture and Backlog metadata, and records targeted verification.
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
