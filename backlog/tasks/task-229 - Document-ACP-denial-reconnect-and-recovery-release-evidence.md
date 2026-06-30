---
id: TASK-229
title: Document ACP denial reconnect and recovery release evidence
status: Done
assignee: []
created_date: '2026-05-10 15:16'
updated_date: '2026-05-10 15:48'
labels:
  - acp
  - release-signoff
  - e2e
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1501'
  - 'https://github.com/rmusser01/tldw_server/issues/1500#issuecomment-4415656791'
  - 'https://github.com/rmusser01/tldw_server/pull/1517'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - apps/packages/ui/src/hooks/useACPSession.tsx
  - apps/tldw-frontend/e2e/workflows/tier-3-automation/
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the #1501 ACP release-signoff workstream by inventorying existing denial, reconnect, and recovery coverage; adding focused deterministic coverage where practical; documenting live-E2E limitations; and posting evidence back to GitHub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing backend, frontend, and Playwright coverage for permission denial, reconnect/session replay, and failed-run recovery is documented with concrete file references.
- [x] #2 Focused deterministic tests cover ACP UI deny/reconnect behavior where current coverage is incomplete.
- [x] #3 Release readiness docs distinguish verified automated coverage from manual/live-backend caveats.
- [x] #4 Any blocker or limitation is linked as a follow-up issue or accepted release caveat.
- [x] #5 Verification includes focused frontend/backend tests, formatting checks, and Bandit for touched Python scope or a documented skip if no Python code changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory existing backend, frontend, and browser ACP evidence for permission denial, reconnect/session replay, and failed-run recovery.
2. Add focused deterministic useACPSession hook coverage for permission denial payload cleanup and transient reconnect progress.
3. Update release readiness docs and changelog with #1501 evidence and live-agent caveats.
4. Verify focused frontend/backend suites, diff hygiene, and security-scan applicability.
5. Package PR and post GitHub evidence to #1501/#1500.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented deterministic useACPSession tests for denial response payload/queue cleanup and transient close retry state.
Updated ACP_Production_Readiness with a #1501 addendum covering permission denial, reconnect/session replay, failed-run recovery, and explicit live downstream-agent caveats tied to #1504/#1505.
Verification: Vitest passed for useACPSession, ACPPermissionModal, ACPChatPanel, and AgentTasksPage.connection (19 tests). Pytest passed for selected ACP permission/reconnect/SSE/replay/orchestration diagnostics coverage (18 tests). git diff --check passed. Bandit skipped because touched source is Markdown plus TypeScript test code only; no Python code was changed.

Review follow-up for PR #1517: replaced the reconnect test hardcoded 1000ms timer advance with WS_CONFIG.RECONNECT_DELAY_MS, added an explicit supported downstream-agent table documenting that no live ACP stdio downstream agent/version is certified for this release host, and added the #1500 parent-tracker evidence link to task references.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
#1501 release evidence now has deterministic frontend hook tests for denial and reconnect behavior plus a readiness addendum mapping existing backend/frontend/browser evidence to release posture and caveats. Review follow-up added an explicit supported-agent table showing no live downstream ACP stdio agent/version is certified for this release host, linked the parent #1500 evidence comment, and tied reconnect timer coverage to WS_CONFIG.RECONNECT_DELAY_MS. Live downstream-agent permission denial and live browser reconnect remain explicitly caveated until #1504/#1505 provide the seeded/live-agent environment.
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
