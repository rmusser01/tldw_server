---
id: TASK-438
title: Address PR 1864 Watchlists review comments
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address live review comments and CI failures for PR #1864, including Watchlists recovery endpoint safety, frontend retry/focus behavior, Backlog checklist consistency, and targeted verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live PR #1864 review findings are verified against current code before patching.
- [x] #2 Delegated recovery operations use the target user's Watchlists DB and Collections DB.
- [x] #3 Delivery retry and diagnostics output only sanitized delivery status summaries.
- [x] #4 Frontend retry and focus states accurately reflect skipped retries and modal close behavior.
- [x] #5 Focused backend/frontend tests, Watchlists gates, diff check, and Bandit are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resolved review findings for PR #1864 by adding target-user Collections DB resolution, async threadpool wrapping for synchronous output DB work, canonical text-artifact selection for delivery retries, sanitized delivery metadata/diagnostics, cross-user email fallback prevention, retry-skip warning states, per-row clone loading state, shared download helper reuse, quick-setup focus restoration hardening, deterministic admin webhook timing, and stronger operator recovery tests. Also checked the completed PR5 task checkboxes in TASK-436.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all currently actionable PR #1864 review comments and the observed CI failures. Verification passed locally: `python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q`, `python -m pytest tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py::TestWebhookDeliveryRecording::test_record_and_list_deliveries -q`, `bunx vitest run src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`, `bun run test:watchlists:typecheck`, `bun run test:watchlists:scale`, `bun run test:watchlists:a11y`, `git diff --check`, and Bandit on touched backend files with `-s B101` for test asserts. Known skip: Bandit B101 is intentionally skipped for test files only.
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
