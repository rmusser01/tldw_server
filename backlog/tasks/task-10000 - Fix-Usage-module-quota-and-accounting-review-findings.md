---
id: TASK-10000
title: Fix Usage module quota and accounting review findings
status: Done
assignee: []
created_date: '2026-06-24 00:00'
updated_date: '2026-06-24 00:00'
labels:
  - usage
  - audio
  - resource-governance
  - code-review
dependencies: []
documentation: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the current-code review findings in `tldw_Server_API/app/core/Usage`: Resource Governor reservation idempotency, ResourceDailyLedger idempotency for repeated audio usage, atomic daily-minute consumption, best-effort LLM usage logging error boundaries, cancellation propagation, and pricing partial-match specificity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resource Governor stream/job reservations use unique operation IDs so concurrent limits are enforced.
- [x] #2 Audio daily-minute ledger writes do not deduplicate distinct equal-size usage events.
- [x] #3 Daily-minute quota check and consume is atomic for ledger-backed enforcement.
- [x] #4 `log_llm_usage` preserves the documented best-effort/no-raise behavior for backend errors.
- [x] #5 Quota helper catch-all behavior no longer swallows `asyncio.CancelledError`.
- [x] #6 Pricing partial-match lookup prefers the most specific known model entry.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing regression tests for each reviewed behavior.
2. Update Usage module implementations with minimal scoped fixes.
3. Run targeted Usage/Resource Governance tests, source-scope Bandit, and diff checks.
4. Update this Backlog task with verification results and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Manual Backlog task-file exception approved by the user because the Backlog CLI hung on task create/list/search in this repository during this session.

Implemented fixes:
- Added unique default audio-minute operation IDs while preserving explicit operation IDs for idempotent retries.
- Added ledger-backed atomic daily-minute consumption and wired audio endpoints to use the single consume operation.
- Changed Resource Governor stream/job reservation IDs to be unique per reservation.
- Preserved cancellation propagation by excluding `asyncio.CancelledError` from quota noncritical exception handling.
- Kept LLM usage logging best-effort for generic backend failures without logging sensitive exception text.
- Removed raw per-user LLM metric labels and kept durable usage logs as the user-level source of truth.
- Updated pricing fallback/placeholder handling and partial model lookup specificity.

PR review follow-up:
- Rebased `codex/usage-module-review-fixes-10000` onto latest `origin/dev` on 2026-06-24.
- Moved endpoint legacy quota fallback decision logic into `app/core/Usage/audio_quota.py`, leaving endpoint helpers as shim resolution plus core delegation.
- Updated unlimited-tier minute consumption so ledger unavailability cannot trigger bounded fail-open denial for users with `daily_minutes=None`.
- Added regression coverage for unlimited-tier ledger unavailability and core-owned legacy fallback behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all reviewed Usage module issues and added regression coverage for quota accounting, Resource Governor concurrency, cancellation propagation, best-effort usage logging, metric label cardinality, and pricing specificity.

Verification:
- `python -m pytest tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py::test_add_daily_minutes_writes_to_resource_daily_ledger tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py::test_consume_daily_minutes_if_allowed_is_atomic_and_idempotent tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py::test_can_start_stream_real_rg_enforces_limit_with_distinct_reservations tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py::test_can_start_job_real_rg_enforces_limit_with_distinct_reservations tldw_Server_API/tests/Audio/test_audio_quota_unit.py::test_can_start_stream_propagates_cancellation tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py::test_log_llm_usage_repo_backend_error_is_best_effort tldw_Server_API/tests/Usage/test_pricing_catalog_overrides.py::test_pricing_partial_match_prefers_most_specific_model tldw_Server_API/tests/Usage/test_usage_review_fixes.py -q` passed.
- `python -m pytest tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py tldw_Server_API/tests/Audio/test_audio_quota_unit.py tldw_Server_API/tests/Usage/test_usage_review_fixes.py tldw_Server_API/tests/Usage/test_audio_rg_minutes_and_heartbeat.py tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py tldw_Server_API/tests/Usage/test_pricing_catalog.py tldw_Server_API/tests/Usage/test_pricing_catalog_overrides.py tldw_Server_API/tests/Usage/test_pricing_catalog_path.py -q` passed: 77 tests after PR review follow-up.
- Post-logging tweak focused recheck passed: 2 tests.
- Bandit scoped to touched production files passed with no findings.
- Scoped `git diff --check` passed.
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
