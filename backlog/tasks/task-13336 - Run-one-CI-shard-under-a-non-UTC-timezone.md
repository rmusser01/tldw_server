---
id: TASK-13336
title: Run one CI shard under a non-UTC timezone
status: Done
assignee: []
created_date: '2026-09-22 04:59'
updated_date: '2026-09-23 23:10'
labels:
  - ci
  - tests
dependencies: []
references:
  - .github/workflows/ci.yml
  - 'tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:2489'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CI runs UTC, where the host-offset class of timestamp bug has a delta of exactly zero. That is why the confirmed ADR-014 violation in Evaluations created timestamps is invisible to 3,121 tasks worth of testing - and why 239 tz-naive datetime.utcnow() sites have never produced a failing test.

tldw_server is a self-hosted product whose target deployment is a user own machine with a real timezone, so UTC-only CI is the wrong validator for this class.

Cheapest durable guard in the whole core-module review: run ONE existing shard with TZ set to something like America/Los_Angeles. No new gate, no new job - an env var on a job that already exists, fitting the six contractual gates unchanged.

Source: synthesis F6 / migration plan Stage 0
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One existing CI shard runs under a non-UTC TZ
- [x] #2 The Evaluations timestamp test fails under it before the fix and passes after
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
1b48a30da1: full-suite-linux-312-shards job env gets TZ: ${{ matrix.shard.tz || 'UTC' }}; product-evaluations-unit (3.12 matrix only) sets tz: America/Los_Angeles. Every other shard and the 3.13/macOS/Windows/release copies are unchanged (explicit UTC == runner default). Contract test tests/CI/test_required_workflow_contracts.py::test_one_linux_312_shard_runs_under_non_utc_timezone pins it (fails on old ci.yml, passes now). AC2 evidence: the Evaluations fix already landed (TASK-13302, 7c348a05ae/a250262292). The pre-fix converter (7c348a05ae^ Evaluations_DB._ensure_unix_timestamp) on '2026-09-21 21:06:55' is off by 0s under TZ=UTC and by 25200s under TZ=America/Los_Angeles; the current to_unix_timestamp is 0 under both. Note test_created_timestamp_utc_contract.py also forces LA itself via tzset, so it guards even outside this shard; the shard TZ additionally exposes every other test in the shard to a non-zero offset. tests/Evaluations/unit locally: 267 passed under TZ=America/Los_Angeles and 267 passed under TZ=UTC (identical), so the shard should stay green. check_shard_coverage OK. Pre-existing unrelated failure in the same contract file: test_full_suite_splits_slow_chat_and_retrieval_shards (tests/Services/test_study_pack_startup_default.py expected in a shard set) fails identically on ea1cbc6941. No CI run observed (branch not pushed). Bandit on the touched test file: no findings. Docs: none (comment in ci.yml).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
product-evaluations-unit on Python 3.12 now runs with TZ=America/Los_Angeles via a per-shard matrix tz; pinned by a workflow contract test; pre-fix Evaluations converter demonstrably wrong by 25200s under it.
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
