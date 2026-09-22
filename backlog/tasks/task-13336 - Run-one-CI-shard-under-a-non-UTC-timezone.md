---
id: TASK-13336
title: Run one CI shard under a non-UTC timezone
status: To Do
assignee: []
created_date: '2026-09-22 04:59'
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
- [ ] #1 One existing CI shard runs under a non-UTC TZ
- [ ] #2 The Evaluations timestamp test fails under it before the fix and passes after
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
