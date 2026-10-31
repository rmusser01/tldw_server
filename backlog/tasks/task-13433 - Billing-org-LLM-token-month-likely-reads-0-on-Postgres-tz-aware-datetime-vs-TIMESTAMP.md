---
id: TASK-13433
title: >-
  Billing: org LLM-token month likely reads 0 on Postgres (tz-aware datetime vs
  TIMESTAMP)
status: To Do
assignee: []
created_date: '2026-10-03 02:25'
labels:
  - billing
  - bug
  - postgres
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the spec 2 review. BillingEnforcer._get_llm_tokens_month (core/Billing/enforcement.py ~384-410) passes a tz-aware month_start to a query on llm_usage_log.ts, which is TIMESTAMP without time zone on Postgres (pg_migrations_extra.py ~1919). asyncpg rejects aware datetimes for timestamp columns with DataError (a ValueError subclass), which _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS swallows, so the hosted LLM_TOKENS_MONTH usage probably reads 0. Not yet confirmed against a live Postgres.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On Postgres, org monthly LLM-token usage equals the sum of llm_usage_log tokens for the org's members in the month
- [ ] #2 A test using the AuthNZ Postgres fixture covers it
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
