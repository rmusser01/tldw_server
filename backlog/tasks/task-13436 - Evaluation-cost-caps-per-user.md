---
id: TASK-13436
title: Evaluation cost caps per user
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from spec 2. Cost caps never fired: every caller passes estimated_cost=0.0 and the monthly cap is never compared. Needs real cost estimation, then limits.evaluation_cost_per_day_usd / _per_month_usd. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The evaluation-only cost caps are removed (no cost field, parameter, header or YAML key remains in the evaluations limiter, its config, GET /evaluations/rate-limits or the frontend type), and evaluation LLM spend is capped by limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd (PR 2a, #3230)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Re-scoped by the owner on 2026-10-09: a general LLM cost quota (limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd) replaces evaluation-only cost caps, and the dead evaluation cost caps are deleted. Spec: Docs/Design/2026-10-09-llm-usage-accounting-design.md, sections 7 and 8; delivered in PR 2 and PR 3 of that spec.
PR 3: the evaluation cost caps are removed: RateLimitConfig and the tier config lose max_cost_per_day/month, the daily reservation's cost branch and every cost parameter of check_rate_limit, _reserve_request_usage, _record_request and record_actual_usage go, /evaluations/rate-limits and the X-RateLimit-*-Cost-Remaining headers stop reporting cost, and the frontend type loses the six cost fields. The DB columns stay (no migration). Evaluation LLM spend is capped by limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The evaluation cost caps never fired: every caller passed a cost of 0.0, the monthly cap was never compared, and /evaluations/rate-limits reported the tier's numbers truncated to integers. PR 3 removes them from the limiter, its config and YAML, the endpoints, the response headers, the frontend type and the docs; the database columns stay (no migration). Evaluation LLM spend is capped by the general limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd quotas. Acceptance criterion 1 was reworded to the 2026-10-09 re-scope.
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
