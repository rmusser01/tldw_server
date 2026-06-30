---
id: TASK-382
title: Implement cache-aware LLM usage persistence
status: Done
assignee: []
created_date: '2026-05-15 15:29'
updated_date: '2026-05-15 15:39'
labels:
  - usage
  - chat
  - cost-control
  - llm-cache
  - implementation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the approved chat/world-book cache cost-control plan. Persist normalized cache usage fields and cache-aware costs while keeping existing usage inserts compatible with old schemas and preserving current cost behavior when provider cache pricing is unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SQLite and Postgres migrations add nullable llm_usage_log cache/diagnostic columns without breaking existing rows.
- [x] #2 Repository insert logic writes new normalized fields when columns exist and falls back cleanly for pre-migration schemas.
- [x] #3 Pricing catalog and compute_costs support optional cache-read and cache-write rates while preserving current behavior when rates are missing.
- [x] #4 log_llm_usage persists normalized cache fields and cache-aware costs without changing existing prompt/completion/total token semantics.
- [x] #5 Focused migration, pricing, and usage tests cover new fields plus compatibility fallback.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing migration/pricing/persistence tests for new llm_usage_log cache columns, cache pricing rates, and log_llm_usage normalized field persistence.
2. Extend SQLite and Postgres usage-table DDL with nullable cache/diagnostic columns.
3. Extend pricing catalog and compute_costs with optional cache-read/cache-write rates while preserving legacy get_rates behavior.
4. Extend AuthnzUsageRepo.insert_llm_usage_log with full insert plus compatibility fallbacks for old schemas.
5. Wire NormalizedLLMUsage through log_llm_usage into repository insert and raw metadata JSON.
6. Run focused migration/usage/pricing tests, diff checks, Bandit, update plan/task, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused Stage 3/4 suite passed (27 passed); SQLite/pricing/usage focused suite passed (14 passed); optional Postgres migration test skipped due local fixture availability; git diff --check passed. Bandit run on touched Python scope reported one existing B608 at migrations.py:616 outside this diff; no new Bandit findings were introduced in changed lines.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 4 cache-aware LLM usage persistence. Added nullable SQLite/Postgres llm_usage_log cache and diagnostic columns, extended usage inserts with pre-088/pre-054 compatibility fallbacks, added optional cache_read/cache_write pricing support, and wired normalized cache usage into persisted fields and cache-aware cost calculation while preserving legacy prompt/completion/total token columns.
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
