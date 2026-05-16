---
id: TASK-386
title: Implement cache-aware LLM usage reporting
status: Done
assignee: []
created_date: '2026-05-15 16:29'
updated_date: '2026-05-15 16:44'
labels:
  - llm-cache
  - usage-reporting
  - cost-control
  - implementation
dependencies:
  - TASK-377
  - TASK-378
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend admin LLM usage reporting after cache-aware usage rows are populated. Reports should expose bounded aggregate cache metrics and estimate-source counts while keeping raw usage metadata and local diagnostics separate from paid-provider billing cache fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LLM usage log and summary reporting can show prompt tokens, cached input tokens, cache write/read tokens, billable input tokens, output tokens, estimated cost, and estimate source.
- [x] #2 Reports distinguish paid-provider billing cache metrics from local inference diagnostics and do not expose raw_usage_metadata_json.
- [x] #3 Existing usage reporting behavior remains backward-compatible when cache-aware columns are missing or null.
- [x] #4 CSV/API tests cover cache-aware aggregates and redaction behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 8 scope located: admin LLM usage reporting is implemented in tldw_Server_API/app/services/admin_usage_service.py, exposed through tldw_Server_API/app/api/v1/endpoints/admin/admin_usage.py, and typed by tldw_Server_API/app/api/v1/schemas/admin_schemas.py. The implementation should use llm_usage_log cache-aware columns and avoid raw_usage_metadata_json.

Implemented admin usage log, summary, and CSV cache-aware reporting in admin_usage_service.py and admin_schemas.py. Added endpoint coverage for provider cache aggregates, local diagnostic counts, legacy table fallback, and raw metadata redaction.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added cache-aware LLM admin reporting fields for list, summary, and CSV export. Reports now surface cached input tokens, cache write/read tokens, billable input tokens, estimate source, and estimate-source aggregate counts while keeping raw_usage_metadata_json out of admin responses. Verified with focused admin usage endpoint tests, related usage tests, py_compile, git diff --check, and Bandit with no findings.
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
