---
id: TASK-325
title: Implement ACP execution health summary contract for issue 1537
status: Done
assignee:
  - codex
created_date: '2026-05-14 01:14'
updated_date: '2026-05-14 01:58'
labels:
  - ACP
  - admin
  - reporting
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1537'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/issues/1512'
  - 'https://github.com/rmusser01/tldw_server/issues/1513'
  - 'https://github.com/rmusser01/tldw_server/issues/1529'
documentation:
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
  - Docs/Development/ACP_Compatibility_Matrix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first #1537 slice as a backend-owned ACP execution-health summary contract that admins/operators and future UI surfaces can consume. The slice should aggregate existing ACP/session/orchestration/registry signals without inventing a separate observability system, and should keep downstream UI work as a follow-up once the API contract is stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP execution-health summary metrics and failure buckets are documented in a durable ACP development/product doc.
- [x] #2 A backend API/service path exposes summary data for ACP sessions/runs/reviews/setup-health using existing storage and status sources where available.
- [x] #3 Summary output distinguishes setup blockers, runner/session failures, reviewer rejections, governance denials, retention/redaction status, and documented-unverified live-agent compatibility states without overstating support.
- [x] #4 Focused tests cover the summary contract, empty-state behavior, representative failure buckets, and redaction/retention status fields.
- [x] #5 Issue #1537 can be updated with backend contract evidence and clearly split follow-up frontend/admin display work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented backend ACP execution-health summary contract with session status counts, failure buckets, retention/redaction posture, compatibility support-state rollup, and fail-closed compatibility enum coercion.

Verification: focused execution-health pytest 3 passed; full admin ACP endpoint pytest 20 passed; git diff --check passed; Bandit touched backend scope reported 0 findings; OpenAPI schema generation exposed /api/v1/admin/acp/execution-health/summary with fake test SINGLE_USER_API_KEY.

PR: https://github.com/rmusser01/tldw_server/pull/1648. Issue update: https://github.com/rmusser01/tldw_server/issues/1537#issuecomment-4446577031.

Review-fix pass for PR #1648: addressing Qodo/Gemini comments on failure bucket taxonomy, setup-health dimensions, Backlog dependency/follow-up links, endpoint type/logging/style issues, aggregation placement, lookback filtering, pagination, and N+1 session loads.

Review-fix coordination links added for #1512 retention, #1513 redacted views, and #1529 admin/deployment baseline.

Trackable follow-up split: backend aggregation hardening covers DB/pre-aggregated execution-health metrics beyond the summary contract; frontend display covers admin UI cards/tables and drill-through entry points; docs/verification covers release evidence, OpenAPI examples, and compatibility matrix signoff. These follow-ups coordinate under #1537 and the linked ACP-adjacent issues rather than being claimed complete by this backend contract slice.

Review-fix verification: PASS pytest tldw_Server_API/tests/Admin/test_admin_acp_new_endpoints.py -q (23 passed); PASS git diff --check; PASS py_compile touched backend files; PASS OpenAPI route/schema smoke for execution-health setup_health; Bandit touched backend scope reports only pre-existing baseline findings in ACP_Sessions_DB.py, with 0 findings in the new execution_health.py and admin endpoint/service changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ACP execution-health backend summary contract and opened draft PR #1648. The contract exposes /api/v1/admin/acp/execution-health/summary with session status counts, normalized failure buckets, retention/redaction posture, per-agent setup and compatibility summary, documented-unverified live-certification flags, and fail-closed enum handling. Validation: admin ACP endpoint pytest 20 passed, git diff --check clean, Bandit touched backend scope 0 findings, OpenAPI registration verified with fake test SINGLE_USER_API_KEY. Issue #1537 was updated with PR evidence and remaining frontend/admin display follow-up.

Review-fix pass added core execution-health aggregation, DB-filtered paginated/batched session loading, expanded failure/setup-health taxonomy, fail-closed timestamp handling, Backlog coordination links, and focused regression tests.
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
