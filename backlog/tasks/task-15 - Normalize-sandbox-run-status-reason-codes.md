---
id: TASK-15
title: Normalize sandbox run status reason codes
status: Done
assignee: []
created_date: '2026-05-03 20:38'
labels:
  - sandbox
  - runtime
  - api
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an additive normalized run status reason-code layer for sandbox run status responses while preserving existing phase/message fields and avoiding storage migrations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox run status responses include a derived normalized status reason code without changing existing phase/message behavior.
- [x] #2 Admin run summaries include the same derived code for list views.
- [x] #3 The reason-code vocabulary is centralized and documented in sandbox runtime capability inventory/README.
- [x] #4 Focused tests cover queued runs, timeouts, cancellations, policy failures, and artifact/output limit signals where existing status data permits derivation.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Added `run_status_taxonomy.py` as the centralized additive taxonomy for stable run status reason codes.
- Wired `status_reason_code` into public run status responses and admin run summaries while preserving raw `phase`, `message`, and `exit_code`.
- Kept the code derived from existing status data to avoid DB migrations; list internals now carry `resource_usage` so admin summaries can classify limit-applied runs.

## Final Summary

Implemented additive sandbox run status reason codes for public and admin status surfaces, documented the vocabulary, and added focused tests for taxonomy mapping, schema exposure, SQLite list preservation, and queued POST/GET durability.

Verification:

- `python -m pytest tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q --timeout=60`
- `python -m py_compile tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/core/Sandbox/store.py`
- `python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/core/Sandbox/store.py -f json -o /tmp/bandit_run_status_reasons.json`
- `git diff --check`

Known caveat: a selected admin TestClient integration path timed out in existing app shutdown/background Jobs worker teardown; the status reason-code coverage was kept to deterministic taxonomy, schema, SQLite-store, and minimal sandbox API tests.
