---
id: TASK-122
title: Add structured sandbox run status reason metadata
status: Done
assignee: []
created_date: '2026-05-08 03:16'
labels:
  - sandbox
  - runtime-taxonomy
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-08-sandbox-status-reason-details-design.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py
  - tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow Phase 3 sandbox slice that exposes structured metadata for existing sandbox run status_reason_code values. Keep this strictly additive: do not change runner behavior, stored run rows, phases, raw messages, or existing status_reason_code literals. The goal is to let clients and operator surfaces display severity/category/retry/action guidance without runtime-specific message parsing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public run status responses expose additive structured details for the existing status_reason_code value without removing or renaming existing fields.
- [x] #2 Admin run summary/detail responses expose the same additive structured details derived from the same central metadata contract.
- [x] #3 Every RunStatusReasonCode literal has complete metadata and a focused completeness test fails if a new code is added without metadata.
- [x] #4 Focused tests cover representative queued/running/completed/limit/policy/runtime-unavailable/timeout/canceled/nonzero/unknown metadata behavior and schema exposure.
- [x] #5 Sandbox capability/API documentation records the structured metadata contract and updates the current Phase 3 gap wording without overstating runtime guarantees.
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design approved and captured in `Docs/superpowers/specs/2026-05-08-sandbox-status-reason-details-design.md`. Implementation followed `Docs/superpowers/plans/2026-05-08-sandbox-status-reason-details-plan.md` using a narrow additive response-field slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created a fresh worktree from `origin/dev` at merge commit `182a29c2e` on branch `codex/sandbox-status-reason-details`. The design keeps the API additive: derive `status_reason_details` from existing `status_reason_code` without changing runner behavior, persisted run rows, phases, raw messages, or existing code literals.

Implemented `RunStatusReasonDetails`, literal metadata types, `RUN_STATUS_REASON_METADATA`, import-time completeness validation, and `run_status_reason_details()` in `run_status_taxonomy.py`. Added nullable `status_reason_details` fields to public and admin schemas and populated them in public start/status responses plus admin list/detail responses.

Updated `Docs/API-related/Sandbox_API.md` and `Docs/Sandbox/sandbox-runtime-capability-inventory.md` to document the additive details object while preserving the Phase 3 gap that runtime discovery `normalized_reasons` still lack equivalent rich details.

Verification recorded:
- RED: taxonomy metadata tests failed before metadata symbols existed.
- RED: schema test failed before `status_reason_details` existed.
- RED: docs guard failed before inventory updates.
- GREEN: `python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q` passed with 17 tests.
- GREEN: the four endpoint response tests passed individually: queued POST status, POST/GET consistency, admin details resource usage, and admin list filters/pagination.
- Static: `python -m py_compile ...` passed for touched production modules.
- Static: `python -m ruff check ...` passed for touched Python files after safe import/style fixes.
- Security: `python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_sandbox_status_reason_details.json` reported zero results.
- Whitespace: `git diff --check` passed.

Known limitation: grouping the four TestClient endpoint checks in one pytest process hung in the existing TestClient shutdown path after the first test. The same endpoint checks passed as isolated pytest processes, so this slice records the grouped run as an existing lifecycle limitation rather than a status metadata failure.

PR review follow-up: addressed two Gemini Code Assist threads by adding bounded caching for status reason detail schema objects and strengthening taxonomy validation so each metadata key must match its internal `code`. Added a focused mismatch regression test.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added additive structured sandbox run status reason metadata to public and admin responses, backed by a complete central taxonomy map and docs/tests that guard the contract. No runner behavior, persisted status rows, existing phases, raw messages, or `status_reason_code` literals were changed.
<!-- SECTION:FINAL_SUMMARY:END -->
