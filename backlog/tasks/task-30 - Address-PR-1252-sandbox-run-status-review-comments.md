---
id: TASK-30
title: Address PR 1252 sandbox run status review comments
status: Done
assignee:
  - Codex
created_date: '2026-05-04 03:01'
labels:
  - sandbox
  - runtime
  - api
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1252'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix verified review comments on PR #1252 for sandbox run status reason-code normalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Taxonomy classifies limit signals before phase-specific completed/failed/timed_out outcomes where existing usage data permits derivation.
- [x] #2 Runtime-unavailable matching is narrow and explicitly covers VZ RuntimeUnavailable-derived messages without treating ordinary missing/not-found command errors as runtime availability issues.
- [x] #3 New helper/module maintainability comments are addressed with type hints and module docstring.
- [x] #4 Malformed resource_usage JSON in store list paths is logged without breaking resilient list responses.
- [x] #5 Focused sandbox tests and security/format checks pass for the touched scope.
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

1. Tighten `normalize_run_status_reason()` so limit signals take precedence for completed, failed, and timed-out runs.
2. Narrow runtime-unavailable matching so explicit runtime/provisioning messages classify as `runtime_unavailable`, but ordinary command `missing`/`not found` failures do not.
3. Add focused regression tests for the taxonomy changes, including the VZ RuntimeUnavailable-derived messages.
4. Add the missing endpoint helper type hints and taxonomy module docstring.
5. Log malformed `resource_usage` JSON in SQLite/Postgres list paths while keeping list responses resilient.
6. Re-run focused tests, compile checks, `git diff --check`, and Bandit before pushing the PR update.

## Implementation Notes

- Moved limit-signal detection ahead of completed/failed/timed-out phase classification so output/artifact ceilings produce `limits_applied` consistently.
- Added explicit VZ RuntimeUnavailable-derived message handling and narrowed the generic runtime-unavailable matcher to structured runtime/provisioning context.
- Added endpoint helper parameter types and a module docstring for the new taxonomy module.
- Logged malformed SQLite/Postgres `resource_usage` JSON at debug level while preserving resilient list behavior.

## Final Summary

Addressed the open PR #1252 review comments by tightening taxonomy classification, adding focused regression coverage, documenting the taxonomy module contract, typing the endpoint helper, and making malformed `resource_usage` decoding observable without breaking list responses.

Verification:

- `python -m pytest tldw_Server_API/tests/sandbox/test_run_status_contract_durability.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q --timeout=60`
- `python -m py_compile tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/core/Sandbox/store.py`
- `python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/core/Sandbox/store.py -f json -o /tmp/bandit_run_status_reasons_review_fix.json`
- `git diff --check`

Known caveat: GitHub Actions were still queued when this local review-fix pass was prepared.
