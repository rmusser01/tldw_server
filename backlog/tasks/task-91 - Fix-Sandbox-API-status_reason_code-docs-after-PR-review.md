---
id: TASK-91
title: Fix Sandbox API status_reason_code docs after PR review
status: Done
assignee: []
created_date: '2026-05-06 00:04'
updated_date: '2026-05-06 00:10'
labels:
  - sandbox
  - docs
  - pr-review
dependencies:
  - TASK-89
documentation:
  - Docs/API-related/Sandbox_API.md
  - tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #1330 review feedback that Docs/API-related/Sandbox_API.md documents status_reason_code outcomes with labels that do not match the server's RunStatusReasonCode literals. Keep the change docs-only and verify against run_status_taxonomy.py and the capability inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox API guide lists the exact status_reason_code literals returned by the server.
- [x] #2 Verification confirms the documented literals align with RunStatusReasonCode and the existing capability inventory table.
- [x] #3 Whitespace verification passes and Bandit is documented as skipped because the change is docs-only.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the generic/hyphenated status_reason_code outcome list in Docs/API-related/Sandbox_API.md with the exact RunStatusReasonCode literals from tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py.
2. Cross-check the updated API guide against Docs/Sandbox/sandbox-runtime-capability-inventory.md so both docs use the same vocabulary.
3. Run focused docs verification and whitespace checks, record the docs-only Bandit skip, then commit and push the PR review fix.
<!-- SECTION:PLAN:END -->

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

<!-- SECTION:NOTES:BEGIN -->
Verified the PR finding against `tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py` and the existing normalized run status table in `Docs/Sandbox/sandbox-runtime-capability-inventory.md`. Updated `Docs/API-related/Sandbox_API.md` to list the exact returned literals instead of hyphenated or collapsed labels. Verification: `rg` confirmed the exact literals appear in the API guide, capability inventory, and canonical taxonomy; `git diff --check` passed. Bandit skipped because this is a docs/backlog-only change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL:BEGIN -->
Fixed PR #1330 review feedback by replacing the Sandbox API guide's generic/hyphenated `status_reason_code` outcome names with the exact `RunStatusReasonCode` literals returned by the server: `limits_applied`, `nonzero_exit`, `policy_failed`, `runtime_unavailable`, `startup_timeout`, `execution_timeout`, `canceled_by_user`, `queue_ttl_expired`, and the other canonical values. This keeps public API documentation aligned with the taxonomy implementation and capability inventory. Verification: focused `rg` cross-check and `git diff --check`; Bandit skipped because no production code changed.
<!-- SECTION:FINAL:END -->
