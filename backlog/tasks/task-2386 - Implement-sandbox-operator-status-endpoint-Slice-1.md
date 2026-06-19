---
id: TASK-2386
title: Implement sandbox operator status endpoint Slice 1
status: Done
labels:
- sandbox
- operator-ux
- vz_linux
- implementation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 1 of sandbox operator/admin status consolidation from the approved design and plan: portable read-only status projection, service wrapper, schema, admin endpoint, tests, and docs. No evidence-file ingestion, generated_at, helper lifecycle mutation, launchd mutation, repair mutation, image-store cleanup mutation, or real VM execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read-only operator status projection exists and validates through schema.
- [x] #2 Admin-only GET /api/v1/sandbox/admin/operator-status endpoint returns structured payload.
- [x] #3 Unconfigured VZ/evidence does not degrade otherwise usable installs.
- [x] #4 Section failures are isolated and visible without preventing other sections from rendering.
- [x] #5 Docs and RBAC coverage are updated.
- [x] #6 Focused pytest and Bandit verification pass or any unrelated/pre-existing issues are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a pure `operator_status.py` projection that consolidates runtime diagnostics, macOS diagnostics, image-store/reconciliation posture, startup warnings, evidence placeholder state, and recommended actions without mutating helper, image-store, repair, launchd, or VM state.
- Hardened projection coercion so malformed booleans/integers do not become false-positive ready/actionable states; expected macOS section failures degrade an otherwise usable install, while runtime diagnostics failure is reported as `unknown`.
- Added `SandboxService.operator_status()` as a narrow wrapper over existing diagnostics sources. Generic `RuntimeError` propagates so programming/invariant failures are not hidden; expected operational failures become section-local `_section_error` payloads.
- Final review follow-up fixed configured-but-unready macOS/VZ classification so it reports an actionable `macos_vz` section and overall `action_required`, and exposed section `_section_error` values through `reasons` for operator troubleshooting.
- Added `SandboxAdminOperatorStatusResponse` schema and `GET /api/v1/sandbox/admin/operator-status` with admin RBAC, `asyncio.to_thread` offload, and startup-warning summary handoff.
- Updated API, Sandbox README, and macOS operator notes to document the read-only consolidated status endpoint and its non-goals.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Slice 1 of the sandbox operator status endpoint on `codex/sandbox-operator-status`. The endpoint is admin-only and read-only, returns the planned top-level status shape, validates through Pydantic, keeps unconfigured VZ/evidence non-degrading, isolates section failures with visible reasons, escalates configured-but-broken macOS/VZ readiness to action-required, and points operators back to detailed diagnostics/dry-run-first repair surfaces.

Verification:
- `python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload tldw_Server_API/tests/sandbox/test_admin_rbac.py::test_admin_endpoints_require_admin_role -q` -> 19 passed, 6 warnings.
- `git diff --check` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Sandbox/operator_status.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_operator_status_final_after_review.json` -> 0 findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Implementation follows Docs/superpowers/plans/2026-06-18-sandbox-operator-status-implementation-plan.md.
- [x] #2 No mutation or host-gated execution paths are introduced.
- [x] #3 Backlog task records verification commands and results.
- [x] #4 Changes are committed on codex/sandbox-operator-status.
<!-- DOD:END -->
