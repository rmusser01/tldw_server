---
id: TASK-2392
title: Implement sandbox operator-status evidence ingestion Slice 2
status: Done
labels:
- sandbox
- operator-ux
- vz_linux
- implementation
modified_files:
- Docs/superpowers/plans/2026-06-19-sandbox-operator-evidence-ingestion-implementation-plan.md
- tldw_Server_API/app/core/Sandbox/operator_evidence.py
- tldw_Server_API/app/core/Sandbox/operator_status.py
- tldw_Server_API/app/core/Sandbox/service.py
- tldw_Server_API/tests/sandbox/test_operator_evidence.py
- tldw_Server_API/tests/sandbox/test_operator_status.py
- backlog/tasks/task-2392 - Implement-sandbox-operator-status-evidence-ingestion-Slice-2.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Slice 2 design for env-configured, read-only host-gated VZ smoke evidence bundle ingestion into the consolidated sandbox operator-status endpoint. Scope includes parser, projection/service integration, portable tests, and docs/task updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a bounded server-side parser for env-configured host smoke evidence directories without Markdown scraping or request-supplied paths.
- [x] #2 Project parsed evidence into the existing operator-status evidence section with advisory status/action behavior from the spec.
- [x] #3 Keep evidence ingestion read-only and fail closed for unsafe paths, symlinks, oversized/malformed JSON, unsupported schema, and parser operational errors.
- [x] #4 Add focused portable parser/projection/service tests covering success, invalid, stale, skipped, failed, and privacy-boundary cases.
- [x] #5 Validate with focused pytest, git diff --check, and Bandit on touched server files.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created implementation plan for Slice 2 evidence ingestion.
- Reviewed the plan against the spec and existing code boundaries.
- Corrected the initial plan to require descriptor-safe direct-child reads from the first parser commit, avoiding an unsafe intermediate direct-read parser.
- Added portability guidance for symlink-dependent parser tests.
- Added `operator_evidence.py` for env-configured, descriptor-safe, bounded host smoke evidence parsing.
- Projected normalized evidence into the consolidated operator-status `evidence` section with advisory actions for invalid, stale/skipped, and failed smoke evidence.
- Integrated evidence collection in `SandboxService.operator_status()` with operational failure isolation and programming-error propagation.
- Added portable parser, projection, and service tests; no real VZ execution is required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Slice 2 host smoke evidence ingestion for sandbox operator status, rebased PR #2412 against latest `origin/dev` with no new rebase changes, and addressed verified review findings. Review remediation preserved `expected_files` on malformed UTF-8, malformed JSON, and top-level non-object summaries; fixed bounded display strings to never exceed `DISPLAY_MAX_CHARS`; replaced tests' private `_dir_fd_operations_available` coupling with a public `safe_open_available` injection seam plus observable safe-open probing; and aligned evidence collection operational-failure fallback with the spec by reporting `available=False` so the evidence section projects as `unavailable`. Skipped the `test_admin_macos_diagnostics.py` casing suggestion because it is a literal filename, not prose. Verification: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py tldw_Server_API/tests/sandbox/test_operator_status.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload tldw_Server_API/tests/sandbox/test_admin_rbac.py::test_admin_endpoints_require_admin_role -q --tb=short` passed with 48 tests; `git diff --check` passed; Bandit on touched server files exited 0 with 0 findings in `/tmp/bandit_sandbox_operator_evidence_review2.json`.
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
