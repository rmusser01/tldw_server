---
id: TASK-87
title: Add cross-runtime sandbox diagnostics summary
status: Done
assignee: []
created_date: '2026-05-05 21:01'
updated_date: '2026-05-05 21:37'
labels:
  - sandbox
  - diagnostics
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow read-only operator diagnostics summary for all sandbox runtimes using existing runtime discovery/preflight truth. This follows the sandbox roadmap Phase 5 direction: help operators understand runtime readiness and warnings without generalizing vz_linux repair or mutating diagnostics state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new additive sandbox admin diagnostics summary exposes per-runtime readiness posture derived from existing runtime discovery/preflight data.
- [x] #2 The summary clearly distinguishes available runtimes from unavailable/scaffold/host-gated runtimes and includes raw plus normalized reason codes where available.
- [x] #3 The design preserves vz_linux-specific reconciliation/repair ownership and does not add generic repair behavior for Docker, Firecracker, Lima, seatbelt, worktree, or vz_macos.
- [x] #4 Focused tests cover mixed runtime states and prove the summary is read-only and derived from existing discovery data.
- [x] #5 Relevant sandbox docs mention the new operator summary without overstating host-local or scaffold runtime guarantees.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive GET /api/v1/sandbox/admin/runtime-diagnostics backed by SandboxService.feature_discovery(), with schema coverage, admin RBAC coverage, startup warning projection, and docs updates. Verification: focused pytest 28 passed; py_compile passed; ruff F/E9 passed; Bandit touched production files produced zero findings; git diff --check passed. Full default Ruff remains blocked by pre-existing file-level I/SIM/B904 findings in touched files.

PR #1328 review follow-up: Qodo posted five actionable findings. Verified against current branch: async handler calls synchronous runtime_diagnostics_summary directly, host_gated/scaffold summary uses implementation_state instead of readiness, new helpers lack docstrings, and new test monkeypatch parameter lacks type annotation.

PR #1328 review follow-up fixed: runtime diagnostics now offloads synchronous discovery with asyncio.to_thread, summary host_gated/scaffold counts use readiness rather than static implementation_state, new helpers have docstrings, and the new test monkeypatch parameter is typed. Added regression coverage for offloading and ready host-gated runtime aggregation.

PR #1328 CodeRabbit follow-up: accepting fail-open startup-warning helper handling; skipping the bucket reclassification suggestion because it conflicts with the readiness-posture contract and would again make available host-gated runtimes not count as ready.

PR #1328 CodeRabbit accepted fix: _sandbox_startup_warning_summary now fail-opens to {present:false, blocking:false, codes:[]} when registry is missing or errors, and both diagnostics endpoints always project the same summary shape. Added regression tests for missing and broken registries. Kept readiness-based bucket logic; CodeRabbit's alternative implementation-state-first suggestion was skipped because it conflicts with the documented readiness posture semantics.

PR #1328 CodeRabbit bucket follow-up: re-verified the second comment and found a still-valid double-count issue in unavailable because it included all non-ready runtimes. Fixing with readiness-based mutually exclusive buckets, not implementation_state-first classification.

PR #1328 final CodeRabbit bucket fix: unavailable now counts only readiness values unavailable/unsupported/not_applicable, making ready/host_gated/scaffold/unavailable mutually exclusive readiness buckets. Regression expectation updated for available host_gated runtime plus host-gated unavailable runtime.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an admin-only, read-only cross-runtime diagnostics endpoint backed by existing sandbox runtime discovery. The response groups runtime readiness, preserves raw and normalized reason codes, surfaces host-local isolation warnings and startup warning summaries, and keeps repair semantics scoped to runtimes that explicitly advertise repair support. PR review follow-up fixed async event-loop blocking, mutually exclusive readiness aggregation, docstrings, test typing, and fail-open startup-warning projection. Verification: targeted sandbox pytest 31 passed, py_compile passed, Ruff F/E9 passed, Bandit touched production scope reported zero findings, test-file Ruff passed, and git diff --check passed. Full default Ruff on all touched production files remains blocked by pre-existing file-level I001/SIM300/B904 findings.
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
