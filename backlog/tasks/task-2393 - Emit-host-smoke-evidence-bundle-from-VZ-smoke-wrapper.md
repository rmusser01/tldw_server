---
id: TASK-2393
title: Emit host smoke evidence bundle from VZ smoke wrapper
status: Done
labels:
- sandbox
- vz_linux
- operator-ux
- implementation
modified_files:
- Docs/superpowers/plans/2026-06-19-vz-smoke-evidence-bundle-implementation-plan.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
- tools/vz-linux-image/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved operator workflow closeout slice: make run-host-e2e-smoke.sh produce the canonical host smoke evidence bundle consumed by sandbox operator-status, expose an evidence-dir option/default, print server wiring guidance, update docs, and add portable tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `run-host-e2e-smoke.sh --dry-run` prints the default evidence directory without creating it.
- [x] `run-host-e2e-smoke.sh --dry-run --evidence-dir PATH` prints the override without creating it.
- [x] The wrapper prints a sourceable `export TLDW_SANDBOX_VZ_EVIDENCE_DIR=...` handoff in dry-run plans.
- [x] The wrapper prints the same handoff after successful real-run evidence finalization.
- [x] Operator-facing docs explain how to use the printed env var with `operator-status`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-19-vz-smoke-evidence-bundle-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Current `dev` already contained default evidence directory handling, expected sidecars, JSON evidence, private evidence directory checks, and failure-preserving cleanup evidence.
- This slice was narrowed to the missing operator handoff contract: printing a sourceable `TLDW_SANDBOX_VZ_EVIDENCE_DIR` export in dry-run plans and after successful real-run evidence finalization.
- Tests were added to assert help advertises `--evidence-dir PATH`, dry-run output includes the export for default and override paths, and real fake-helper runs print the export after writing evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added `print_evidence_env_hint()` to `run-host-e2e-smoke.sh` and call it from dry-run evidence planning and successful evidence finalization.
- Updated host smoke wrapper tests for the env handoff contract.
- Documented using the printed `TLDW_SANDBOX_VZ_EVIDENCE_DIR` value to surface local smoke evidence in `GET /api/v1/sandbox/admin/operator-status`.
- Verification: focused wrapper tests passed (`25 passed, 3 skipped`), `bash -n` passed, `git diff --check` passed, and Bandit produced the same 122 findings as the `dev` baseline for the touched test file.
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
