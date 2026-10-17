---
id: TASK-13639
title: Promote test-quality ratchet to CI gate
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recreated under a fresh ID: original TASK-13407 assignment collided on dev with an unrelated task (Restore standalone documentation manifest) - task IDs are reused across concurrent sessions; see AGENTS.md collision warning. Work executed in batch-4 worktree commit bd7448d6fb.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Executed 2026-10-14 (commit bd7448d6fb on codex/credit-batch4-uat-hardening). Baseline regenerated 743->1107 (July baseline had 412 drifted offenses - enforcing it would fail every PR). Three exemplar fixes: tautology_suspect (admin audit: assert emitted fields + log-before-flush order), mock_density (dsr: parametrized 5-mock test split into per-case tests via shared fixture), status_only (notification: assert payload forwarded into alert). stub_injection/ambiguous_accept exemplars ruled idiomatic/intentional per the 2026-07-04 audit guidance. Wrapper test tldw_Server_API/tests/CI/test_test_quality_ratchet.py wired into backend-required.yml ratchet step; green under CI env in 103s; red/green verified via baseline tamper.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ratchet is a required CI gate: wrapper test (tests/CI/test_test_quality_ratchet.py) in backend-required.yml enforces --enforce against the regenerated baseline (743->1107 offenses, matching live scan totals exactly). Three exemplar fixes strengthen assertions against real production contracts (admin audit fields+ordering per admin_audit_service.py:89-105; DSR split preserves all 4 case behaviors; notification payload forwarding per monitoring.py:568-583). Red/green independently re-verified by review (baseline tamper -> exit 1 naming offense). Review: Approved, minors only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
