---
id: TASK-12766
title: Preserve ACP conservative support guardrails for Aider Continue and custom
  profiles
status: Done
labels:
- ACP
- guardrail
- github-2399
references:
- https://github.com/rmusser01/tldw_server/issues/2399
- https://github.com/rmusser01/tldw_server/issues/2398
- https://github.com/rmusser01/tldw_server/pull/2417
modified_files:
- IMPLEMENTATION_PLAN_acp_support_guardrails_2399.md
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2399. Reconcile and harden the conservative ACP support state for Aider, Continue, and the seeded custom profile across registry tests, setup-guide responses, smoke manifests, docs, runner config, and Agent Registry UI. This is a guardrail audit only; do not certify or upgrade support for these profiles without new live evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Aider remains documented_unverified/documented_only and is modeled only as an unverified external aider-acp adapter candidate until adapter evidence exists.
- [x] #2 Continue remains documented_unverified/documented_only with no ACP stdio command or maintained adapter identified.
- [x] #3 The seeded custom profile remains a template-only/non-runnable profile requiring a distinct named profile and evidence bundle before support claims.
- [x] #4 Setup guide, registry/smoke manifest tests, compatibility docs, runner config, and Agent Registry UI surfaces are audited for overclaims.
- [x] #5 Verification results, skipped surfaces, and final issue updates are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Reproduced the stale Aider guardrail test failure on current dev: `test_default_agents_yaml_keeps_aider_blocked_without_acp_entrypoint` failed because Aider is now correctly modeled as `external_acp_adapter` with `acp_command=aider-acp`.
- Updated the Aider registry test to assert the current conservative adapter-candidate model while keeping `support_state=documented_unverified` and `verification_level=documented_only`.
- Added setup-guide guardrail coverage for Aider, Continue, and the seeded custom profile so runtime/setup metadata cannot silently become a release support claim.
- Audited `agents.yaml`, `ACP_Compatibility_Matrix.md`, `ACP_OSS_Custom_Certification_2026_05_11.md`, bundled runner config, helper smoke manifests, Go runner passive status copy, and Agent Registry UI copy. No support overclaims were found; no docs/UI/runner edits were needed.
- UI Vitest was attempted but blocked by this worktree's incomplete Bun dependency symlink layout (`vitest/config` / `@testing-library/jest-dom/vitest` resolution). Since no UI files changed and the UI audit found conservative copy, this is recorded as an environment skip.
- Verification so far: focused pytest for ACP registry/entrypoint/setup-guide/smoke manifests passed with 139 tests; `git diff --check` passed; Bandit on touched Python tests reported only B101 assert baseline, and rerun with `-s B101` reported `results=[]`, `errors=[]`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented as test guardrails only. Updated the Aider default registry assertion to match the current conservative external adapter candidate model while preserving `documented_unverified` / `documented_only`. Added setup-guide regression coverage for Aider, Continue, and the seeded custom profile so entrypoint metadata cannot be mistaken for certified support. No production code, docs, runner config, or UI code required changes after audit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved ACP conservative support guardrails for GitHub #2399 in PR #2417. Verification: focused ACP pytest batch passed (139 passed, 6 warnings); git diff --check passed; Bandit on touched Python tests passed with B101 test-assert baseline excluded and reported results=[], errors=[]. UI Vitest was attempted but skipped because this fresh worktree lacks the Bun dependency symlink layout needed to resolve vitest/config and @testing-library/jest-dom/vitest; no UI files changed and UI copy was audited as conservative.
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
