---
id: TASK-405
title: Rebaseline sandbox roadmap after VZ stability PRs
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 06:56'
labels:
  - sandbox
  - docs
  - roadmap
  - vz_linux
dependencies: []
references:
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the sandbox roadmap/status documentation after the merged VZ Linux stability slices so contributors can see which immediate queue items are complete, which items remain host-gated/manual, and what the next pragmatic implementation slice should be. Keep this docs/status slice narrow and do not change runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Roadmap immediate queue shows current completion/status for items 1-8 instead of presenting completed work as pending.
- [x] #2 Roadmap identifies the next pragmatic implementation slice after merged VZ stability work, with clear host-gated/manual boundaries.
- [x] #3 Docs link the current acceptance-policy, capability inventory, security matrix, and public/operator docs as evidence rather than duplicating their details.
- [x] #4 Verification notes record docs-only checks and explicitly skip Bandit with rationale if no Python/runtime code changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-16-sandbox-roadmap-rebaseline-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the sandbox roadmap spec from a stale pending immediate queue into a rebaselined status map. Added source-of-truth links for the capability inventory, security matrix, host-gated CI acceptance policy, and public/operator docs. Added the next pragmatic queue with prepared-host acceptance evidence first, followed by remaining lifecycle drill gaps, operator/admin status consolidation, and expansion only after evidence. Updated the Sandbox README pointer to note the roadmap is now status-rebaselined.

Verification: rg smoke check found updated queue headings and source-of-truth references; ls verified referenced docs/workflow exist; focused docs/workflow pytest passed with 22 tests; git diff --check passed. Bandit skipped because this slice changed only Markdown docs and Backlog task metadata.

PR review follow-up: replaced the placeholder `rg ...` verification snippet with a copy/paste-safe command using explicit file targets, linked the expansion queue item to Phase 6, and clarified `vz_macos` as the exact runtime identifier rather than changing it to non-canonical mixed-case spelling. Verification rerun: exact `rg` command passed, focused docs/workflow pytest passed with 22 tests, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebaselined the sandbox roadmap after the initial VZ Linux stability slices. The roadmap now distinguishes completed foundations from residual host-gated/manual boundaries, identifies prepared-host acceptance evidence as the next pragmatic work, and preserves expansion guardrails for the `vz_macos` runtime, vmnet, APFS clones, Firecracker parity, and Apple containerization. Verification passed for reference smoke checks, focused docs/workflow tests, and diff hygiene; Bandit was not applicable for docs-only changes.
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
