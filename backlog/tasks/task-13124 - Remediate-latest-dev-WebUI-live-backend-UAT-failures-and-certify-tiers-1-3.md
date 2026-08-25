---
id: TASK-13124
title: Remediate latest-dev WebUI live-backend UAT failures and certify tiers 1-3
status: In Progress
assignee: []
created_date: '2026-08-25 15:03'
updated_date: '2026-08-25 15:30'
labels:
  - webui
  - uat
  - live-backend
  - playwright
  - remediation
dependencies: []
references:
  - origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb
documentation:
  - >-
    Docs/superpowers/specs/2026-08-25-webui-live-backend-uat-remediation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate every confirmed product, development-runtime, and live-server release-gate issue discovered during the 2026-08-25 exhaustive WebUI UAT on origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb. After the initial child tasks, run the complete Playwright Tier-1, Tier-2, and Tier-3 projects against a real isolated backend, create child tasks for every newly confirmed defect, fix them test-first, and repeat the affected tiers until failures are resolved or explicitly classified as unsupported-provider/environmental skips.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All eight initial UAT findings have dedicated child tasks with evidence-backed root causes and regression coverage.
- [ ] #2 Every newly confirmed Tier-1 through Tier-3 product defect receives a child Backlog task and test-first remediation.
- [ ] #3 Affected tier suites are rerun after remediation and remaining skips/failures are explicitly justified with captured evidence.
- [ ] #4 The exhaustive route sweep and dedicated real-server suites are rerun after fixes.
- [ ] #5 Frontend typecheck, lint on touched scope, focused unit/integration tests, and applicable security validation pass.
- [ ] #6 The user worktree remains untouched; work is committed in the isolated codex branch.
- [ ] #7 The complete Tier-1, Tier-2, and Tier-3 inventories run against isolated real backend and deterministic model-service processes with offline fallback disabled.
- [ ] #8 API-intercepted and mocked scenarios are inventoried and are not misreported as live-backend evidence; critical mock-only coverage gains a supported live counterpart.
- [ ] #9 Before final certification the branch is synchronized with then-current origin/dev and all final tier results record the exact commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Pre-implementation review revised the design to: compare bundlers with bounded memory/responsiveness criteria before selecting a default; inventory mocked/intercepted tier cases separately from live-backend evidence; use the repository mock OpenAI-compatible service through the real backend; coverage-map and delete redundant legacy live tests; gate saved views only on narrow workspace existence; re-prove the Prompt layout issue cleanly; use the generic Kanban error mechanism; and synchronize with then-current origin/dev before final certification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
