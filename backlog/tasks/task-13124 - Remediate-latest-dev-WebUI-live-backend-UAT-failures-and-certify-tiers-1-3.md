---
id: TASK-13124
title: Remediate latest-dev WebUI live-backend UAT failures and certify tiers 1-3
status: In Progress
created_date: 2026-08-25 15:03
labels:
- webui
- uat
- live-backend
- playwright
- remediation
priority: high
references:
- origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb
documentation:
- Docs/superpowers/specs/2026-08-25-webui-live-backend-uat-remediation-design.md
updated_date: 2026-08-25 15:07
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate every confirmed product, development-runtime, and live-server release-gate issue discovered during the 2026-08-25 exhaustive WebUI UAT on origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb. After the initial child tasks, run the complete Playwright Tier-1, Tier-2, and Tier-3 projects against a real isolated backend, create child tasks for every newly confirmed defect, fix them test-first, and repeat the affected tiers until failures are resolved or explicitly classified as unsupported-provider/environmental skips.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All eight initial UAT findings have dedicated child tasks with evidence-backed root causes and regression coverage.
- [ ] #2 The complete Tier-1, Tier-2, and Tier-3 Playwright projects run against an isolated real backend, not offline or mocked fallback mode.
- [ ] #3 Every newly confirmed Tier-1 through Tier-3 product defect receives a child Backlog task and test-first remediation.
- [ ] #4 Affected tier suites are rerun after remediation and remaining skips/failures are explicitly justified with captured evidence.
- [ ] #5 The exhaustive route sweep and dedicated real-server suites are rerun after fixes.
- [ ] #6 Frontend typecheck, lint on touched scope, focused unit/integration tests, and applicable security validation pass.
- [ ] #7 The user worktree remains untouched; work is committed in the isolated codex branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
