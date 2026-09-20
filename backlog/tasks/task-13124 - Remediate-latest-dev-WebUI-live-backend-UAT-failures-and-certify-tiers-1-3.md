---
id: TASK-13124
title: Remediate latest-dev WebUI live-backend UAT failures and certify tiers 1-3
status: Done
assignee: []
created_date: '2026-08-25 15:03'
updated_date: '2026-08-26 04:43'
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
  - Docs/superpowers/plans/2026-08-25-webui-live-backend-uat-remediation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate every confirmed product, development-runtime, and live-server release-gate issue discovered during the 2026-08-25 exhaustive WebUI UAT on origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb. After the initial child tasks, run the complete Playwright Tier-1, Tier-2, and Tier-3 projects against a real isolated backend, create child tasks for every newly confirmed defect, fix them test-first, and repeat the affected tiers until failures are resolved or explicitly classified as unsupported-provider/environmental skips.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All eight initial UAT findings have dedicated child tasks with evidence-backed root causes and regression coverage.
- [x] #2 Every newly confirmed Tier-1 through Tier-3 product defect receives a child Backlog task and test-first remediation.
- [x] #3 Affected tier suites are rerun after remediation and remaining skips/failures are explicitly justified with captured evidence.
- [x] #4 The exhaustive route sweep and dedicated real-server suites are rerun after fixes.
- [x] #5 The user worktree remains untouched; work is committed in the isolated codex branch.
- [x] #6 The complete Tier-1, Tier-2, and Tier-3 inventories run against isolated real backend and deterministic model-service processes with offline fallback disabled.
- [x] #7 API-intercepted and mocked scenarios are inventoried and are not misreported as live-backend evidence; critical mock-only coverage gains a supported live counterpart.
- [x] #8 Before final certification the branch is synchronized with then-current origin/dev and all final tier results record the exact commit.
- [x] #9 Frontend typecheck reports no diagnostics in the touched scope; remaining baseline diagnostics are documented, and touched-scope lint, focused tests, build, and applicable security validation pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the approved five-stage implementation plan inline: remediate TASK-13124.1 through TASK-13124.8 test-first, build the isolated live-tier runner under TASK-13124.9, run complete Tier-1 through Tier-3 inventories, create/fix child tasks for new confirmed findings, synchronize current origin/dev, and certify the exact final commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Pre-implementation review revised the design to: compare bundlers with bounded memory/responsiveness criteria before selecting a default; inventory mocked/intercepted tier cases separately from live-backend evidence; use the repository mock OpenAI-compatible service through the real backend; coverage-map and delete redundant legacy live tests; gate saved views only on narrow workspace existence; re-prove the Prompt layout issue cleanly; use the generic Kanban error mechanism; and synchronize with then-current origin/dev before final certification.

2026-08-25: Requester confirmed the revised design. Detailed implementation plan completed and self-reviewed; execution proceeds inline without subagent delegation.

Final exact-commit certification used 81a36bef786eed82540b23e59a4d6c485db51321, which contains origin/dev b1d0aed671dcf45bbe4211a9690022c083c99feb as an ancestor. Full typecheck has no diagnostics in touched files; remaining baseline diagnostics are documented. Two focused backend tests were environment-dependent skips; the live browser certification had no skips.

Closeout review corrected the typecheck acceptance criterion to match the scoped gate actually verified; it does not claim the unrelated repository-wide baseline is green.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All 43 child tasks are complete. Exact-commit real-backend UAT passed Tier 1 34/34, Tier 2 104/104, and Tier 3 37/37 with zero retries, skips, failures, or interruptions; health before/after and teardown all passed. Review, regression, build, lint, security, isolation, and evidence results are recorded in Docs/superpowers/reviews/2026-08-25-tier-1-3-live-uat-results.md.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
