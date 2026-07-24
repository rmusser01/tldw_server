---
id: TASK-12986
title: Sequence expensive PR CI behind trusted frontend license status
status: In Progress
assignee: []
created_date: '2026-07-24 04:08'
updated_date: '2026-07-24 04:16'
labels:
  - ci
  - github-actions
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-23-trusted-license-first-ci-sequencing-design.md
  - >-
    Docs/superpowers/plans/2026-07-23-trusted-license-first-ci-sequencing-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep lightweight PR diagnostics immediate while preventing matrix, Docker, browser, full-test, packaging, and security jobs from starting until the branch-qualified trusted frontend license status succeeds for the exact pull-request head SHA. Preserve existing repository rulesets and avoid privileged workflow_run chaining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A shared status-wait contract checks the exact pull-request head SHA and base-qualified trusted status context.
- [ ] #2 Expensive PR jobs do not start before the trusted status succeeds; failure or timeout fails closed.
- [ ] #3 Actionlint, pre-commit, and docs-only checks remain immediate.
- [ ] #4 Manual workflow_dispatch behavior remains available without pull-request metadata.
- [ ] #5 Existing rulesets and the trusted frontend-license publisher remain unchanged.
- [ ] #6 Focused workflow contract tests and actionlint pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User approved the narrow shared-wait-job design on 2026-07-23. Live rulesets currently require only frontend-license-policy/trusted/main and /dev; all other CI checks are informational. The sequencing layer is resource control, not a replacement trust boundary.

Isolated worktree created from origin/dev at codex/trusted-license-first-ci. Baseline verification passed: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py (54 passed, 2 warnings). Approved design written to Docs/superpowers/specs/2026-07-23-trusted-license-first-ci-sequencing-design.md.

Independent spec review iteration 1 found non-PR/unsupported-base event handling and caller status-read permissions underspecified. The design now passes through all non-PR events and PR bases other than main/dev, grants each wait-call job contents:read plus statuses:read, fixes polling at 30 seconds for 15 minutes with a 17-minute job timeout, and requires always() rollups to guard on gate success. Iteration 2 approved the revised spec with no blocking issues.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
