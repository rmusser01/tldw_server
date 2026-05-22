---
id: TASK-479
title: Risk Gate 6 prototype owner review and promotion UX hardening
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 19:47'
labels:
  - prototype-workspaces
  - risk-gate
  - frontend
  - product
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1458'
  - 'https://github.com/rmusser01/tldw_server/issues/1440'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
  - Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1458 under tracker #1440. Harden the Frontend/Product owner review and promotion UX so owners can distinguish review, runtime, validation, stale/conflict, and promotion failure states; promotion actions align with backend authority and validation semantics; branch/session inventory supports review decisions; and focused frontend tests cover the main owner review transitions. Keep backend changes limited to verified contract gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Owner workspace detail exposes promotion request inventory for review.
- [x] #2 Owner UI distinguishes promotion review status from branch runtime and preview status.
- [x] #3 Frontend review mutation preserves structured contract errors and invalidates workspace detail plus promotion query state.
- [x] #4 Focused backend/frontend verification and Bandit completed.
- [x] #5 Approve actions require pending requests with present candidate snapshot and actionable branch session; pending reject remains available for stuck revoked/missing-snapshot requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-prototype-risk-gate-6-owner-review-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the narrow backend contract gap by adding promotion request summaries to owner workspace detail through PrototypeWorkspacesRepo, Pydantic response schemas, and the detail endpoint builder. Added frontend review input/result types, review service method, TanStack mutation hook, and owner review UI that renders review state separately from runtime/preview health. Added regression coverage for repository filtering/ordering, endpoint response shape, hook request/error/invalidation behavior, and owner-view review/actionability states.

Verification on 2026-05-22 after rebasing onto origin/dev:
- Addressed PR #1945 review comments: explicit validation-running/validation-failed/promotion-failed owner copy; separate approve/reject predicates matching backend reject semantics; structured error frontend_state/category rendering; defensive filtering of unparseable promotion request rows.
- bunx vitest run src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceSessionView.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceOwnerView.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx --maxWorkers=1 --no-file-parallelism: 4 files, 28 tests passed.
- source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q: 112 passed, 5 warnings.
- source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py -f json -o /tmp/bandit_prototype_risk_gate_6_review_fixes.json: 0 findings.
- git diff --check HEAD: passed.
- NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false: exits 2 with 284 existing repo-wide diagnostics; filtering the output for touched PrototypeWorkspace files returned no matches.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Risk Gate 6 owner review UX is implemented. Owners can see promotion request inventory in workspace detail, review pending requests from the owner surface, and distinguish review status from runtime/preview health. Promotion review and promotion creation hooks now refresh the workspace detail state that carries review inventory. PR review fixes keep reject available for stuck pending requests, render structured failure states, explicitly cover validation/promotion failure states, and avoid empty promotion-request rows in API detail responses.

Known skips/blockers: no required verification skipped. Repo-wide UI typecheck remains blocked by existing diagnostics outside the touched PrototypeWorkspace files; AI-generated PR still needs the required human-written Change summary before merge.
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
