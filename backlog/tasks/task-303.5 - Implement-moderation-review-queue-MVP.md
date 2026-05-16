---
id: TASK-303.5
title: Implement moderation review queue MVP
status: Done
assignee: []
created_date: '2026-05-12 23:32'
labels:
  - moderation
  - frontend
  - ux
dependencies:
  - TASK-303.4
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 5 of the moderation review/rules remediation plan. Build the /moderation review queue UI against the Stage 4 review backend contract, including service functions, queue state, filters, list/detail surfaces, decision controls, empty/loading/error/permission states, compact extension behavior, and focused frontend tests. Keep this slice scoped to review queue MVP behavior; bulk workflows, audit timeline, retention/redaction UI, and advanced keyboard workflows remain later stages unless required for MVP correctness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Moderation service types and functions cover Stage 4 review list detail decision undo bulk and audit endpoints with tested query/body construction.
- [x] #2 /moderation loads review items through queue state with filters for status category severity source user search sort cursor and selected item.
- [x] #3 Review queue toolbar list and detail surfaces show required fields including status severity category phase source user or session created time sanitized excerpt recommended action matches policy and safe-field warnings.
- [x] #4 Decision controls support approve block redact dismiss and escalate with required reasons for block redact and escalate plus confirmation for destructive or escalation actions.
- [x] #5 Single-item decision success refreshes queue state updates the selected item and exposes undo when the backend returns an undo token.
- [x] #6 Empty loading error permission-denied backend-unsupported and partial-data states are rendered without redirecting users to content rules.
- [x] #7 Compact extension mode keeps queue controls usable in narrow width and offers an Open full review action.
- [x] #8 Focused service and component tests cover query state filter refresh state panels decision validation and undo display.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented Stage 5 review queue MVP in the WebUI/extension shared ModerationReview surface. The queue now has service functions for list/detail/decision/undo/bulk/audit endpoints, queue state, supported filters, local sort, selected item detail, decision controls, undo affordance, state panels, compact mode, and focused tests.
- Kept Stage 5 aligned with the actual Stage 4 backend contract: date filters are not exposed in the UI because the backend review list endpoint does not support date_from/date_to yet. Date/audit filters remain later-stage scope.
- Verification 2026-05-12: `bunx vitest run src/components/Option/ModerationReview/__tests__/ModerationReviewShell.test.tsx src/components/Option/ModerationReview/__tests__/review-utils.test.ts src/services/__tests__/moderation.service.contract.test.ts` passed, 3 files and 13 tests.
- Verification 2026-05-12: `bun run verify:openapi` passed with 265 ClientPath entries verified, 49 MEDIA_ADD_SCHEMA_FALLBACK fields verified, and the existing 10 reviewed exception paths allowed.
- Verification 2026-05-12: `TLDW_WEB_URL=http://127.0.0.1:18055 TLDW_WEB_AUTOSTART=false bunx playwright test e2e/workflows/tier-5-specialized/moderation-review.spec.ts --project=tier-5 --reporter=line` passed, 2 tests.
- Verification 2026-05-12: `TLDW_WEB_URL=http://127.0.0.1:18055 TLDW_WEB_AUTOSTART=false bunx playwright test e2e/workflows/tier-5-specialized/moderation-routes.spec.ts --project=tier-5 --reporter=line` passed, 4 tests.
- Verification 2026-05-12: Playwright CDP script `/private/tmp/moderation-review-stage5-browser-check.mjs` passed desktop and 390px mobile checks with no console errors and no horizontal overflow; screenshots are under `output/playwright/moderation-review-stage5`.
- Verification 2026-05-12: `git diff --check` passed.
- Bandit: not run because this Stage 5 slice only touches frontend TypeScript/TSX, E2E specs, docs, and Backlog metadata.
- Known baseline issue: `bun run verify:design-system-state` still exits 1 because of existing stale AntD product-state baseline entries in AgentRegistry and AgentTasks, not Stage 5 ModerationReview files. Stage 5 labels were moved through `getDesignSystemState` where applicable.
- Known baseline issue: `bunx tsc -p tsconfig.json --noEmit` still exits 2 with repo-wide pre-existing TypeScript errors in audio/chat/flashcards/playground/etc.; visible output did not implicate Stage 5 ModerationReview files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL:BEGIN -->
Implemented the moderation review queue MVP against the Stage 4 backend contract. Added typed service coverage for review endpoints, queue state/filtering, responsive list/detail views, decision and undo behavior, state panels, compact-mode affordances, focused Vitest coverage, and mocked Playwright E2E coverage for list-detail-decision-undo plus mobile overflow. Date filters and audit/prior-decision history are intentionally deferred to later stages because they require backend audit/date support.
<!-- SECTION:FINAL:END -->
