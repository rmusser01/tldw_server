---
id: TASK-45.44.9.13
title: Migrate TakeQuizTab alerts and quiz-state badge to design-system primitives
status: Done
labels:
- design-system
- webui
- product-state
- quiz
priority: medium
parent_task_id: TASK-45.44.9
references:
- apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/components/ui/primitives/Alert.tsx
- apps/packages/ui/src/components/ui/primitives/__tests__/Alert.test.tsx
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system product-state migration by replacing TakeQuizTab's remaining product-state AntD Alert and flagged quiz/result Tag usage with shared design-system primitives while preserving existing quiz flow behavior, copy, dismissal, and actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TakeQuizTab product-state Alerts render through the shared design-system Alert primitive while preserving existing copy, dismissal, and actions.
- [x] #2 The TakeQuizTab product-state quiz/result status Tag renders through the shared design-system Badge primitive where flagged by the product-state guard.
- [x] #3 Focused tests assert the migrated states render inside design-system primitive markers.
- [x] #4 The matching TakeQuizTab product-state baseline entries are removed and the touched-file design-system guard passes.
- [x] #5 Verification records focused tests, product-state guard tests, design-system verifier status, diff whitespace, TypeScript status, and Bandit applicability.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated TakeQuizTab hint, score summary, assignment, start-confirmation, autosave, queued-submission, review/practice guidance, practice feedback, unanswered-warning, and highlight-notice callouts from AntD Alert to the shared design-system Alert primitive.
- Migrated the review/practice mode labels and result correctness labels to the shared design-system Badge primitive while leaving non-product quiz metadata AntD tags alone.
- Added TakeQuizTab design-system regression coverage for list-level notices, modal notices, study-mode guidance, graded result states, and queued submission recovery.
- Removed the nine TakeQuizTab product-state baseline exceptions; direct guard API over TakeQuizTab now reports no product-state guard issues.
- Full `bun run verify:design-system-state` is currently blocked by 24 unrelated current-dev findings outside TakeQuizTab, primarily ScheduledTasks plus Skills/KnowledgeQA/Onboarding/ACP readiness drift.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated TakeQuizTab product-state Alerts and flagged quiz-state Tags to design-system Alert/Badge primitives, added focused design-system regression coverage, and removed the nine matching TakeQuizTab baseline exceptions. Review follow-up rebased the PR on latest dev, split the bundled TakeQuizTab design-system regression tests into 16 focused state cases, made Alert ignore null/boolean/whitespace-only children before rendering its body wrapper, added Alert primitive regression coverage for whitespace-only children, and made the result correctness Badge spacing explicit with flex/gap classes plus a text span. Verification: RED Alert regression failed before the primitive fix; Alert regression now passes 1/1; focused TakeQuizTab design-system suite passes 16/16; combined product-state guard plus adjacent TakeQuizTab suites pass 102/102; direct guard API over TakeQuizTab reports no product-state findings; git diff --check passes. Full verify:design-system-state remains red on 24 unrelated current-dev findings outside TakeQuizTab: 12 antd-product-state-import, 11 canonical-state-label, and 1 local-loading-state across KnowledgeQA, Onboarding, ScheduledTasks, Skills, and ACP readiness files. UI TypeScript with NODE_OPTIONS=--max-old-space-size=8192 still fails on inherited Notes/background/voice-cloning diagnostics with no touched-file diagnostics; the default-heap tsc run OOMed before diagnostics. Bandit is not applicable because this slice touches frontend TypeScript/TSX and Backlog markdown only.
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
