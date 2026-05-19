---
id: TASK-45.44.8.1
title: Migrate PromptStudioPlaygroundPage alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.8
references:
- https://github.com/rmusser01/tldw_server/issues/1665
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/src/components/Option/PromptStudio/PromptStudioPlaygroundPage.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/PromptStudio/PromptStudioPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/PromptStudio/__tests__
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate PromptStudioPlaygroundPage product-state alerts from AntD Alert to the shared design-system Alert primitive, then remove the matching product-state guard baseline entries for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PromptStudioPlaygroundPage no longer imports or renders AntD Alert for product-state messaging.
- [x] #2 A focused Prompt Studio test asserts migrated alert copy renders through the design-system Alert primitive.
- [x] #3 The product-state guard baseline no longer contains entries for src/components/Option/PromptStudio/PromptStudioPlaygroundPage.tsx.
- [x] #4 Focused UI tests and design-system guard verification pass, with non-code security skip documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated PromptStudioPlaygroundPage product-state Alert usages to the shared design-system Alert primitive while preserving existing AntD layout controls.
- Added PromptStudioPlaygroundPage.connection.test.tsx assertions that connection guidance titles are rendered inside `data-ds-component="Alert"`.
- Removed nine PromptStudioPlaygroundPage entries from `design-system-product-state-baseline.json` and normalized the inherited stale SourceFormModal baseline entry into the two current duplicate-stable entries required by the guard.
- Addressed PR #1849 Gemini review comments by routing the migrated connection-state Alert titles, action labels, and descriptions through `t()`, with focused assertions for the reviewed labels/descriptions.
- Bandit was not run because this slice touches only TypeScript/TSX and JSON baseline files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PromptStudioPlaygroundPage now uses the canonical design-system Alert primitive for product-state alert messaging. The product-state baseline has zero entries for that page; the repo baseline is 360 after removing nine PromptStudio entries and normalizing one unrelated stale SourceFormModal baseline entry into two current entries. PR #1849 review follow-up internationalized the migrated connection-state Alert titles, labels, and descriptions via `t()`. Verification: focused Prompt Studio Vitest passed after a red i18n assertion, product-state guard Vitest passed, verify:design-system-state passed, git diff --check passed. Full TypeScript still fails on inherited repo-wide debt; /tmp/promptstudio-alerts-review-tsc.log has 240 lines and no diagnostics for PromptStudioPlaygroundPage, its test, SourceFormModal, or the baseline file.
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
