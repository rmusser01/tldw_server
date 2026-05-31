---
id: TASK-418.9.2
title: Implement WP11B evaluations readiness and recovery states
status: Done
labels:
- ux
- webui
- extension
- wp11b
- evaluations
- testing
priority: High
parent_task_id: TASK-418.9
documentation:
- Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/components/Option/Evaluations/components/EvaluationRecoveryCallout.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationRecoveryCallout.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/DatasetsTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/EvaluationsTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/HistoryTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/SyntheticReviewTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/WebhooksTab.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/DatasetsTab.pagination.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/EvaluationsTab.empty-state.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/HistoryTab.filters.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RunsTab.benchmark-option.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/SyntheticReviewTab.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/WebhooksTab.contract.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11B Task 2 from the WebUI study/safety/specialized implementation plan. Make /evaluations ready-or-recoverable for first-time and power users by adding focused readiness/recovery tests and scoped UI adjustments for workspace setup, beta identity, tab discoverability, empty states, worker/endpoint unavailability, and filter/form preservation. Route-boundary work from the original task is already covered by TASK-418.9.1 and PR #1938, so this slice starts from the post-merge route contract state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Endpoint and worker failures on `/evaluations` use design-system recovery states with clear failing request paths.
- [x] Evaluation empty state gives first-time users a direct Recipes path while preserving custom-evaluation workflow.
- [x] Runs, datasets, webhooks, history, recipes, and synthetic review preserve adjacent controls/state when endpoint calls fail.
- [x] Evaluations tabs avoid new AntD product-state alerts and keep the design-system guard clean for touched files.
- [x] Focused unit, route-boundary, design-system, and browser smoke verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a shared `EvaluationRecoveryCallout` wrapper around `RecoveryCallout` for evaluations endpoint diagnostics. Converted list/detail load failures in Recipes, Evaluations, Runs, Datasets, Webhooks, History, and Synthetic Review to the shared recovery callout. Added an Open Recipes action to the evaluations empty state. Replaced remaining product-state AntD Alerts in touched evaluations tabs with `StatePanel` or plain content and removed stale evaluations entries from the product-state guard baseline. Fixed a `SyntheticReviewTab` render loop by using a stable empty queue array and returning current selection state for no-op cleanup updates. Tightened the embeddings recipe config's local dataset/media-search typing so the evaluations subtree no longer contributes TypeScript errors.

PR #1943 review follow-up: internationalized the recovery callout default message and diagnostics labels, added FastAPI `detail`/`msg` extraction including arrays, and prevented duplicate diagnostics such as `HTTP 404: HTTP 404`.

Verification recorded: focused evaluations Vitest suite 42 tests passed; route-boundary Vitest 8 tests passed; design-system product-state guard passed; Playwright evaluations recipes/synthetic smoke tests passed; targeted Chromium QA observed Unavailable diagnostics for `/api/v1/evaluations/recipes`, `/api/v1/evaluations`, `/api/v1/evaluations/datasets`, and `/api/v1/evaluations/webhooks` with Register webhook still enabled. TypeScript check was attempted with increased heap; repo-wide baseline failures remain outside this slice, and the filtered evaluations output is empty. Bandit was not run because this slice touched TypeScript/JSON/Backlog only, no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP11B evaluations readiness/recovery states for endpoint and worker unavailability, with preservation-oriented regression coverage across the affected tabs. Browser QA confirmed rendered recovery diagnostics for recipes, evaluations, datasets, and webhooks, and existing guided/synthetic evaluations smoke flows still pass. PR review comments on recovery-callout i18n, FastAPI detail parsing, and duplicate HTTP detail formatting are addressed with component-level regression tests.
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
