---
id: TASK-45.48
title: Migrate settings health status product-state alerts to design-system primitives
status: Done
labels:
- design-system
- webui
- product-state
- settings
priority: medium
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Settings/health-status.tsx
- apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by replacing the remaining AntD product-state usage in the /settings/health status surface with shared design-system primitives while preserving the existing diagnostics copy and recovery actions. Current dev also had one new Quick Ingest offline-review AntD Alert and stale llama.cpp admin baseline rows blocking the product-state verifier; this slice migrates that single Quick Ingest warning and removes stale baseline entries rather than adding new debt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Settings health status no longer imports AntD Alert or Spin for product-state UI.
- [x] Health status connected, no-server, queue/interval warning, and per-check loading states render through design-system primitives.
- [x] The current-dev Quick Ingest offline review warning renders through the design-system Alert primitive instead of adding a new baseline exception.
- [x] Product-state baseline removes migrated health entries and stale llama.cpp admin entries without introducing new unbaselined findings.
- [x] Focused Vitest coverage and the design-system product-state verifier pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Inventory health-status AntD Alert/Spin usage and map each case to shared Alert/LoadingState primitives.
- [x] Add failing tests proving health status banners/loading render with design-system primitive markers.
- [x] Migrate health-status.tsx to DesignSystemAlert and LoadingState while preserving existing copy/actions.
- [x] Handle current-dev verifier drift by migrating the single Quick Ingest offline Alert and removing stale llama.cpp admin baseline entries.
- [x] Run focused Vitest, verify:design-system-state, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated /settings/health info/success/warning banners from AntD Alert to components/ui/primitives Alert.
- Migrated per-check running indicator from AntD Spin to LoadingState with a stable data-testid for regression coverage.
- Migrated Quick Ingest review-step offline warning to the design-system Alert primitive because it was a new unbaselined current-dev verifier blocker.
- Removed 5 health-status baseline entries and 14 stale llama.cpp admin baseline entries.
- PR review follow-up: added a true inline LoadingState return path, removed the HealthStatus !important padding override, and made the connected-status test resilient to i18n interpolation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the next product-state design-system slice for /settings/health and resolved current-dev verifier drift without adding new baseline debt. PR review follow-up hardened inline LoadingState behavior and removed the callsite !important override. Verification: health-status + QuickIngest + LoadingState focused Vitest passed (42 tests); bun run verify:design-system-state passed; git diff --check passed. Bandit is not applicable because the touched implementation is TypeScript/JSON/backlog metadata only.
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
