---
id: TASK-12038
title: Adopt Models catalog capability recovery state
status: Done
created_date: 2026-06-26 02:45
labels:
- webui
- models
- ux
- accessibility
priority: medium
references:
- TASK-420
- TASK-418
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage9-models-catalog-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/Models/AvailableModelsList.tsx
- apps/packages/ui/src/components/Option/Models/__tests__/AvailableModelsList.test.tsx
updated_date: 2026-06-26 02:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Models catalog route. Replace the full model catalog load failure alert with a shared user-language RecoveryCallout and non-secret diagnostics, while preserving successful model catalog rendering, empty state behavior, provider defaults, and refresh/retry actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Models full-catalog load failures render through the shared RecoveryCallout primitive instead of an AntD Alert.
- [x] #2 Recovery state includes a user-language title/message, retry action, and non-secret diagnostics for request path and raw message/status when available.
- [x] #3 Abort-like metadata request failures remain non-fatal and continue to render the existing empty state rather than an error recovery callout.
- [x] #4 Existing successful catalog rendering and provider/default model behavior remain covered by focused tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 9 plan document for Models catalog capability recovery.
2. Add failing focused tests that expect RecoveryCallout diagnostics for catalog metadata failures and preserve abort-empty/success behavior.
3. Replace the catalog load Alert with RecoveryCallout/buildCapabilityState in AvailableModelsList.
4. Run focused Models tests, touched-file ESLint, whitespace checks, and record Bandit applicability.
5. Update Backlog and commit the Stage 9 slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 9 Models catalog capability recovery. The full catalog metadata error branch now renders a shared RecoveryCallout with retry and diagnostics for GET /api/v1/llm/models/metadata, status, and redacted raw message. Abort-like metadata request failures still return the existing empty state, and successful catalog rendering remains covered by focused tests. Removed explicit any usage from the touched catalog normalization/rendering path while preserving model card output.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Models full-catalog load failures now use the shared RecoveryCallout primitive instead of a local AntD Alert, with safe diagnostics and the existing retry behavior. Focused tests cover successful object-shaped metadata rendering, abort-as-empty behavior, and shared recovery diagnostics. Verification: focused Models Vitest passed; touched-file ESLint passed with only the known repo-level Next pages-directory notice; git diff --check passed. Bandit not applicable because changes are TS/TSX/docs/task metadata only.
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
