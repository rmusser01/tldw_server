---
id: TASK-45.31
title: Adapt Sidepanel StatusDot to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-09 20:08'
updated_date: '2026-05-09 20:17'
labels:
  - design-system
  - ui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Sidepanel/Chat/StatusDot.tsx
  - >-
    apps/packages/ui/src/components/Sidepanel/Chat/__tests__/StatusBadges.design-system.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining Sidepanel Chat StatusDot local-status-badge adapter so connection status variants resolve through getDesignSystemState before selecting the shared Badge variant, while preserving icon-only rendering, tooltip labels, retry behavior, disabled checking state, and demo visual treatment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StatusDot resolves connection statuses through getDesignSystemState before choosing shared Badge severity styling.
- [x] #2 Icon-only Badge rendering preserves tooltip/accessibility text, retry click behavior, disabled checking behavior, and demo visual treatment.
- [x] #3 Focused tests cover connected, checking, demo, config/error, failed/retry, and state-registry mapping.
- [x] #4 The design-system product-state baseline no longer contains the Sidepanel StatusDot local-status-badge exception.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the existing StatusBadges design-system tests to assert StatusDot state-registry calls and Badge variants for connected, checking, demo, config/error, and failed states, plus retry/disabled behavior.
2. Watch the focused test fail before implementation.
3. Update StatusDot to map UX connection states to canonical design-system state keys and derive Badge variants through getBadgeVariantForDesignSystemSeverity, preserving demo as the explicit demo variant.
4. Remove the Sidepanel StatusDot local-status-badge baseline exception.
5. Run focused tests, product-state guard tests, design-system verifier, git diff --check, and a touched-file TypeScript filter; document Bandit skip for UI-only TS/JSON changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused StatusBadges design-system test passed 9/9; product-state guard test passed 49/49; verify:design-system-state passed with 511 baseline exceptions and no StatusDot local-status-badge entry; git diff --check passed. Full UI tsc exited 2 on existing repo-wide test typing debt; /tmp/tldw_ui_tsc_sidepanel_status_dot.txt has 236 lines and no diagnostics matching StatusDot, StatusBadges, or design-system-product-state-baseline. Bandit skipped because touched files are TS/TSX/JSON UI files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Sidepanel StatusDot to resolve connection status through the design-system state registry before deriving Badge severity variants, while preserving icon-only rendering, accessible labels, retry behavior, disabled checking behavior, and the explicit demo variant. Added focused coverage for connected, checking, demo, setup/config error, and retryable failure states, and removed the obsolete StatusDot local-status-badge baseline exception.
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
