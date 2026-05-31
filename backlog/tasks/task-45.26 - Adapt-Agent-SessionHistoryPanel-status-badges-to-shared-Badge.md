---
id: TASK-45.26
title: Adapt Agent SessionHistoryPanel status badges to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 15:37'
updated_date: '2026-05-09 15:42'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/Agent/SessionHistoryPanel.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate Agent SessionHistoryPanel local StatusBadge from bespoke span styling to the shared design-system Badge primitive with explicit canonical state mapping, preserving localized labels and removing its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SessionHistoryPanel agent statuses render through the shared Badge primitive
- [x] #2 Agent status badge variants are selected from design-system state registry mappings while preserving existing visible labels and icons
- [x] #3 The local-status-badge baseline exception for src/components/Agent/SessionHistoryPanel.tsx is removed without new unbaselined findings
- [x] #4 Focused SessionHistoryPanel tests, product-state guard tests, design-system verifier, diff checks, and touched-file TypeScript filter are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red evidence: bunx vitest run src/components/Agent/__tests__/SessionHistoryPanel.status-badge.test.tsx --reporter=dot failed because Idle was not rendered inside data-ds-component="Badge".

Implementation: SessionHistoryPanel StatusBadge now maps AgentStatus values to canonical design-system state keys, selects shared Badge variants through getBadgeVariantForDesignSystemSeverity, preserves the translated visible labels and lucide status icons, marks icons aria-hidden, and removes the SessionHistoryPanel local-status-badge baseline exception. Running maps to retrying so it keeps an info/primary visual treatment while still using the canonical state registry.

Verification: bunx vitest run src/components/Agent/__tests__/SessionHistoryPanel.status-badge.test.tsx --reporter=dot passed 1/1; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46; bun run verify:design-system-state passed with baseline exceptions 510 and local-status-badge 4; git diff --check passed; touched-file TypeScript filter over bunx tsc --noEmit --pretty false returned no matches.

Bandit: skipped because this slice only changes TypeScript/TSX, JSON, and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Agent SessionHistoryPanel StatusBadge to the shared Badge primitive with design-system state mapping and removed its local-status-badge baseline exception. Focused tests, product-state guard tests, design-system verifier, diff checks, and touched-file TypeScript filter passed; full frontend tsc remains covered by the existing repo-wide baseline caveat.
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
