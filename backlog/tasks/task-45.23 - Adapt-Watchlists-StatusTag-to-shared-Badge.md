---
id: TASK-45.23
title: Adapt Watchlists StatusTag to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 04:33'
updated_date: '2026-05-09 04:41'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Watchlists/shared/StatusTag.tsx
  - >-
    apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/StatusTag.accessibility.test.tsx
  - apps/packages/ui/src/components/ui/primitives/Badge.tsx
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Watchlists run-status tag adapter from AntD Tag to the shared design-system Badge primitive with explicit canonical state mapping, while preserving accessibility labels, icons, fallback labels, and removing its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists StatusTag renders through the shared Badge primitive while preserving known and unknown status labels, icons, title, and aria-label behavior.
- [x] #2 Run statuses map through the design-system state registry before selecting Badge variants.
- [x] #3 The local-status-badge baseline exception for src/components/Option/Watchlists/shared/StatusTag.tsx is removed without introducing new unbaselined findings.
- [x] #4 Focused StatusTag tests, product-state guard tests, the design-system verifier, and diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red evidence: after removing the AntD Tag mock, the focused StatusTag accessibility test failed because StatusTag did not render a shared Badge marker yet.

Implementation: StatusTag now maps Watchlists run statuses through getDesignSystemState before choosing shared Badge variants, preserves known/unknown labels and icons, forwards aria-label/title through Badge, and removes the StatusTag local-status-badge baseline exception.

Verification: bunx vitest run src/components/Option/Watchlists/shared/__tests__/StatusTag.accessibility.test.tsx --reporter=dot passed 3/3; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46; bun run verify:design-system-state passed with baseline exceptions 512 and local-status-badge 6; git diff --check passed.

TypeScript caveat: bunx tsc --noEmit --pretty false still fails on the existing repo-wide frontend baseline, but a filtered tsc pass for StatusTag, Badge, VisualPackEditor, and design-system-product-state-baseline produced no touched-file errors.

Bandit: skipped because this slice only changes TypeScript, JSON, and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Watchlists StatusTag from AntD Tag to the shared Badge primitive with design-system state mapping, preserving the accessibility/title behavior and removing its local-status-badge baseline exception. Verification passed for the focused StatusTag test, product-state guard suite, design-system verifier, and diff checks; full tsc remains blocked by unrelated baseline errors outside this slice.
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
