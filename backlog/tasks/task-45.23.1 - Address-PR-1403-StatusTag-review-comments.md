---
id: TASK-45.23.1
title: Address PR 1403 StatusTag review comments
status: Done
assignee: []
created_date: '2026-05-09 05:11'
updated_date: '2026-05-09 05:15'
labels:
  - design-system
  - webui
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1403'
  - apps/packages/ui/src/components/Option/Watchlists/shared/StatusTag.tsx
  - >-
    apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/StatusTag.accessibility.test.tsx
  - apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
parent_task_id: TASK-45.23
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable PR #1403 review comments on the Watchlists StatusTag shared Badge migration. Scope: keep PersonaGarden loading labels locale-reactive while still using the design-system state fallback, map Watchlists running status to the most appropriate canonical state, and remove brittle Badge internal class assertions from StatusTag tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PersonaGarden loading and refresh labels are evaluated inside VisualPackEditor render through translation APIs and design-system fallback labels
- [x] #2 Watchlists running status maps to a semantically appropriate canonical state without reintroducing product-state guard findings
- [x] #3 StatusTag small-size coverage verifies the adapter contract without asserting Badge internal Tailwind class strings
- [x] #4 Focused tests, product-state guard tests, design-system verifier, diff checks, and touched-file TypeScript filter are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review surface inspected on PR #1403. Actionable unresolved threads: PersonaGarden module-level loading fallback should move into render/t(), candidate Refresh should be localized, StatusTag running should avoid retrying semantics, and StatusTag small-size test should avoid Badge internal Tailwind class assertions. CI was pending at inspection time.

Red evidence: StatusTag focused test failed on missing data-ds-variant/data-ds-size attributes and VisualPackEditor focused test failed because candidate loading used the static Loading fallback instead of the localized loading label.

Implementation: Badge now exposes stable data-ds-size and data-ds-variant attributes for adapter contract tests; StatusTag maps running to loading instead of retrying; VisualPackEditor computes loading/refresh labels inside render via t(...) with getDesignSystemState("loading").label as the loading fallback.

Verification: bunx vitest run src/components/Option/Watchlists/shared/__tests__/StatusTag.accessibility.test.tsx --reporter=dot passed 3/3; bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --reporter=dot passed 7/7; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46; bun run verify:design-system-state passed with baseline exceptions 512 and local-status-badge 6; git diff --check passed.

TypeScript caveat: bunx tsc --noEmit --pretty false still fails on existing repo-wide frontend baseline errors, but filtering the output for StatusTag, VisualPackEditor, Badge, and their focused tests returned no touched-file errors.

Bandit: skipped because this review pass only changes TypeScript/TSX and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1403 review comments by making PersonaGarden loading/refresh labels render-time localized, mapping Watchlists running status to the loading state, and replacing brittle Tailwind class assertions with stable Badge adapter contract attributes. Focused tests, guard tests, design-system verifier, and diff checks passed; full tsc remains blocked by unrelated baseline errors.
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
