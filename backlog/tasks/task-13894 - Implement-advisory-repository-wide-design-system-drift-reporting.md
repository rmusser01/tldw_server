---
id: TASK-13894
title: Implement advisory repository-wide design-system drift reporting
status: In Progress
labels:
- design-system
- webui
- extension
- governance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed five-stage plan for repo-wide advisory token/color/radius/layout reporting; related design task TASK-45.44.16 and issue #1673.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Four UI roots are scanned with syntax-aware rules and scoped definitions
- [ ] #2 Reports preserve findings/exceptions/provenance and distinguish incomplete scans
- [ ] #3 CI is advisory and respects existing license-first contracts
- [ ] #4 Focused tests, report sample review, lint and touched-scope Bandit are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-design-system-drift-reporting.md. Spec: Docs/superpowers/specs/2026-10-20-design-system-drift-reporting-design.md. ADR check: proposed ADR-079 governs the lasting advisory UI convention; no new ADR. Existing source/dependency state will be verified before implementation.
Implementation progress: locked JS dependencies installed without lockfile changes. Token authority39 tests and rule engine37 tests pass. Report15 tests pass. CLI19 tests pass, including source/output protection, hard-linked output replacement, unavailable sparse source and hidden Git-index edits; actual RED verified for both review bugs and both hidden-index cases before fixes. Existing product-state guard54 tests remains passing; CI contract113 tests pass. Scoped new-test TypeScript compilation passes; pinned frontend ESLint returns0 with existing Next pages-directory advisory. Fullrepo first scan completed on3,644 source files before vendor/system-color refinements; final recount pending. Review findings are being addressed before publication. Bandit touched Python test reports233 existing low findings, no scan errors; baseline comparison pending. Ruling: default jsdom Vitest environment replaces plan node annotation because shared setup requires window. Ruling: exact upstream react-pdf.css copy excluded as vendor based on file license/header; malformed-vendor regression verified RED/GREEN. Atomic report replacement prevents hard-link truncation.
Final review: CLI/report scoped re-review found no remaining findings; independent probes verify atomic replacement preserves hard-linked originals, report permissions0600, temp cleanup, and genuine deletions stay complete/dirty. A separate final token/rule/CI review reproduced3 silent rule gaps: non-definition custom-property color overrides, quoted/spread CVA options, and literal RGB channels with variable alpha. Rules worker is adding regression tests and minimal fixes. Parent current pre-fix matrix169 focused tests and113 CI tests passes; final counts and fullscan will be rerun after these fixes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
