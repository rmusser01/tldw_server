---
id: TASK-418.10
title: Plan WebUI route governance QA implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- qa
- governance
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 12. Scope maps all findings, especially F2, F15, F17, and F18, into route inventory, metadata, heading, command target, capability fixture, responsive, screenshot, and final browser QA governance without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Created the documentation-only Task 12 implementation plan at `Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md`.
- [x] Covered all findings, with explicit emphasis on `F2`, `F15`, `F17`, and `F18`.
- [x] Mapped route inventory, metadata coverage, page headings, command targets, sidepanel availability, hosted visibility, capability fixtures, responsive checks, Axe checks, and browser evidence protocol into concrete implementation tasks.
- [x] Identified CI-suitable checks separately from manual browser QA evidence.
- [x] Included source documents, current QA surface, governance contracts, file structure, implementation tasks, acceptance criteria, and verification commands.
- [x] Kept this task limited to Markdown planning artifacts with no product frontend or backend code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created the Task 12 child implementation plan as the final route-governance and QA slice for `TASK-418`.
- Cross-checked current QA ownership before writing the plan:
  - Smoke inventory lives in `apps/tldw-frontend/e2e/smoke/page-inventory.ts`.
  - All-pages smoke lives in `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`.
  - High-risk Axe checks live in `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`.
  - Stage 2 route contracts live in `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`.
  - Smoke diagnostics and allowlist machinery live in `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`.
  - Command palette tests live under `apps/packages/ui/src/components/Common/__tests__/`.
  - Route registry and sidepanel tests live under `apps/packages/ui/src/routes/__tests__/`.
- Added explicit implementation guidance for CI-suitable gates versus manual browser evidence, route metadata fields, smoke inventory rules, heading exceptions, command target rules, capability fixtures, responsive checks, and final Backlog closure.
- Bandit was not run because this task touched only Markdown planning and Backlog task files.
- Verification performed for the plan artifact:
  - `rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.\\.\\.|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md`
  - `rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md`
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md`
  - `node -e` required-finding, file, script, and governance-topic coverage check
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Task 12 route governance QA implementation plan. The plan preserves the existing Playwright and Vitest test surfaces, avoids backend changes, and turns the final audit regression risks into enforceable metadata, inventory, heading, command, sidepanel, capability, responsive, accessibility, and browser-evidence gates.
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
