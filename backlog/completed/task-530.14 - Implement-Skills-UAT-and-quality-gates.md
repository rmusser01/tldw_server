---
id: TASK-530.14
title: Implement Skills UAT and quality gates
status: Done
labels:
- skills
- uat
- frontend
priority: High
parent_task_id: TASK-530
documentation:
- Docs/superpowers/specs/2026-06-30-skills-uat-quality-gates-design.md
- Docs/superpowers/plans/2026-07-04-skills-uat-quality-gates.md
- Docs/Reviews/skills-page-uat.md
- Docs/Design/2026-07-14-skills-ux-gap-closure-design.md
- Docs/Design/2026-07-15-skills-extension-parity-design.md
references:
- https://github.com/rmusser01/tldw_server/pull/2629
- https://github.com/rmusser01/tldw_server/pull/2554
- https://github.com/rmusser01/tldw_server/pull/2732
- https://github.com/rmusser01/tldw_server/pull/2740
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the full /skills UAT quality-gates bundle: deterministic workflow-level Playwright coverage, manual QA checklist, and success metrics documentation without duplicating existing component-level Skills tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec is written and reviewed before implementation planning.
- [x] #2 Automated UAT scope is bounded to beginner, power-user, and trust-risk failure workflows.
- [x] #3 Manual QA checklist and success metrics are included in the planned deliverables.
- [x] #4 The task record links the spec and implementation plan artifacts as they are produced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-04-skills-uat-quality-gates.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-07-04: Added the implementation plan at `Docs/superpowers/plans/2026-07-04-skills-uat-quality-gates.md`.
- 2026-07-04: Added manual `/skills` UAT scenarios and non-telemetry success metrics at `Docs/Reviews/skills-page-uat.md`.
- 2026-07-04: Added deterministic Playwright UAT coverage for beginner seed/copy/test-run workflows, power-user large-library search/filter/sort/bulk-delete confirmation, and representative import/execution/delete/loading failure states.
- 2026-07-04: Unsupported Skills capability remains covered by `apps/packages/ui/src/components/Option/Skills/__tests__/SkillsWorkspace.test.tsx`; E2E coverage was kept to deterministic page states because the browser capability layer can fall back to the bundled OpenAPI spec.
- 2026-07-12: Rebased PR #2629 onto current `dev` and added regression coverage for missing execute payloads plus body-less DELETE requests with trailing slashes.
Closeout reconciliation: PR #2629 merged the deterministic WebUI Skills UAT quality gates and manual checklist on 2026-07-13. PR #2554 supplied the associated accessibility-state polish; PR #2732 closed the remaining confirmed UX/reliability gaps; PR #2740 certified six strict packaged-extension workflows. All specified design, plan, checklist, metrics, and automated-scope artifacts are present on dev. Merged verification records report 13 deterministic mocked WebUI workflows and 6 strict packaged-extension workflows with zero skips in the extension gate. Three optional live-backend WebUI scenarios remain environment-gated and are explicitly carried forward as the next integration-certification opportunity. This closeout changes Backlog Markdown only, so Bandit is not applicable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Skills UAT quality-gates bundle. The repository now contains the reviewed design and implementation plan, deterministic beginner/power-user/trust-risk WebUI workflows, scenario-level fixtures, a manual accessibility/responsive/failure checklist, documented success metrics, comprehensive UX/reliability remediation, and strict packaged-extension parity coverage. The remaining evidence gap is deliberately narrow: three live-backend WebUI scenarios and an actual MV3 background-relay live smoke are not part of the deterministic release gate and should be handled by a separate verify-first integration task.
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
