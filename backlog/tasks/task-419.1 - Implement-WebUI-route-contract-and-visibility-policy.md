---
id: TASK-419.1
title: Implement WebUI route contract and visibility policy
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-17 21:27'
labels:
  - ux
  - webui
  - extension
  - implementation
  - route-contract
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md
parent_task_id: TASK-419
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first WP1 slice from the approved WebUI/extension UX remediation plan. Scope: add the canonical route metadata contract and initial coverage tests for audited root routes, route aliases, visibility classes, option registry validation, sidepanel availability, command target trust, and smoke inventory ownership. Keep changes scoped to WebUI/extension route contract files and tests; do not do route-family UX remediation in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Route metadata contract types and helpers exist for audited root routes.
- [x] #2 Initial tests prove all audited root routes have metadata and canonical paths where applicable.
- [x] #3 Option route registry validation is wired to metadata without changing route behavior.
- [x] #4 Extension sidepanel and options availability are represented in metadata or tested as an explicit follow-up gap.
- [x] #5 Command palette route target mismatch is covered by tests before any behavior change.
- [ ] #6 Smoke inventory ownership is derived from or checked against route metadata.
- [ ] #7 Implementation remains scoped to WP1; no route-family visual remediation is included.
- [ ] #8 Focused frontend tests and diff checks are recorded in the task before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Baseline before route metadata edits: `bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line` from `apps/tldw-frontend` initially failed in the sandbox because Next could not bind `0.0.0.0:8080` (`listen EPERM`). Rerun with approved escalation passed: 1 test passed in 33.6s.

Task 1 red/green: added `src/routes/__tests__/route-metadata.coverage.test.ts`; first focused Vitest run failed as expected because `../route-metadata` did not exist. Added pure route metadata/types/helpers in `src/routes/route-metadata.ts`; focused run then passed: 5 tests passed.

Additional verification: `bunx tsc --noEmit --pretty false` from `apps/packages/ui` currently fails on existing unrelated TypeScript errors across audio, composer, flashcards, playground, route registry, and service tests. No reported error referenced `src/routes/route-metadata.ts` or `src/routes/__tests__/route-metadata.coverage.test.ts`.

Task 2 red/green: added `src/routes/__tests__/route-registry.visibility.test.ts`. First attempt imported `route-registry.tsx` and failed for the wrong reason by resolving optional OCR dependency `pa-tesseract.js`; corrected the test to inspect registry source text plus pure route-path constants. The corrected red run reported missing metadata for 41 non-dynamic option registry paths. Added registry metadata for settings, admin, nested source, companion, presentation, moderation, prototype workspace, research-studio, and workspace-studio routes. Focused run passed: 4 registry visibility tests passed. Combined metadata plus registry run passed: 9 tests passed.

Task 3 red/green: added `src/routes/__tests__/route-registry.sidepanel-availability.test.ts` over shared and extension sidepanel registry source files. Initial red run reported missing sidepanel availability for `/agent`, `/clipper`, and `/error-boundary-test`; the nav parser was tightened to avoid dynamic/sidepanel false positives. Added sidepanel-only and debug metadata for `/agent`, `/clipper`, `/error-boundary-test`, `/__debug__/sidepanel-chat`, and `/__debug__/sidepanel-error-boundary`. Focused sidepanel run passed: 4 tests passed. Combined metadata, registry, and sidepanel run passed: 13 tests passed. Standalone route metadata type check passed.

Task 4 red/green: added a command palette regression test asserting the `Go to Chat` row exposes `data-command-id="nav-chat"`, `data-target-path="/chat"`, and navigates to `/chat`. Red run failed because command rows had no command id/target path attributes and `nav-chat` targeted `/`. Updated `CommandPalette.tsx` to use `CHAT_PATH`, set `nav-chat.targetPath` to `/chat`, navigate to `/chat`, and expose nonvisual data attributes for command id/target path. Focused command palette run passed: 5 tests passed. Combined route plus command run passed: 18 tests passed.
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
