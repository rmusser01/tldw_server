---
id: TASK-418.10.3
title: Implement WP12 sidepanel availability and hosted visibility governance
status: Done
labels:
- wp12
- webui
- extension
- route-governance
priority: High
ordinal: 3
parent_task_id: TASK-418.10
references:
- TASK-418.10
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/1960
modified_files:
- apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts
- apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.persona.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.companion.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry-ast-helpers.ts
- apps/packages/ui/src/routes/option-route-visibility.ts
- apps/packages/ui/src/routes/route-hosted-visibility.ts
- apps/packages/ui/src/routes/route-path-normalization.ts
- apps/packages/ui/src/routes/route-metadata.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 3 from the WebUI route governance QA plan: add sidepanel availability and hosted visibility governance against route metadata without page-level redesign or backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sidepanel governance tests verify metadata availability matches shared and extension sidepanel registries.
- [ ] #2 Hosted visibility tests verify hosted-visible and hosted-hidden routes are metadata-owned, non-internal, and explicitly reasoned.
- [ ] #3 Existing chat, clipper, flashcards, persona, and companion sidepanel handoff coverage remains intact.
- [ ] #4 Focused sidepanel and hosted visibility Vitest checks pass, with unrelated baseline failures documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP12 Task 3 sidepanel and hosted visibility governance. Added metadata-backed hosted option visibility, sidepanel registry union governance across shared and extension registries, and metadata assertions for chat, clipper, flashcards, persona, and companion sidepanel handoff tests. PR #1963 review follow-up moved the production hosted allowlist into lightweight route-hosted-visibility and route-path-normalization modules so option-route-visibility no longer imports the full route metadata registry, included dynamic option routes in hosted-hidden governance, added safer optional metadata checks, and made extension source lookups conditional when extension sources are absent. Focused route governance Vitest suite passed after review fixes with 42 tests. git diff --check passed. Broad bunx tsc --noEmit remains blocked by package-wide baseline TypeScript errors outside this slice; the default heap also OOMs before diagnostics. Bandit is not applicable because this slice changed TypeScript and Backlog Markdown only.
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
