---
id: TASK-325
title: Surface Persona Visual import-preview renderer diagnostics in Persona Garden
status: Done
assignee: []
created_date: '2026-05-14 01:20'
updated_date: '2026-05-14 01:41'
labels:
  - persona
  - buddy
  - webui
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1645'
  - 'https://github.com/rmusser01/tldw_server/pull/1642'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow Buddy/Persona visual-pack slice under GitHub issue #1645. PR #1642 added backend diagnostics for Manifest V2/non-sprite renderer import previews while keeping V1 sprite_frames as the only activatable renderer. Persona Garden should make those diagnostics understandable in the import-preview review panel and avoid offering commit when backend eligibility says the preview is blocked. Scope stays frontend review UX only: no Live2D runtime renderer, no V2 activation, no MCP provider expansion, and no VN/CYOA behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Renderer import-preview diagnostics returned in proposed_plan are typed or normalized before UI use.
- [x] #2 Blocked Manifest V2/non-sprite renderer previews show renderer status, blockers, warnings, and activation eligibility in Persona Garden import preview.
- [x] #3 Import commit controls are disabled or withheld when the preview is not backend commit-eligible or renderer diagnostics mark it uncommittable.
- [x] #4 Existing V1 sprite_frames import preview and commit flow remains unchanged.
- [x] #5 Focused frontend tests cover blocked renderer diagnostics display and commit gating.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation:
- Typed and normalized proposed_plan.renderer_import_preview in the Persona Visual frontend contract.
- Rendered renderer import diagnostics in Persona Garden import previews, including status, blockers, warnings, role categories, and activation eligibility.
- Gated import commit on backend commit_eligible and renderer can_commit signals while preserving existing sprite_frames import behavior.

Verification:
- PASS: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testNamePattern "surfaces blocked renderer import diagnostics"
- PASS: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
- PASS: git diff --check
- BASELINE FAIL: bun run verify:design-system-state reports unrelated existing blocked product-state findings in AgentTasks and other files outside this task.
- BASELINE FAIL: bunx tsc --noEmit --project tsconfig.json reports unrelated existing type errors outside the touched Persona Visual files.
- SKIP: Bandit not applicable; this slice touches TypeScript UI and Backlog metadata only.

Review fix pass:
- Address Gemini localization threads for new renderer diagnostics copy.
- Address Qodo double-submit reliability thread for import commit enqueue.

Review fixes implemented:
- Added a synchronous import-commit in-flight ref guard and readability short-circuit to prevent duplicate commit enqueue before UI disabled state propagates.
- Localized new renderer diagnostics labels, fallback unknown text, activation status text, blocker labels, warning labels, asset-role labels, and commit-blocked copy.
- Extended VisualPackEditor coverage for localized diagnostics keys and duplicate click commit guarding.

Review fix verification:
- PASS: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testNamePattern "import-preview|blocked renderer"
- PASS: bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
- PASS: git diff --check
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Surface Persona Garden import-preview renderer diagnostics, gate blocked commits, localize new diagnostics copy, and guard import commit enqueue against duplicate rapid submits.
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
