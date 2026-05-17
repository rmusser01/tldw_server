---
id: TASK-415
title: Polish main chat cockpit composer control grouping
status: In Progress
dependencies:
- TASK-406
- TASK-414
labels:
- chat
- webui
- cockpit
- ux
- frontend
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx
references:
- https://github.com/rmusser01/tldw_server/pull/1809
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement a narrow main WebUI /chat cockpit UX follow-up for power-user composer controls. Scope is strictly the main /chat Playground composer/cockpit surface: improve scan grouping and hover/focus names for dense advanced composer controls without changing chat behavior, adding bottom UI, touching extension sidepanel/sidebar routes, or broadening into unrelated pages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main /chat composer power-user controls are visually grouped for scanning without removing existing controls or changing send behavior.
- [ ] #2 Icon-only composer/cockpit controls expose clear hover/focus names using existing UI patterns or accessible labels.
- [ ] #3 Keyboard/focus behavior remains intact for grouped controls and existing popovers/dialogs.
- [ ] #4 Focused Vitest coverage proves grouping labels and no behavior regression for the composer/cockpit surface.
- [ ] #5 Verification records focused tests, git diff check, and Bandit applicability; real-server/browser proof is added only if it materially helps this UI slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented narrow main /chat ComposerToolbar grouping labels for casual, pro, and mobile layouts. Added aria-labeled role=group regions, aria-controls wiring for advanced controls, and compact pro cockpit panel headings without moving controls or adding any bottom bar.
- Verification: `bunx vitest run src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx src/components/Option/Playground/__tests__/ComposerToolbar.layout.guard.test.ts src/components/Option/Playground/__tests__/PlaygroundForm.composer-options.guard.test.ts --config vitest.config.ts` passed: 3 files, 29 tests. `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx --config vitest.config.ts` passed: 2 files, 24 tests; existing mocked-test stderr still reports `tldw server not configured`. `git diff --check` passed.
- Real-server visual proof: started local WebUI from this worktree at `http://127.0.0.1:3100` and used the real API server at `http://127.0.0.1:8000` with the API key loaded from the project `.env` into the browser context. Screenshot: `/private/tmp/chat-cockpit-composer-grouping-real-auth.png`. Browser evidence showed `/chat` loaded, group labels existed, and `data-testid="composer-bottom-bar"` was absent.
- Known verification caveat: `bun run verify:design-system-state` exits 1 on unrelated existing product-state baseline/stale-entry findings under Watchlists and other non-/chat areas. No touched files are listed in that failure. Bandit is not applicable because this slice only changes TypeScript/React frontend files. Pending external gate: rendered screenshot still needs user visual approval before treating the visible UI work as fully done.
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
