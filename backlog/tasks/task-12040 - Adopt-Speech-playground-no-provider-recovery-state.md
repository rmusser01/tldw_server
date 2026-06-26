---
id: TASK-12040
title: Adopt Speech playground no-provider recovery state
status: Done
created_date: 2026-06-26 03:08
references:
- TASK-420
- TASK-418.8
- TASK-12039
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage11-speech-no-provider-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
updated_date: 2026-06-26 03:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Speech playground server TTS no-provider state. Replace the route-local alert-only server-audio unavailable banner with the shared setup-required state while preserving locked TTS route copy, provider strip, editable draft text, and disabled generation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speech locked TTS mode shows one shared setup-required recovery state when the selected tldw server TTS path has no audio/provider capability.
- [x] #2 The old local alert title is no longer the primary no-provider state on the Speech page.
- [x] #3 Existing provider strip, editable text draft, and play disabled reason remain intact for the no-provider state.
- [x] #4 Focused Speech component tests cover the no-provider recovery state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused SpeechPlaygroundPage regression test for the no-provider recovery state and run it red.
2. Replace the no-provider local alert in SpeechPlaygroundPage with the shared StatePanel setup-required state.
3. Re-run the focused Speech test, touched-file lint, and diff whitespace checks.
4. Record verification and final summary on this task before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented with a test-first pass:
- Initial focused test run failed before reaching the new assertion because the existing no-provider alert rendered a router Link without a router wrapper in the component test harness.
- Added a local `react-router-dom` Link mock for this component test, then reran the focused test and confirmed the intended red failure: missing `speech-tts-no-provider-recovery` shared state while the page still rendered the old alert.
- Replaced the Speech server TTS no-provider alert with the shared `StatePanel` setup-required state while preserving the settings link, Browser fallback copy, provider strip, editable draft behavior, and disabled Play button state.

Verification:
- `bun run test:run ../packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx` from `apps/tldw-frontend`: PASS, 26 tests.
- `bun apps/node_modules/.bun/eslint@9.39.2+288993669ddeca06/node_modules/eslint/bin/eslint.js -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`: PASS with no errors; warnings are pre-existing in the large Speech files plus the known Next pages-directory notice.
- `git diff --check`: PASS.
- `bun run verify:design-system-state` and `bun scripts/verify-design-system-product-state.mjs` from `apps/packages/ui`: unable to start because the local shared-UI install lacks a `typescript` package symlink even though the Bun store has `typescript@5.9.3`; recorded as an environment/dependency-layout skip.
- Bandit: not applicable; this slice touches TS/TSX/docs/Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Speech locked TTS mode now uses the shared setup-required StatePanel when the tldw server TTS provider path has no audio/provider capability. This removes the route-local alert as the primary no-provider state while keeping the settings link, Browser fallback language, provider strip, editable draft text, and disabled Play behavior intact. A focused regression test covers the shared state and the existing no-provider interaction contract.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Speech tests pass
- [x] #8 Touched-file lint check run or documented
- [x] #9 git diff whitespace check run
- [x] #10 Bandit run for touched code when applicable or documented as not applicable
<!-- DOD:END -->
