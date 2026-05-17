---
id: TASK-407
title: Implement main chat role-play preset remediation
status: Done
labels:
- chat
- ux
- roleplay
- implementation
documentation:
- Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-main-chat-role-play-preset-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved staged remediation plan for the main /chat role-play preset workflow. Scope stays limited to crash/recovery/accessibility, visible state, mobile parity, setup consolidation, saved role-play presets, and compatibility guardrails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Crash/recovery/accessibility fixes implemented and verified.
- [x] #2 Visible state and terminology cleanup implemented and verified.
- [x] #3 Mobile role-play preset parity implemented and verified.
- [x] #4 Role-play setup consolidation implemented and verified.
- [x] #5 Saved role-play setup UX implemented and verified.
- [x] #6 Compatibility/request-inclusion guardrails implemented and verified.
- [x] #7 Final focused tests, lint, compile, browser verification, and security applicability recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final closeout completed for the staged /chat role-play preset remediation plan. Child tasks TASK-407.1 through TASK-407.6 are all Done and cover crash/recovery/accessibility, visible state and terminology, mobile parity, setup consolidation, saved role-play setups, and compatibility/request-inclusion guardrails.

Final verification evidence:
- Focused role-play Vitest closeout suite passed from apps/tldw-frontend: 9 files, 60 tests. Non-fatal React key warning remains in the existing PromptSelect test Dropdown mock.
- Broader affected Playground Vitest suite passed: 5 files, 35 tests.
- `bun run lint` exited 0 with 130 pre-existing warnings and no errors.
- `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` passed; Next build completed and token sync verified OK. Running compile without NEXT_PUBLIC_API_URL fails the expected WebUI networking config validation.
- CDP first-time verification passed: Role-play setup discoverable, layers visible, saved setup area visible.
- CDP saved setup lifecycle verification passed: character switch, behavior template, generation style, scene edit/apply, recovery controls, save, rename, preview/apply, delete, and generic-template-hidden behavior.
- CDP mobile verification passed for role-play controls and wrapping: Generation style and Role-play setup reachable, no horizontal overflow, composer textarea visible. The measured textarea width was about 87.6px on a 390px viewport, which is cramped and should be treated as a broader mobile composer ergonomics risk outside this role-play slice.
- CDP compatibility verification passed for included, blended, excluded, override-risk, and persona-specific states.
- Bandit skipped because no Python files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the staged main /chat role-play preset remediation plan across six committed PR-sized slices. The workflow now has safer starter/recovery behavior, clearer role-play state terminology and chips, mobile access to role-play controls, a consolidated Role-play setup drawer, saved role-play setups backed by existing startup template persistence, and truthful request-inclusion guardrails for character/persona context. Final focused tests, lint, compile, and CDP verification are recorded; remaining TypeScript baseline failures are unrelated to this role-play work, and the only new UX risk noted is a cramped mobile composer textarea measurement outside the role-play control scope.
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
