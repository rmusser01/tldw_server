---
id: TASK-438
title: Implement Character Chat Phase 2 readiness errors and empty states
status: Done
assignee: []
created_date: 2026-05-19 04:18
labels:
- chat
- characters
- role-play
- phase-2
- frontend
- accessibility
dependencies: []
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- TASK-426
- TASK-428
- TASK-429
- TASK-431
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 2 from the first-class Character Chat PRD: make incomplete Character Chat setup, loading, no-provider, deleted/missing character, prompt/assistant catalog failures, and persistence state local and actionable on /chat. Preserve selected character intent through model/settings recovery and expose setup status changes to assistive tech.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing server, missing character, missing model, unavailable model, prompt load failure, and character catalog failure all have visible local Character Chat states.
- [x] #2 Selecting a character before model setup preserves character intent through settings handoff and retry.
- [x] #3 Character Chat setup/readiness status changes are exposed through appropriate live-region semantics.
- [x] #4 Focused unit/integration tests and real-backend browser smoke coverage are recorded; Bandit is run or explicitly skipped for non-Python scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow `Docs/superpowers/plans/2026-05-19-character-chat-phase2-readiness-plan.md`. 2. Add failing tests for readiness panel, selector failure states, persistence labels, and restored missing character recovery. 3. Implement minimal existing-surface UI changes. 4. Verify with focused Vitest, real-backend browser smoke where available, Bandit skip note for frontend-only scope, and closeout evidence.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Touched Files

- `apps/packages/ui/src/components/Option/Playground/CharacterChatReadinessPanel.tsx`
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx`
- `apps/packages/ui/src/utils/chat-model-availability.ts`
- `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts`
- `Docs/superpowers/plans/2026-05-19-character-chat-phase2-readiness-plan.md`

## Verification

- `bun run test -- src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/utils/__tests__/chat-model-availability.test.ts --maxWorkers=1 --no-file-parallelism` passed: 6 files, 88 tests.
- Real backend smoke used FastAPI on `127.0.0.1:8000` and WebUI on `127.0.0.1:8080`; `/chat?mode=character&characterId=missing-character` rendered `Character missing-character could not be loaded`, `Choose character`, and `Retry` after the real backend returned `404` for `/api/v1/characters/missing-character`.
- Browser smoke also confirmed fallback character intent persisted in local storage as `selectedCharacter` and `selectedAssistant` with id `missing-character`.
- `bunx tsc --noEmit --pretty false` still fails on existing repo-wide TypeScript debt; captured output in `/tmp/tldw_phase2_tsc_after_strict.txt` has 255 lines and no matches for the touched files listed above.
- Bandit skipped: this slice only touches frontend TypeScript/tests plus Backlog/plan documentation, with no Python source changes.

## Final Summary

Implemented local Character Chat setup/readiness states, selector loading/error states, persistence labels, and restored-character recovery without adding a parallel role-play UI. A real-backend smoke found a strict-mode route recovery race; added strict-mode coverage and fixed the stale in-flight guard so restored missing characters keep their recovery panel visible.

## Known Skips Or Blockers

- Repo-wide TypeScript baseline remains red outside this slice.
- Real browser console reports expected 404 warnings for the intentionally missing smoke-test character.
