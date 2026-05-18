---
id: TASK-431
title: Implement Character Chat Phase 1 first-class /chat mode
status: Done
labels:
- chat
- characters
- role-play
- phase-1
- frontend
priority: High
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- TASK-426
- TASK-428
- TASK-429
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 1 from the first-class Character Chat PRD: make /chat visibly support a durable Character Chat mode, support URL bootstrapping for mode=character and characterId intent, route starter/header/Characters launch paths into that mode, and prevent implicit active-chat clearing without explicit user intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /chat exposes and persists a Character Chat mode state from the first viewport/header starter path.
- [x] #2 /chat?mode=character enters Character Chat mode and optional characterId intent selects or preserves the intended character for a new/empty chat.
- [x] #3 /characters Chat and Chat in new tab launch into /chat with character intent where applicable.
- [x] #4 Header and starter Character Chat actions do not silently clear an active chat; they require explicit new-session intent, confirmation, or preserve the current session.
- [x] #5 Focused tests cover mode state, URL bootstrapping, launch handoff, and no implicit clearing; verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Phase 1 first-class character chat mode in the isolated worktree: durable chat workflow mode storage, /chat?mode=character route parsing, header character-mode dispatch without clearing active chat, /characters same-tab/new-tab launch URLs, visible chat-mode chip, route character hydration, and focused test coverage.

Verification so far:
- git diff --check passed.
- bunx vitest run src/routes/__tests__/route-paths.research.test.ts src/utils/__tests__/character-chat-mode-intent.test.ts src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --maxWorkers=1 --no-file-parallelism passed: 5 files, 104 tests.
- bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --maxWorkers=1 --no-file-parallelism passed: 13 tests.
- bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx --maxWorkers=1 --no-file-parallelism passed: 5 files, 43 tests.
Final verification after rebase/race fix:
- Rebased branch on latest origin/dev at e61681e99.
- Real backend was started with the project virtualenv and served http://127.0.0.1:8000; health and frontend-driven API calls returned 200 in backend logs.
- Fresh Next.js WebUI was started on http://127.0.0.1:8081 with NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 and verified in the browser against the real backend, not mocked routes. Browser initially exposed a storage-load race where /chat?mode=character could repaint to Standard chat; fixed by keeping character intent active independently from late storage hydration, then reloaded the real page and observed the visible Character Chat mode chip.
- git diff --check passed after the fix.
- bunx vitest run src/routes/__tests__/route-paths.research.test.ts src/utils/__tests__/character-chat-mode-intent.test.ts src/components/Layouts/__tests__/Header.character-mode.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --maxWorkers=1 --no-file-parallelism passed: 6 files, 117 tests.
- bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx --maxWorkers=1 --no-file-parallelism passed: 5 files, 43 tests.
- Bandit not run: touched implementation is TypeScript/TSX plus Backlog metadata only; no Python source changed.
- Known local artifact: real backend startup generated two untracked watchlist template files under tldw_Server_API/Config_Files/templates/watchlists; they are not part of this task and were left untracked.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 1 first-class Character Chat mode for /chat: durable character workflow state, /chat?mode=character URL bootstrapping with optional characterId hydration, header/starter intent dispatch without silent chat clearing, /characters launch handoff URLs, extension/WebUI URL consistency, and visible active chat-mode feedback. Real-backend browser verification found and drove a fix for a storage hydration race so route character intent remains visible on first load.
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
