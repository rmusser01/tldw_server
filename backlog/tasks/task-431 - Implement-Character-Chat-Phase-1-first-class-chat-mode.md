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
PR review remediation reopened TASK-431. Live PR #1846 review sweep found actionable Qodo/Gemini/CodeRabbit comments in Playground route-intent handling: one-shot route hydration to avoid reverting manual character switches, validation for URL characterId before API hydration, no partial fallback set from header intent, fallback on null character responses, route flag included in first-render workflow state, reset for all non-character starter modes, and character-only label rendering in the Character Chat chip.
PR review feedback addressed for #1846:
- Consolidated character route-intent parsing and normalized URL characterId before hydration, so malformed ids such as ../secret do not trigger backend character hydration.
- Made route character hydration one-shot per route id with in-flight/applied guards so manual character switches are not reverted by the URL effect.
- Removed partial selected-character writes from the header character-mode event; the header now changes mode only.
- Added typed fallback character handling for null/failed route hydration responses.
- Kept route-requested Character Chat active on first render, reset all non-character starter modes back to standard workflow, and limited the mode chip secondary label to actual character assistants/characters rather than persona names.

Verification after PR review fixes:
- git diff --check passed.
- bunx vitest run src/utils/__tests__/character-chat-mode-intent.test.ts src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --maxWorkers=1 --no-file-parallelism --reporter=verbose passed: 2 files, 22 tests.
- bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx --maxWorkers=1 --no-file-parallelism --reporter=verbose passed: 5 files, 43 tests. Existing test stderr about tldw server not configured remained non-fatal.
- Real backend/WebUI smoke used uvicorn on http://127.0.0.1:8000 and Next.js on http://localhost:8081 with a Playwright browser. /chat?mode=character&characterId=..%2Fsecret displayed Character Chat and made zero /api/v1/characters requests; /chat?mode=character&characterId=missing-character displayed Character Chat and made the expected real backend /api/v1/characters/missing-character request, which returned 404.
- Bandit not run: only TypeScript/TSX UI files and Backlog metadata changed; no Python source changed.
- Left unrelated untracked watchlist template files out of the task/commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 1 first-class Character Chat mode for /chat and addressed PR #1846 review feedback: durable character workflow state, validated route characterId handling, one-shot route hydration that does not revert manual switches, mode-only header intent dispatch, typed fallback hydration, broad non-character starter reset, character-only chip labels, and /characters launch handoff. Verified with focused Vitest coverage and a real backend/WebUI browser smoke against uvicorn and Next.js.
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
