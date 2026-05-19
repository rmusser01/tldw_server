---
id: TASK-434
title: Harden character chat real-backend E2E and quickstart proxy
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 01:56'
labels:
  - chat
  - characters
  - role-play
  - frontend
  - e2e
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up hardening for first-class /chat character role-play after PR merges: make the compact character selector operable in the real toolbar, preserve backend-canonical chat creation routes through the Next quickstart proxy, and keep the real-backend character journey from passing on catalog-only/unconfigured model inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compact character/persona selector trigger opens the real AssistantSelect menu from the /chat toolbar.
- [x] #2 Quickstart Next.js proxy preserves backend-canonical trailing slash chat endpoints such as POST /api/v1/chats/.
- [x] #3 Real-backend character E2E preflight prefers runnable model metadata and does not skip provider-error recovery on catalog-only models.
- [x] #4 Focused unit tests and real backend/WebUI Playwright verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in branch codex/character-chat-stage2-visible-state after rebasing onto latest origin/dev.

Changes:
- Fixed the compact AssistantSelect trigger so AntD Dropdown wraps the real button directly, with accessible title and expanded state.
- Hardened the E2E page object to target the character selector instead of broad navigation buttons.
- Added Next quickstart trailing-slash preservation and regression coverage for backend-canonical /api/v1/chats/.
- Updated real-backend E2E preflight to prefer /api/v1/llm/models/metadata, filter unconfigured/catalog/deprecated model entries, and fall back to /api/v1/llm/providers only when necessary.
- Updated the character journey to exercise visible provider-credential recovery instead of skipping when no runnable model is available.

Verification:
- git diff --check passed.
- bunx vitest run __tests__/frontend-quickstart-networking.test.ts __tests__/e2e-fixture-models.test.ts --reporter=verbose passed: 2 files, 11 tests.
- bunx vitest run src/services/__tests__/tldw-api-client.chat-mutations.test.ts src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --reporter=verbose passed: 2 files, 20 tests.
- Real backend/WebUI E2E passed against backend http://127.0.0.1:18001 and WebUI http://localhost:8081: character-chat.spec.ts --project=journeys --reporter=line passed: 1 test.
- Bandit not run: touched implementation is TypeScript/TSX/frontend test/config plus Backlog metadata only; no Python source changed.

Known local artifacts:
- Backend startup generated two untracked watchlist template files under tldw_Server_API/Config_Files/templates/watchlists; they are unrelated and left untracked.
- A stale duplicate untracked Backlog task file with TASK-433 was not committed because TASK-433 already belongs to unrelated VZ work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the first-class character chat path around real backend behavior: the compact role-play selector now opens reliably, quickstart preserves backend-canonical chat creation routes with trailing slashes, and the E2E harness filters to runnable model metadata while still validating visible provider-error recovery. Focused unit tests and the real backend/WebUI character journey pass.
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
