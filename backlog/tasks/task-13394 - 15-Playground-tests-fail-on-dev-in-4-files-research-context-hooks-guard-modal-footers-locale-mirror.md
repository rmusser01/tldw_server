---
id: TASK-13394
title: >-
  15 Playground tests fail on dev in 4 files (research context, hooks guard,
  modal footers, locale mirror)
status: Done
assignee: []
created_date: '2026-09-28 19:59'
updated_date: '2026-09-29 15:33'
labels:
  - frontend
  - tests
  - ci
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found 2026-09-28 while burning down TASK-13390. From apps/packages/ui, the way CI runs them, these fail identically on dev with no local changes:
- Playground.research-context.integration.test.tsx: 12 tests (follow-up surfaces, auto-restore of attached research context, pin/unpin, StrictMode readiness)
- PlaygroundHooks.jsx-extension.guard.test.ts: 'stores JSX-bearing hooks in .tsx modules'
- PlaygroundModalFooters.design-system.test.tsx: startup template footer buttons
- playground-locale-mirror.test.ts: English playground strings not mirrored into the extension locale

Like the Media and chat-submit breaks fixed in #3035/#3046, these are hidden by the frontend ratchet, which only runs impacted tests, and will block any PR whose changes reach these files. Two of them (the hooks .tsx guard and the locale mirror) are source-contract checks and likely point at real drift, not stale tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each file triaged: stale test (fix the test) vs real drift (fix the code), recorded here
- [x] #2 All four files pass from apps/packages/ui
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-29 triage (AC1). One real drift, three stale tests:
- playground-locale-mirror.test.ts: REAL DRIFT. 54 English playground strings were never mirrored into public/_locales/en/playground.json; synced with the repo's own apps/extension 'locales:sync' script, scoped to playground.json (54 keys added, 0 removed, 0 changed, the rest reordering). The test also had a latent key bug: it joined nested keys raw, so the literal dotted keys under 'actor' ('preset.sliceOfLife', 'templateMode.merge', ...) became 'actor_preset.sliceOfLife'. Chrome message names allow only [A-Za-z0-9_], and sync-public-locales.js writes 'actor_preset_sliceOfLife'; the test now sanitizes segments the same way.
- PlaygroundHooks.jsx-extension.guard.test.ts: STALE. usePlaygroundPersistence.tsx still carries JSX, but #1987 replaced antd <Button with a native <button; the marker moved to '<button'.
- PlaygroundModalFooters.design-system.test.tsx: STALE FIXTURE. The preview is cast from an object without systemPrompt, which PlaygroundStartupTemplate declares as a required string (loaders normalize it); describeRolePlaySetupPreview now reads it. Added systemPrompt: ''.
- Playground.research-context.integration.test.tsx (12): STALE. da0f1cd3a3 (2026-09-23) gated the attachment restore and persistence on historySelection.settingsMode(serverChatId) !== 'pending', which holds only once history selection has bound a native owner via loadConversation. The test (last edited 2026-09-05) injects serverChatId through a mocked useMessageOption and never binds an owner, so settingsMode stayed 'pending' and nothing restored. It now mocks useHistorySelectionContext to report an ordinary owner, the pattern Playground.search.integration.test.tsx already uses.
AC2 / verification: all four files pass (18/18 research context). Whole src/components/Option/Playground: 115 files, 861 tests passed. DoD: no docs change; bandit N/A (frontend tests and locale data); no skips.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
