---
id: TASK-552
title: Verify sidepanel chat handoff regression and packaged smoke
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 07:49'
labels:
  - chat
  - extension
  - verification
dependencies: []
references:
  - TASK-546
  - TASK-547
  - TASK-548
  - TASK-549
  - TASK-551
documentation:
  - Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
  - >-
    Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 4 from the sidepanel chat WebUI handoff plan: run focused unit regressions, relevant existing playground/sidepanel tests, UI type/build sanity, packaged/browser smoke where available, record evidence and skips, and close the sidepanel chat handoff implementation verification slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused sidepanel handoff unit regression set passes or any unrelated baseline failures are documented.
- [x] #2 Existing relevant playground/sidepanel tests pass or failures are documented with exact failing tests.
- [x] #3 UI type/build sanity is run and recorded.
- [x] #4 Packaged/browser smoke is run when a harness is available, otherwise the skip reason is recorded.
- [x] #5 Bandit skip reason is documented for UI-only TypeScript/markdown scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification evidence:

- Focused handoff regression set passed from `apps/packages/ui`:
  `bun run test src/services/__tests__/sidepanel-chat-handoff.test.ts src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx --maxWorkers=1 --no-file-parallelism`
  Result: 5 files, 36 tests passed. Only the known Node localStorage ExperimentalWarning was emitted.
- Existing relevant playground/sidepanel regression set passed from `apps/packages/ui`:
  `bun run test src/utils/__tests__/sidepanel-full-app-route.test.ts src/utils/__tests__/character-chat-mode-intent.test.ts src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --maxWorkers=1 --no-file-parallelism`
  Result: 4 files, 46 tests passed. Existing provider-status mock warnings were emitted (`getProvidersStatus is not a function` and mock provider config fetch warnings); no tests failed.
- UI type sanity passed from `apps/packages/ui`:
  `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`
- Packaged extension smoke harness exists at `apps/extension/tests/e2e/sidepanel-chat-smoke.spec.ts` and global setup built `.output/chrome-mv3` successfully. The full smoke command was run from `apps/extension`:
  `npx playwright test tests/e2e/sidepanel-chat-smoke.spec.ts --project=chromium-extension --reporter=line --workers=1`
  Result: 3 skipped. JSON follow-up for the handoff case reported `Extension launch unavailable in this environment (TimeoutError: browserType.launchPersistentContext: Timeout 30000ms exceeded...)`.
- Additional packaged-smoke diagnosis:
  `npx playwright test ... --grep "keeps packaged /chat"` outside the sandbox still skipped at the default browser launch timeout.
  `TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS=90000 npx playwright test ... --grep "keeps packaged /chat"` reached the longer launch wait but failed the test timeout before `/chat` assertions ran.
  `TLDW_E2E_EXTENSION_HEADLESS=1 TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS=90000 npx playwright test ... --grep "keeps packaged /chat"` skipped with `Could not determine extension id from [no extension targets]`.
  Conclusion: packaged browser smoke is unavailable on this host because Chrome extension launch does not become controllable; no packaged `/chat` product assertion ran or failed.
- Bandit skipped: touched verification scope is TypeScript/TSX/markdown and generated extension build output only; no Python paths were changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Task 4 verification pass for sidepanel chat handoff. Focused unit regressions, existing relevant playground/sidepanel regressions, and UI type sanity all pass. The packaged smoke harness builds successfully but cannot execute product assertions on this host because Playwright cannot complete Chrome extension launch in headful mode, and CI-style headless mode starts without extension targets.
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
