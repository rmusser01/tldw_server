---
id: TASK-585
title: Address PR 2208 sidepanel handoff review comments
status: Done
references:
- TASK-485
- https://github.com/rmusser01/tldw_server/pull/2208
modified_files:
- apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts
- apps/packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx
- apps/packages/ui/src/components/Layouts/Layout.tsx
- apps/tldw-frontend/components/layout/WebLayout.tsx
- apps/extension/tests/e2e/sidepanel-options-handoff.spec.ts
- apps/extension/tests/e2e/sidepanel-chat-smoke.spec.ts
- Docs/superpowers/plans/2026-05-31-chat-siderail-edge-expand-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the actionable PR #2208 review feedback on the sidepanel WebUI chat handoff and chat rail collapse slice, including URL safety, fragment transport, handoff expiry, prompt/history restoration, localized rail labels, focused tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #2208 review clusters: validated explicit WebUI URLs as http(s), preserved explicit WebUI subpaths, moved encoded sidepanel WebUI payloads from query string to URL fragment, enforced a 10 minute max handoff age, preserved explicit prompt clear values, restored local history IDs via loadLocalConversation, localized the desktop chat edge label from the existing chat sidebar title key, and fixed the plan mock snippet noted by cubic. Verification: focused Vitest suite passed with 6 files / 28 tests; `bun run compile` passed in apps/extension; `bun run build:chrome` completed successfully with existing Rollup/chunk warnings; `git diff --check origin/dev...HEAD` passed. Focused Playwright sidepanel handoff e2e was attempted with `.output/chrome-mv3` and `TLDW_E2E_EXTENSION_HEADLESS=1`, but both tests were skipped by the harness because Chromium extension launch aborts in this environment (`launchPersistentContext` closed/SIGABRT, kill EPERM). Bandit not run because this slice only touches TypeScript/TSX/docs/backlog files, not Python.
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
