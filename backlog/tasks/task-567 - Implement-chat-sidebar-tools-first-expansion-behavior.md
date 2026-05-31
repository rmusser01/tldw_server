---
id: TASK-567
title: Implement chat sidebar tools-first expansion behavior
status: Done
labels:
- webui
- extension
- chat
- sidebar
- ux
priority: high
references:
- TASK-401
- TASK-404
documentation:
- Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md
- Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved shared ChatSidebar tools-first behavior from TASK-401/TASK-404: every sidebar open or foreground should show shortcuts/tools expanded, keep recent conversations collapsed until explicitly expanded or searched, gate lazy history loading and selection controls behind recent visibility, and wire open-reset signals from shared WebUI/extension layout shells.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChatSidebar opens shortcuts/tools first and keeps Recent conversations collapsed on direct expanded mount.
- [x] #2 Collapsed-to-expanded sidebar opens and explicit layout open-reset signals restore the tools-first state.
- [x] #3 Recent conversations, search, selection controls, history rendering, lazy history loading, and coordinator visibility are gated behind recent visibility or active search.
- [x] #4 Shared WebUI and package layout shells pass `openResetKey` to desktop and mobile ChatSidebar mounts.
- [x] #5 Focused sidebar/layout/WebLayout tests and a live browser check verify the current merged behavior; no production UI code changes were needed in this branch because the implementation is already present on `origin/dev`.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-05-17-chat-sidebar-tools-first-expansion-implementation-plan.md with TDD slices for ChatSidebar tools-first reset, recent disclosure gating, lazy history/coordinator visibility, layout openResetKey wiring, focused tests, browser verification where practical, and Backlog finalization.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created this implementation tracker after forking the thread onto sidebar-related work, then verified the current `origin/dev` baseline at merge commit `b91af76404e91bdad8b9b19727c4c2f8d1eefe7a`. The planned sidebar behavior is already implemented in the clean baseline: `ChatSidebar` owns `recentCollapsed`, `recentHistoryVisible`, `openResetKey` reset handling, recent disclosure rendering, history/control gating, and coordinator visibility; `Layout.tsx` and `WebLayout.tsx` both pass `openResetKey` into desktop and mobile ChatSidebar mounts. Because the production and test code already matched the approved plan, this branch only records verification and Backlog closeout.

Verification:
- `cd apps/packages/ui && bun install` was required in the fresh worktree because initial `bunx vitest`/`bun run test` attempts could not resolve the package-local Vitest install.
- `cd apps/packages/ui && bun run test src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts` passed: 4 files, 14 tests.
- `cd apps/tldw-frontend && bunx vitest run __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx` passed: 1 file, 5 tests.
- Live browser check used `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -H 127.0.0.1 -p 18051` with approved local binding. After setting local `apiKey`, `tldwConfig`, and `__tldw_test_bypass`, `/chat` showed the collapsed chat sidebar. Expanding it showed Shortcuts open and Recent conversations closed; manually expanding Recent exposed search; collapse/reopen reset back to Shortcuts open and Recent closed.
- The first dev-server attempt failed until `NEXT_PUBLIC_API_URL` was supplied. A sandboxed bind attempt then failed with `EPERM`, so the browser check used approved elevated localhost binding. The server was stopped afterward and `lsof -nP -iTCP:18051 -sTCP:LISTEN` confirmed no listener remained.
- Browser console showed backend/unreachable-style errors during the local-only check because no FastAPI backend was running on `127.0.0.1:8000`; this did not block verification of the sidebar open/reset behavior.
- Bandit skipped: this closeout branch changes Backlog Markdown only; no Python files or production source files were touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and closed the sidebar tools-first implementation against latest `origin/dev`. The current baseline already implements the approved behavior: chat sidebar opens tools/shortcuts first, keeps Recent conversations collapsed by default, gates history/search/selection/coordinator work behind recent visibility or active search, and receives reset signals from both shared layout shells. This branch records the focused test and live browser evidence only.

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
