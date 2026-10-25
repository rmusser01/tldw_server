---
id: TASK-13934
title: Redesign the chat history sidebar as a calm conversation navigator
status: In Progress
labels:
- frontend
- chat
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resume the separately requested chat history sidebar redesign from current dev after PRs 3309, 3310, 3311, 3313 and 3315 merged. Recover the existing navigation, ownership and preference contracts, confirm the direction with the requester, and improve hierarchy without changing data ownership or management permissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The requester confirms the sidebar design direction before UI implementation.
- [x] #2 Search, new chat, source/type filters, pinned/recent conversations, folders, and shortcuts remain accessible with their existing ownership and saved preferences.
- [x] #3 Focused behavior tests, TypeScript, scoped lint and bounded desktop/mobile light/dark review validate the finished sidebar.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting point: origin/dev 1deeea56a6ec1a1c81a872986c2135baa8ba41c0. User resumed sidebar work on 2026-10-25. Existing work was paused during direction selection; no sidebar implementation or approved sidebar design was found. A direction question is pending. Worktree reused on codex/chat-history-sidebar-calm; completed research branch and all original worktrees/evidence are preserved. Primary checkout remains on codex/rpg-dice-engine.
Read-only baseline: all 152 existing sidebar tests across 8 suites passed on current dev (log /private/tmp/chat-sidebar-baseline-20261025.log). Relevant source is Common/ChatSidebar.tsx plus ServerChatList/ServerChatRow, LocalChatList and FolderChatList; source/type controls, pinned/recent grouping, selection and ownership handling already exist. Direction selection is pending; UI implementation has not started. ADR required: no for this presentation scope; ADR-049 (Docs/ADR/049-chat-history-selection-ownership.md) continues to govern history ownership. Impeccable context resolved the nested package as root and did not find the existing root DESIGN.md; root design tokens and established chat design briefs were read directly, with original PRODUCT.md treated as existing product context. No global context drift repair is in scope.
Requester selected option 1 on 2026-10-25 and asked whether folders would be removed. Confirmed folders stay as a visible source destination with existing organization tools and saved preferences. Approved bounded implementation recorded in Docs/Design/chat-history-sidebar-calm.md; three-stage own plan IMPLEMENTATION_PLAN_chat_history_sidebar_20261025.md. Current dev freshly fetched and unchanged at 1deeea56. ADR required: no; ADR-049 remains governing.
Implementation complete; qualification in progress. Folders remains a visible radio destination alongside All/Server/On this device, and clicking the already-selected destination reopens collapsed history. Search/source controls stay above the shared history/shortcut scroller; New Chat is labeled and fits narrow widths. Server rows are flat and pin/unpin is in the existing menu, with source/fork identity and wrapped metadata preserved. Final focused sidebar run: 158/158 tests across 8 suites pass (/private/tmp/chat-sidebar-unit-final-20261025.log). Normal frontend TypeScript noEmit with 8GB heap passed (/private/tmp/chat-sidebar-typescript-20261025.log). Scoped ESLint checked 7 files: 0 errors/5 messages byte-identical in rule/severity/text to origin/dev (/private/tmp/chat-sidebar-eslint-comparison-20261025.json). Advanced webpack wrapper passed, but does not certify CI Turbopack. Independent sidebar_code_review resolved metadata clipping and the AntD radio handler issue and reports no material code findings. Bandit scanned 0 Python LOC; no TypeScript security clearance. Browser desktop/mobile light/dark qualification is running with synthetic API data, real folder controls, menu actions, draft/scroll/overflow and scoped axe.
Final qualification: 476 focused tests / 37 suites passed (409 shared sidebar/Layout + 67 web Layout). New real-Drawer regression failed first on the redundant Close control; both actual web and shared mobile callers now share full available sizing and one heading, retaining accessible dialog name and collapse/Escape/selection callbacks. All four desktop1440/mobile390 light/dark browser scenarios passed folder access, native radio keyboard navigation, search restoration, pin/unpin, draft retention, independent scrolling and overflow; mobile footer and150px control bounds passed. Scoped axe0/pageErrors0. Log /private/tmp/chat-sidebar-browser-final-20261025.log; valid captures .impeccable/review/sidebar. Normal frontend TS8GB passed, final installed lint11files0errors/7inherited rule/severity/message tuples identical to dev (/private/tmp/chat-sidebar-eslint-final-comparison-20261025.json), webpack wrapper passed (/private/tmp/chat-sidebar-web-frame-build-20261025.log). First visual review identified source-grid/theme/contrast, mobile frame/footer and narrow footer issues; one repair batch corrected them, including the separate actual WebLayout caller, and the reviewer scored all five fixes resolved with disposition ship at that scope. Fresh code review clean. Local duplicate React required temporary build aliases; no dependency/config change shipped and exact original Next config restored. Incumbent DESIGN.md/.impeccable/design.json preserved. Synthetic data only; live backend/providers/microphone, full repo/hosted CI/Turbopack not checked. Bandit0PythonLOC provides no TS clearance. Own generated build/trace trees and all prior evidence preserved outside checkout.
Requester approved the final preview (ok that looks good) and then requested continuation. Publication/CI phase resumed from unchanged dev1deeea56 on codex/chat-history-sidebar-calm. Exact commit df3653530a requalified:409sharedUI/Layout tests34suites and67web-layout tests3suites passed again (/private/tmp/chat-sidebar-pr-ui-20261025.log and chat-sidebar-pr-web-20261025.log). Opening a reviewable draft PR with protected auto-merge off; no sidebar-specific human Change summary or waiver has been supplied, so that merge policy remains pending. No duplicate task/new feature scope; ADR-049 and approved design unchanged. Preserve all worktrees, dependency symlink and review evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved calm chat history navigator from dev1deeea56 while keeping Folders visible and preserving ownership, read-only permissions, search and saved preferences. Fixed search/New chat and compact sources lead flat rows; pin/unpin is in the existing menu. Both mobile callers fit their drawer and footer controls wrap at150px. Qualified with476 focused tests, normal TypeScript, zero lint errors/no new findings, four-theme/viewport browser scenarios and scoped axe0, independent code review and five-fix visual ship verdict. No dependency/workflow/global design-system change; synthetic/live-system and CI limits documented in the design brief.
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
