---
id: TASK-13925
title: Redesign the chat header and conversation toolbar
status: In Progress
labels:
- chat
- webui
- design
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reimagine the two /chat header rows as a compact conversation bar and a quiet view toolbar, matching the separately approved composer and rails. User approved title/save state, Search/New chat/Share, Focus/Artifacts, and secondary utilities in a menu.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation title and acknowledged persistence/privacy state lead the compact header, with editing and save failure behavior retained.
- [x] #2 Search, new chat and sharing remain easy to reach; secondary navigation, appearance and chat-mode actions are available by keyboard and touch in a menu.
- [x] #3 The conversation view toolbar groups Focus, Artifacts and chat help without the duplicate theme control or decorative timeline label.
- [x] #4 Preserve notification lifecycle, unread counts, sharing restrictions, title ownership, existing callbacks and non-chat header behavior.
- [x] #5 Verify header and toolbar actions, light/dark desktop/mobile layout, keyboard focus, accessibility and draft preservation; record reviews and limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User approved both /chat header rows together: compact conversation bar anchored by title and save/privacy state, Search/New chat/Share, a quieter view toolbar for Focus/Artifacts, and secondary utilities in More. This separate branch, codex/chat-header-calm, begins at the accepted rails commit 566fdddeb3 in the existing attached managed worktree. TASK-492 is an older Companion Home launcher design record and does not duplicate this hierarchy work.

Design: Docs/Design/chat-header-calm.md. ADR required: no; existing route policy and callbacks are retained, with ADR-049 title/history ownership and sharing restrictions unchanged. The completed temporary three-stage implementation plan was removed per repository instructions. The composer, rails, backend and global design system were preserved; no dependencies were added.

The conversation title and acknowledged persistence state now lead the header. Search/new saved chat remain direct; sharing stays direct on desktop and is available in the mobile More menu with the same restrictions. More groups temporary/character entry, navigation, settings, theme and keyboard help. Notifications retain their lifecycle dialog and recovery actions. The lower row is a flat conversation-view group with Focus, chat help and Artifacts; the duplicate theme control and static timeline pill are removed. Existing non-chat header behavior remains the alternate branch of the existing showChatTitle route policy. New English strings are mirrored in the flat extension catalogs.

Tests were verified failing first for title/menu ownership, Escape focus, overlay handoff and artifact-count accessibility. After three Escape-focus iterations, implementation paused to inspect Ant/RC source and similar menus. Root cause: RC Dropdown schedules autofocus after three frames without cancelling it on close, so it can refocus a closed menu. The fix uses a cancellable frame and the existing visible-focus helper. Fresh code review also found the transient overlay return target, overridden artifact-count accessible name and obsolete toolbar test; all three were corrected and scored resolved. Every More action focuses its persistent trigger before invoking the existing callback, artifact counts remain in the toggle name, and the replacement view-group test runs in the current cockpit-controls fixture.

Final verification: 94 tests pass across eight focused header/title/share/launcher/notification/locale files, plus one replacement view-group contract (95 focused tests). Final locale cleanup recheck passes all three parity tests. TypeScript noEmit exits 0. ESLint comparison across seven touched TS/test files reports no new findings; inherited warnings are unchanged after excluding shifted line references. New option locale messages also match their extension mirrors. Five Chromium cases pass across desktop 1440x1000 and mobile 390x844 in light/dark: scoped axe has zero violations; default header stays at most 80px; no horizontal viewport overflow; More/Escape, real page launcher and keyboard-help return focus, Search and Focus-mode open/close, draft preservation, temporary-chat sharing restrictions and real artifact-counter accessibility pass. Six validated captures and manifest are in .impeccable/review/header/. The first batched visual inspection required no material correction, and the one-time detector returned zero findings.

Fresh code review: all three findings resolved. Fresh visual review: ship, with no material fixes at the supplied offline/local-save capture scope. Reviewer and documenter used the shipped skill's degraded/fallback instructions because no native role tool was available. The documenter compared source, CSS, product/design context and final evidence; it preserved global PRODUCT/DESIGN files and sidecar, leaving pre-existing context-loader scope/schema drift untouched.

Security and limits: Bandit was run recursively over touched UI directories and found no Python files, no findings and no errors, so it is an applicability check rather than TypeScript security coverage. Scoped lint and code review checked changed UI behavior. Live providers, microphone use, the full repository test suite and a production bundle were not checked. Local preview required temporary React aliases for the mixed dependency installation and full restarts to reload shared source; the exact original Next config has been restored. No preview workaround is shipped.
PR reconciliation: existing composer PR #3309 and rails PR #3310 were found and preserved. Rebuilt the header as a header-only change directly on dev commit 9acce42975, on codex/chat-header-pr-dev, excluding the original prototype branch ancestry. Current dev History on and Temporary chat (on this device) semantics, Search commands naming, Show pages/Hide pages non-chat behavior, shared keyboard-shortcuts dialog, and character/persistence callbacks are retained. Restored dev semantic-background regression coverage. The new toolbar test uses the existing real QueryClientProvider fixture.

Fresh dev reconciliation verification: 101 tests across eight focused suites, one view-controls contract, and 39 palette-host/focus-ownership tests pass (141 total). Installed TypeScript 5.9.3 noEmit exits 0 with the project 8 GB heap; a missing local workspace dependency symlink and bun's unrelated downloaded compiler were environment issues, with no dependency/config changes shipped. Seven-file ESLint comparison has no new findings; final test edits were checked separately. Fresh code review: ship, no remaining reconciliation findings. Bandit applicability remains no Python files/findings in touched UI directories.

Browser scope: unchanged dev header also fails to open the command palette in the same offline fixture; this is reproduced independently, not attributed to the redesign. The new offline browser contract verifies the existing Search open event, while palette ownership is covered by 39 passing component tests. Full palette opening with a valid account/history scope remains outside this offline check. Preview caches shared UI and requires a full restart after the baseline comparison. Final browser matrix and PR publication are pending. ADR required: no; existing ADR-049 ownership and action policy are unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Redesigned both chat header rows as a compact conversation bar and quiet view toolbar. Preserved title, sharing, notification, navigation and artifact behavior; resolved focus and accessibility review findings. Verified 95 focused tests, five desktop/mobile light/dark browser cases, TypeScript and no new lint findings. Design and scope: Docs/Design/chat-header-calm.md.
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
