# Chat Sidebar Tools-First Expansion Design

Date: 2026-05-17
Status: Approved for planning
Owner: Codex brainstorming session
Backlog: TASK-401

## Summary

This spec defines the shared WebUI/extension chat sidebar behavior for opening
or expanding the sidebar. Every sidebar open should present tools and shortcuts
first: the `Shortcuts` section is expanded, and recent conversations are
collapsed. Users can still open recent conversations when they intend to browse
history, and search should keep conversation results reachable.

The fix should live in the shared `ChatSidebar` surface used by the WebUI and
extension shells. It should not become route-specific behavior, and it should
not redesign the separate sidepanel active-chat drawer unless that drawer is
confirmed to have the same shortcuts/history disclosure model.

## Goals

- Make every chat sidebar open land on the tools/shortcuts affordances.
- Collapse recent conversations by default on each open or expand transition.
- Preserve explicit user intent to browse or search conversations.
- Keep server-history loading lazy while recent conversations are collapsed.
- Keep WebUI and extension behavior aligned through shared UI code.
- Add regression coverage for desktop expanded sidebar, direct open/mount, and
  recent-history expansion.

## Non-Goals

- Redesign the broader chat page, sidepanel chat route, or layout shell.
- Change chat history APIs or server-side conversation storage.
- Remove or rewrite server/folder chat lists.
- Change the user's selected shortcut list.
- Persistently force recent conversations closed while the sidebar is already
  open and the user has manually expanded it.

## Current State Evidence

The shared persistent chat sidebar lives in:

- [ChatSidebar.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar.tsx)
- [ServerChatList.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx)
- [FolderChatList.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar/FolderChatList.tsx)
- [shortcut-actions.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/ChatSidebar/shortcut-actions.ts)
- [ui-settings.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/settings/ui-settings.ts)

The sidebar is mounted from both shared and Next.js layout shells:

- [Layout.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/Layout.tsx)
- [WebLayout.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/components/layout/WebLayout.tsx)

The current `ChatSidebar` already has a persisted `shortcutsCollapsed` setting
and a server/folder conversation tab, but the recent conversation area is not a
separate disclosure. That lets recent conversations dominate the expanded
sidebar while shortcuts can remain collapsed from prior navigation.

The extension sidepanel chat route also has a different
`SidepanelChatSidebar`:

- [Sidebar.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/Sidebar.tsx)
- [sidepanel-chat.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/sidepanel-chat.tsx)
- [sidepanel-chat.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/sidepanel-chat.tsx)

That component is primarily an active tab and history drawer, and does not
currently expose the same tools/shortcuts section. It is out of scope unless a
follow-up confirms the same UX contract should apply there.

## Behavior Contract

Every sidebar open event should reset the shared `ChatSidebar` presentation to
tools-first:

- `Shortcuts` is expanded.
- `Recent conversations` is collapsed.
- Footer tools remain available.
- The current server/folder tab choice is preserved.
- The selected shortcuts are preserved.
- Search text is preserved while the user is actively searching, and a
  non-empty search query must keep conversation results keyboard-reachable.

This applies to:

- Desktop WebUI/sidebar icon rail expansion.
- Mobile chat sidebar drawer opens.
- Programmatic sidebar open events such as `tldw:open-chat-sidebar`.
- Direct expanded mounts, such as when the drawer opens with
  `collapsed={false}`.

Route navigation can still collapse shortcuts during navigation if that remains
useful for page transition behavior. The next open or expand transition must
restore the tools-first state.

When the user expands recent conversations, the current server/folder list
behavior should continue unchanged. When the user searches in the sidebar,
conversation results must remain reachable because search is explicit
history-browsing intent. Implementation can satisfy this either by auto-opening
the `Recent conversations` disclosure while `searchQuery.trim()` is non-empty,
or by rendering a search-results region outside the collapsed recent-history
body. It must not hide the search input or results behind a collapsed disclosure
while a query is active.

## Architecture

The state owner should be `ChatSidebar`, because it owns the shortcuts
disclosure, server/folder tab, search input, and conversation list rendering.
Parent layouts should not duplicate sidebar-internal presentation state.

`ChatSidebar` should maintain two independent disclosure states:

- `shortcutsCollapsed`, backed by the existing
  `SIDEBAR_SHORTCUTS_COLLAPSED_SETTING`.
- `recentCollapsed`, backed by a new sidebar recent-conversations setting or a
  local component state if persistence is not needed.

On every open/reset event:

- call `setShortcutsCollapsed(false)`;
- set `recentCollapsed` to `true`;
- leave `currentTab`, `shortcutSelection`, and `searchQuery` intact;
- if `searchQuery.trim()` is non-empty, keep recent search controls/results
  reachable despite the default collapsed state.

Open/reset detection should cover both transitions and direct expanded mounts.
A straightforward implementation is to detect `collapsed` changing from `true`
to `false`, and also run the reset when the component mounts already expanded.
This handles desktop expansion and mobile drawer mounting without requiring
each layout to know about sidebar internals.

If implementation proves that direct mount detection causes redundant writes,
use a small internal ref to run the reset only when the visible state becomes
expanded.

## Component Flow

The expanded `ChatSidebar` should be reorganized around two disclosure blocks:

1. `Shortcuts`
   - Uses the existing shortcut rendering.
   - Expanded after every sidebar open.
   - Still user-toggleable while the sidebar is open.

2. `Recent conversations`
   - Wraps the existing search input, server/folders segmented control, and
     server/folder list content.
   - Collapsed after every sidebar open.
   - User-toggleable while the sidebar is open.
   - Shows existing `ServerChatList` and `FolderChatList` when expanded.
   - Must auto-expand or render a reachable search-results region when
     `searchQuery` is non-empty.

The existing footer remains below the main sidebar content. If recent
conversations are collapsed, shortcuts and footer tools should be immediately
visible instead of being pushed below a history list.

Server-history selection controls must be scoped to the visible recent-history
body. The existing select-chats action should be hidden or disabled while
recent conversations are collapsed, and `selectionMode` should exit when recent
conversations collapse so hidden rows cannot remain selected.

## Data And Loading

Server history should remain lazy:

- When recent conversations are collapsed and search is empty, do not mark
  `server-history` visible or engaged.
- When the user expands recent conversations on the server tab, mark
  `server-history` visible and allow overview fetching through the existing
  coordinator gate.
- When search is non-empty, continue to mark history engaged so search results
  can load.
- Server tab badges or count labels must not bypass this gate. They should
  either use already-available cached data or stay unloaded until recent
  conversations are expanded or search is active.
- Folder list behavior should remain local and unchanged.

This preserves the performance intent of the current lazy-history tests while
changing the visual default.

## Error Handling

No new backend errors are introduced. If settings writes fail, the sidebar
should still update local visible state for the current render. Existing
settings helpers should handle storage failures the same way they do for other
sidebar preferences.

If server history is unavailable, the existing `ServerChatList` loading, stale
data, and error states should remain unchanged and should only appear after the
user expands or searches recent conversations.

## Accessibility

- Both `Shortcuts` and `Recent conversations` controls should expose
  `aria-expanded`.
- Each disclosure should have an `aria-controls` target.
- The collapsed icon rail behavior and tooltips should stay unchanged.
- Keyboard users should be able to expand recent conversations and reach the
  search input, segmented control, and list without relying on pointer input.

## Tests

Focused tests should cover:

- Rendering `ChatSidebar` collapsed, expanding it, and asserting shortcuts are
  visible while recent conversation content is hidden.
- Rendering `ChatSidebar` already expanded and asserting it initializes
  tools-first.
- Clicking `Recent conversations` and asserting the search input, server/folder
  controls, and list region become visible.
- Expanding recent conversations, closing/collapsing the sidebar, reopening it,
  and asserting the sidebar resets to shortcuts visible and recent content
  hidden.
- Collapsing recent conversations while server selection mode is active and
  asserting selection mode exits or selection controls become unavailable.
- Searching in the sidebar and asserting conversation results remain reachable.
- Updating the existing lazy-history test so collapsed recent history does not
  trigger the server overview request.
- Proving the server chat count badge does not trigger the overview fetch while
  recent conversations are collapsed and search is empty.

Manual/browser verification should check desktop WebUI and extension/mobile
drawer behavior if the app runs cleanly from the current checkout.

## Implementation Plan Handoff

The implementation plan should stay small:

1. Add recent-conversation disclosure state and reset-on-open behavior to
   `ChatSidebar`.
2. Wrap search/tabs/list content in the new disclosure.
3. Adjust coordinator visibility and lazy-history gating to account for
   `recentCollapsed`.
4. Add/update targeted component tests.
5. Run focused frontend tests and, if possible, browser-check the visible
   desktop and drawer states.
