# Folder-To-Notes Sources UI Exposure Design

Date: 2026-05-17
Status: Critique pass applied, pending user review
Owner: Codex brainstorming session
Backlog: TASK-400

## Summary

This spec defines the first slice for exposing folder-to-notes sync in the
WebUI and browser extension. The backend already has an ingestion source model
that can scan a server-local directory and sync files into Notes. The slice
should make that existing capability discoverable from Notes, add Sources to
shortcut/navigation surfaces, and avoid claiming bidirectional mirroring until a
later backend design extends the sync model.

This is mostly UI work, but it has one backend/capability precondition: the
direct folder-sync entry point must not be exposed unless the server can safely
advertise and enforce local-directory source access for the current user. The
original product boundary remains binding: single-user setups may enable this by
default; multi-user setups require administrator enablement at user or
organization scope.

The canonical management page remains Sources. Notes gets a focused entry point
that opens the existing new-source flow with a Notes folder-sync preset.

## Goals

- Make "sync a folder into Notes" reachable from the Notes page.
- Keep WebUI and extension behavior shared through `apps/packages/ui`.
- Reuse the existing Sources routes, API client, source form, detail page, item
  table, manual sync, schedule controls, and detached-conflict recovery.
- Add Sources to the shortcut/help surfaces without creating a parallel
  navigation system.
- Keep the copy honest: v1 imports and updates notes from a server-local folder;
  it is not bidirectional mirroring.
- Preserve the security boundary that local-directory sync reads server-host
  paths and must be controlled by backend capability/authorization checks.
- Avoid expanding local-directory source discoverability until the server
  exposes an explicit current-user capability for it.

## Non-Goals

- Implement bidirectional note-to-disk sync.
- Implement live filesystem watching. The approved first sync mode is manual
  and scheduled rescans.
- Add browser or extension access to arbitrary client-machine folders.
- Create a second folder-sync UI outside Sources.
- Change the notes database schema to match `tldw_chatbook` sync metadata.
- Replace the existing ingestion source worker, sink, or tracked item tables.
- Build the full admin UI for user/org local-folder-sync entitlement management.
  This first slice may consume the entitlement but does not need to become the
  complete admin-management surface.

## Current-State Evidence

The server already supports local directory sources that sync into Notes:

- [local_directory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Ingestion_Sources/local_directory.py)
- [notes_sink.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Ingestion_Sources/sinks/notes_sink.py)
- [ingestion_sources_worker.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/services/ingestion_sources_worker.py)
- [ingestion_sources.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py)
- [test_local_directory_sync_integration.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/tests/Ingestion_Sources/integration/test_local_directory_sync_integration.py)
- [test_notes_detached_integration.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/tests/Ingestion_Sources/integration/test_notes_detached_integration.py)

The shared WebUI/extension Sources surface already exists:

- [SourcesWorkspacePage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx)
- [SourceForm.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Sources/SourceForm.tsx)
- [SourceDetailPage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Sources/SourceDetailPage.tsx)
- [SourceItemsTable.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Sources/SourceItemsTable.tsx)
- [option-sources.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/option-sources.tsx)
- [sources.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/pages/sources.tsx)
- [option-sources.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/option-sources.tsx)

The shortcut/navigation surfaces are split across:

- [PageHelpModal.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/PageHelpModal.tsx)
- [KeyboardShortcutsModal.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/KeyboardShortcutsModal.tsx)
- [useShortcutConfig.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts)
- [HeaderShortcuts.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx)
- [header-shortcut-items.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/header-shortcut-items.ts)

One important issue discovered during review: the help modal lists mode
shortcuts, but the current shared layout only wires chat/sidebar/help shortcuts.
Implementation must not add a Sources row that is display-only. It should wire a
shared navigation shortcut handler or explicitly keep the Sources change to
launcher/search surfaces only. The recommended path is to wire the existing
mode shortcuts and Sources together.

The existing `SourceForm` already carries `schedule_enabled` through payloads,
but the current rendered form does not expose a schedule control. A first-class
"manual and scheduled rescans" UX therefore requires adding a visible schedule
toggle or explicitly deferring schedule configuration. The recommended path is
to add the toggle now, default it off, and avoid exposing cadence fields unless
the backend supports a concrete schedule contract.

## Critique Findings And Corrections

### Finding 1: Local Directory Access Cannot Be UI-Only

The ingestion sources API validates local-directory paths against configured
allowed roots, but the reviewed endpoint currently creates sources for the
authenticated user without an obvious single-user/admin-enabled entitlement
check in the route itself. Making a direct Notes "Sync folder" action prominent
would therefore widen discoverability of a filesystem-reading feature before
the per-deployment policy is explicit.

Correction:

- Implementation must start with a minimum local-directory source access gate.
- The UI may show the generic Sources page if `hasIngestionSources` is true.
- The Notes "Sync folder" action and `notes-folder-sync` local-directory preset
  require a more specific current-user capability, such as
  `hasLocalDirectoryIngestionSources`, `canCreateLocalDirectoryIngestionSource`,
  or an equivalent server-owned field.
- If the specific capability does not exist yet, the implementation plan should
  include a small backend capability/enforcement slice before the direct Notes
  entry point ships.
- UI-only hiding is not sufficient for multi-user safety.

### Finding 2: Shortcut Config Migration Needs A Fallback

Adding `modeSources` to `ShortcutConfig` can leave existing persisted
`keyboardShortcuts` objects without the new key. A navigation hook that reads
`configuredShortcuts.modeSources` directly could silently skip the shortcut or
throw if consumers assume all keys exist.

Correction:

- `useShortcutConfig` should merge persisted shortcut values over
  `defaultShortcuts` before returning them.
- Tests should cover a persisted legacy shortcut object that lacks
  `modeSources`.
- Display rows should use the resolved shortcut config, not only the static
  defaults, when they are meant to describe user-customized shortcuts.

### Finding 3: Header Launcher Selection Has A Legacy-Default Trap

Adding a `sources` header launcher item will not necessarily show for existing
users if they already have a persisted shortcut selection from before the new
ID existed. For users who customized the launcher, forcing Sources in would be
surprising; for users who only have the old full-default selection, hiding the
new page would weaken discoverability.

Correction:

- Add `sources` to `HEADER_SHORTCUT_IDS` and the Library group.
- Update selection coercion so legacy full-default selections gain `sources`.
- Do not force `sources` into genuinely customized, trimmed selections.
- Add a regression test for both cases.

### Finding 4: Schedule UX Is Assumed But Not Rendered

The spec said to keep scheduling visible, but the current form only carries
`schedule_enabled` through values and payloads; it does not render a schedule
switch.

Correction:

- Add a visible "Scheduled rescans" switch in the source form if this slice
  claims scheduled rescan support.
- Keep the default off.
- Do not add cadence controls unless the backend schedule object has a stable
  contract in this implementation slice.
- If the backend treats `schedule_enabled` as a fixed scheduler cadence, the UI
  copy should say "Use the server's scheduled rescan cadence."

## User Experience

### Notes Entry Point

The Notes page should expose a compact action named "Sync folder" or "Sync
folder to Notes." It should sit near the existing Notes management actions, not
inside an unrelated settings area.

Activating the action routes to:

```text
/sources/new?preset=notes-folder-sync
```

The new-source form reads this preset and defaults to:

- source type: `local_directory`
- destination: `notes`
- policy: `canonical`
- enabled: `true`
- schedule: off by default

The form should still show the normal path input and advanced controls. Users
can change settings before creating the source.

The Notes action should be disabled or hidden unless capabilities indicate all
of the following:

- the server is online
- Notes are available
- ingestion sources are available
- local-directory source creation is allowed for the current user

### Sources Form Copy

When the preset is active, the page should frame the flow as "Sync folder to
Notes" while preserving the existing source-management form. The path helper
must remain explicit:

```text
This is a path on the tldw server host, not a local browser or extension folder.
```

The preset copy should also say that the initial behavior is manual or scheduled
rescan. It should not use "mirror" for this first slice because source-to-notes
sync is not bidirectional yet.

### Shortcut And Launcher Exposure

Sources should be discoverable in three places:

- `PageHelpModal`: add "Go to Sources" under Navigation.
- `KeyboardShortcutsModal`: add the same legacy modal row for compatibility.
- `HeaderShortcuts`: add Sources under the Library group using the existing
  `/sources` route and `option:header.sources` label.

For the keyboard binding, add `modeSources` to `ShortcutConfig` with default
`Alt+2`. The existing mode sequence skips 2, so this fills the gap without
moving existing bindings.

The implementation must merge persisted shortcut settings with the expanded
defaults before any component reads `modeSources`.

Because the header shortcut launcher already uses command-number indexes while
the launcher is open, the Sources launcher item should not receive a
`shortcutIndex` unless a later design rebalances that launcher. The real Sources
shortcut for this slice is the global `Alt+2` mode shortcut.

Existing users with a legacy full-default header shortcut selection should get
Sources added automatically. Existing users with customized or trimmed launcher
selections should keep their choices.

### Navigation Shortcut Handler

The implementation should add one shared navigation shortcut hook instead of
special-casing Sources. It should map the configured mode shortcuts to routes:

- `modePlayground` -> `/chat` or the existing chat route convention
- `modeSources` -> `/sources`
- `modeMedia` -> `/media`
- `modeKnowledge` -> `/knowledge`
- `modeNotes` -> `/notes`
- `modePrompts` -> `/prompts`
- `modeFlashcards` -> `/flashcards`
- existing secondary mode shortcuts where already displayed

The hook should be called from shared layout code used by WebUI and extension
option routes. It should not fire while typing into inputs unless the existing
shortcut system already allows that specific shortcut.

## Data Flow

1. User opens Notes.
2. User selects "Sync folder."
3. UI navigates to the existing Sources new route with
   `preset=notes-folder-sync`.
4. `SourceForm` applies default local state from the preset.
5. User enters a server-local directory path and creates the source through the
   existing ingestion sources API.
6. User lands on the existing source detail page.
7. User runs "Sync now" manually or enables the existing schedule control.
8. Backend scans the allowed server-local path, creates or updates notes, and
   records tracked items.
9. Detached note conflicts continue to use the existing item status and
   reattach flow.

## Security And Deployment Boundary

This UI slice must not expand backend access. The backend remains the authority
for whether local-directory sources are allowed.

The current local-directory scanner already treats the path as a server-host
path and validates it against configured allowed roots. The UI should surface
backend rejection clearly instead of trying to validate filesystem access in the
browser.

For multi-user deployments, the backend must expose and enforce an explicit
entitlement/capability before this slice adds a direct local-folder Notes entry
point. The later bidirectional mirror design can expand the entitlement model,
but this first exposure still needs a minimum current-user local-directory
access decision:

- enabled by default only in single-user setups
- disabled by default in multi-user setups
- enableable by an administrator at user or organization scope

Until that entitlement exists, implementation should not ship the direct Notes
"Sync folder" entry point. It may still add neutral Sources-page discoverability
if the existing generic ingestion-sources capability is available.

## Error Handling

- Server offline: show the existing Sources offline state and avoid opening a
  dead-end form when possible.
- Unsupported server: show the existing unsupported Sources state.
- Disallowed path: keep the backend error visible near the path field and in
  the submit error region.
- Browser/extension path confusion: copy must say the path is on the server
  host.
- Source conflict or detached note: keep the existing Sources detail/item table
  status and reattach action.
- Schedule unavailable or rejected: keep manual "Sync now" available when the
  source itself is otherwise valid.

## Testing Plan

- Unit test the preset parser/defaults for `SourceForm`.
- Unit or component test that the Notes "Sync folder" action navigates to the
  preset route.
- Unit or API test the local-directory source entitlement behavior, if the
  capability/enforcement field does not already exist.
- Component test that the Notes action is disabled or hidden when Notes,
  ingestion sources, or local-directory source access is unavailable.
- Component test that `PageHelpModal` and `KeyboardShortcutsModal` include "Go
  to Sources" and display `Alt + 2`.
- Unit test `useShortcutConfig` merges legacy persisted shortcuts with new
  defaults so `modeSources` exists.
- Unit test the navigation shortcut hook maps `modeSources` to `/sources`.
- Header launcher test that Sources appears under Library and searches as
  "Sources."
- Header launcher test that legacy full-default selections gain Sources while
  customized selections are preserved.
- Existing Sources API/client tests remain the backend contract guard.
- Browser verification after implementation:
  - WebUI Notes -> Sync folder -> prefilled Sources form.
  - Extension options Notes -> Sync folder -> prefilled Sources form.
  - `?` help modal shows Sources.
  - `Alt+2` navigates to `/sources` outside text inputs.

## Later Bidirectional Mirror Work

The later feature that copies `tldw_chatbook` folder sync semantics should be a
separate backend design. It should compare Chatbook's `SyncDirection`,
`ConflictResolution`, file hash tracking, and profile loop against the server's
existing `ingestion_source_items` tracking and note-detached conflict model.

The likely extension points are:

- a new sink/export path for notes-to-disk writes
- explicit sync direction: disk-to-notes, notes-to-disk, bidirectional
- conflict policy: ask, disk wins, notes win, newer wins
- admin/user/org feature entitlement for local-folder sync
- background job visibility for scheduled runs
- safe path and allowed-root enforcement for every disk write

That work should not be folded into the UI exposure slice.

## Implementation Stages For Planning

1. Add or consume a server-owned current-user capability for local-directory
   source creation, and ensure direct Notes exposure respects it.
2. Add the Sources shortcut config, shortcut-config migration fallback,
   navigation shortcut hook, help modal rows,
   and launcher item.
3. Add the Notes page entry point and preset route builder.
4. Add `SourceForm` preset support, schedule switch, and focused copy for notes
   folder sync.
5. Add tests and browser verification for WebUI and extension parity.

## Decisions For Implementation Planning

- Notes should expose this as a secondary toolbar action named "Sync folder."
  If the Notes header is already crowded, place it in the existing overflow or
  action menu rather than making it a primary call to action.
- The preset path should keep scheduling visible because the approved v1 sync
  mode includes manual and scheduled rescans. The default remains off, and
  cadence controls are out of scope unless the backend contract already supports
  them.
- Existing generic capability/unsupported/offline states are enough for neutral
  Sources-page routing, but not enough for a direct local-directory Notes
  action. That action needs server-owned current-user authorization.
