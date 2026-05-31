# Folder Notes Sources UI Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing Sources local-directory-to-Notes workflow in the shared WebUI/extension, with a real Sources shortcut and a server-owned single-user/admin-enabled multi-user permission gate.

**Architecture:** Keep folder sync behavior server-owned: the backend decides whether the current user can create local-directory ingestion sources, the frontend only reflects that capability. The UI adds a Notes "Sync folder" entry point, a `/sources/new?preset=notes-folder-sync` preset, visible schedule controls, and real mode navigation shortcuts shared by WebUI and extension layouts.

**Tech Stack:** FastAPI, existing AuthNZ/admin feature-flag service, existing ingestion sources core/service layer, React, Ant Design, React Router, Plasmo storage, Vitest, pytest, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-17-folder-notes-sources-ui-exposure-design.md`
- Backlog task: `TASK-403`
- Implementation task should be split from this planning task before code edits begin if a new worker executes it.
- Admin feature flag key: `ingestion_sources.local_directory`

## Non-Goals

- Do not implement bidirectional filesystem mirroring in this slice.
- Do not add browser or extension local-folder filesystem access.
- Do not add cadence selection unless the backend already has a schedule contract for ingestion source cadence.
- Do not make shortcut modal rows display-only; any new Sources shortcut must invoke actual navigation.
- Do not call the Notes action "Mirror folder" in this slice. Use "Sync folder".

## File Map

Backend:
- Create: `tldw_Server_API/app/core/Ingestion_Sources/access_policy.py`
  - Resolve whether the current authenticated user may create or retarget `local_directory` ingestion sources.
  - Single-user mode returns true.
  - Multi-user mode requires an enabled admin feature flag with global, org, or user scope.
- Modify: `tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`
  - Enforce local-directory entitlement on create and identity-changing patch.
  - Add an authenticated `/api/v1/ingestion-sources/capabilities` endpoint before `/{source_id}` routes.
- Test: `tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py`
- Test: `tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py`

Frontend capabilities and routing:
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
  - Add `canCreateLocalDirectoryIngestionSource`.
  - Fetch and merge authenticated ingestion source capabilities when ingestion source routes exist.
  - Bump persisted capabilities cache key/version and include auth scope so user-scoped capabilities do not leak across accounts.
- Modify: `apps/packages/ui/src/hooks/useServerCapabilities.ts`
  - No API shape change expected, but use existing `refresh()` after source creation failures if needed.
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
  - Add a route builder for `/sources/new?preset=notes-folder-sync`.
- Test: `apps/packages/ui/src/services/__tests__/server-capabilities.test.ts`

Sources form:
- Modify: `apps/packages/ui/src/routes/option-sources-new.tsx`
  - Read the `preset` query param and pass it to `SourceForm`.
- Modify: `apps/packages/ui/src/components/Option/Sources/SourceForm.tsx`
  - Support `preset="notes-folder-sync"`.
  - Render a visible `Scheduled rescans` switch mapped to existing `schedule_enabled`.
  - Disable local-directory creation with clear copy when capability is false.
- Test: `apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx`

Notes entry point:
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
  - Navigate to the preset route when the user clicks "Sync folder".
- Modify: `apps/packages/ui/src/components/Notes/NotesSidebar.tsx`
  - Thread the action down to `NotesListPanel`.
- Modify: `apps/packages/ui/src/components/Notes/NotesListPanel.tsx`
  - Add the secondary "Sync folder" action beside Import/Export in active Notes view.
  - Gate by online state, Notes support, ingestion sources support, and `canCreateLocalDirectoryIngestionSource`.
- Test: create `apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx`
- Test: create or extend `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx`

Shortcut config and global navigation:
- Modify: `apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts`
  - Add `modeSources` defaulting to `Alt+2`.
  - Merge persisted shortcuts over `defaultShortcuts` so legacy configs pick up new keys.
- Modify: `apps/packages/ui/src/hooks/keyboard/useKeyboardShortcuts.ts`
  - Add a shared `useModeNavigationShortcuts` hook that maps existing mode shortcuts plus Sources to real navigation callbacks.
- Modify: `apps/packages/ui/src/components/Layouts/Layout.tsx`
  - Initialize mode navigation shortcuts in the extension/options layout.
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
  - Initialize the same mode navigation shortcuts in the WebUI layout.
- Test: create `apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts`
- Test: create `apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx`

Shortcut visibility:
- Modify: `apps/packages/ui/src/components/Common/KeyboardShortcutsModal.tsx`
  - Add "Go to Sources" in Navigation.
- Modify: `apps/packages/ui/src/components/Common/PageHelpModal.tsx`
  - Add "Go to Sources" in Navigation.
- Modify: `apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx`
  - Update mocks and assert Sources appears.
- Test: create `apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx`

Header launcher:
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts`
  - Add `sources` to `HEADER_SHORTCUT_IDS`.
  - Migrate only legacy full-default persisted header selections to include `sources`; keep custom/trimmed selections unchanged except required ids.
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
  - Add Sources to the Library group with the existing Sources route/icon.
- Modify: `apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts`
  - Assert default includes Sources.
  - Assert legacy full-default selection is migrated.
  - Assert custom selection is not forcibly expanded with Sources.
- Modify if needed: `apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx`
  - Update mocks to include `sources`.
- Modify if needed: `apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts`
  - Update expected default ids.

## Task 1: Backend Entitlement and Enforcement

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Sources/access_policy.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py`
- Test: `tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py`
- Test: `tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py`

- [ ] **Step 1: Write access policy unit tests**

Test the policy helper before touching endpoints.

Required cases:
- Single-user mode allows local-directory creation.
- Multi-user mode denies by default.
- Multi-user user-scoped flag allows the exact user.
- Multi-user org-scoped flag allows users whose `active_org_id` or `org_ids` includes the org.
- Disabled flags do not allow.
- `target_user_ids` narrows org/global flags when present.
- Existing admin flag records from `admin_system_ops_service.list_feature_flags()` are enough; do not create a second flag store.

Use a fake user object, not a real database user:

```python
class FakeUser:
    def __init__(self, user_id: int, *, active_org_id: int | None = None, org_ids: list[int] | None = None):
        self.id = user_id
        self.active_org_id = active_org_id
        self.org_ids = org_ids or []
```

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py -q
```

Expected: fails because `access_policy.py` does not exist yet.

- [ ] **Step 2: Implement `access_policy.py`**

Use this shape:

```python
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode
from tldw_Server_API.app.services.admin_system_ops_service import list_feature_flags

LOCAL_DIRECTORY_INGESTION_SOURCE_FLAG_KEY = "ingestion_sources.local_directory"


def can_create_local_directory_ingestion_source(current_user: Any) -> bool:
    if is_single_user_mode():
        return True
    user_id = _user_id(current_user)
    if user_id is None:
        return False
    org_ids = _org_ids(current_user)
    for flag in list_feature_flags():
        if flag.get("key") != LOCAL_DIRECTORY_INGESTION_SOURCE_FLAG_KEY:
            continue
        if _enabled_flag_applies(flag, user_id=user_id, org_ids=org_ids):
            return True
    return False
```

Keep the helper deterministic and side-effect free. If rollout percent exists, hash `f"{flag_key}:{user_id}"` with SHA-256 and bucket into 0-99.

- [ ] **Step 3: Run unit tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py -q
```

Expected: pass.

- [ ] **Step 4: Write endpoint enforcement tests**

Add tests that exercise the FastAPI endpoint dependency path:
- `POST /api/v1/ingestion-sources/` with `source_type="local_directory"` returns `403` in multi-user mode without the flag.
- The same request succeeds when the flag applies to the current user.
- `source_type="archive_snapshot"` and `source_type="git_repository"` are not blocked by this local-directory entitlement.
- `PATCH /api/v1/ingestion-sources/{source_id}` cannot change or retarget a source into `local_directory` without the entitlement.
- `GET /api/v1/ingestion-sources/capabilities` returns `can_create_local_directory: false` without the flag and `true` with an applicable user/org/global flag.

Prefer existing ingestion source fixtures from `tldw_Server_API/tests/Ingestion_Sources/integration/` and monkeypatch the policy helper when full AuthNZ setup makes endpoint tests too broad.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py -q
```

Expected: fails because endpoint enforcement is not implemented.

- [ ] **Step 5: Enforce in `ingestion_sources.py`**

Add a small endpoint-local guard:

```python
def _requires_local_directory_entitlement(
    *,
    source_type: str | None,
    config_changed: bool = True,
) -> bool:
    return config_changed and str(source_type or "").strip().lower() == "local_directory"
```

In `create_ingestion_source`, after payload validation and before database writes:

```python
if prepared_payload.get("source_type") == "local_directory" and not can_create_local_directory_ingestion_source(current_user):
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Local directory ingestion sources are not enabled for this user",
    )
```

In `patch_ingestion_source`, enforce only when `source_type` or `config` changes and the effective type is `local_directory`. Do not block changing enabled/policy/schedule on an existing local-directory source.

- [ ] **Step 6: Add authenticated capabilities endpoint**

Add before `@router.get("/{source_id}")`:

```python
@router.get("/capabilities")
async def get_ingestion_source_capabilities(current_user: User = Depends(get_request_user)):
    return {
        "can_create_local_directory": can_create_local_directory_ingestion_source(current_user),
    }
```

Use auth because this is a current-user entitlement. Do not put this in unauthenticated `/config/docs-info`.

- [ ] **Step 7: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py -q
```

Expected: pass.

- [ ] **Step 8: Commit backend slice**

```bash
git add tldw_Server_API/app/core/Ingestion_Sources/access_policy.py tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py
git commit -m "feat: gate local directory ingestion sources"
```

## Task 2: Frontend Capability Contract

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Modify: `apps/packages/ui/src/services/__tests__/server-capabilities.test.ts`

- [ ] **Step 1: Write capability merge tests**

Add cases:
- OpenAPI advertises ingestion sources, `/ingestion-sources/capabilities` returns `{ can_create_local_directory: true }`, and `getServerCapabilities()` returns `canCreateLocalDirectoryIngestionSource: true`.
- If the authenticated capabilities endpoint fails, keep `hasIngestionSources` true but `canCreateLocalDirectoryIngestionSource` false.
- A cached capability value is scoped by server/auth/user context. Reuse `buildChatSurfaceScopeKeyFromConfig` rather than keying only by server URL and auth mode.

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
```

Expected: fails because the field and fetch do not exist yet.

- [ ] **Step 2: Add capability field and default**

In `ServerCapabilities`, add:

```ts
canCreateLocalDirectoryIngestionSource: boolean
```

Default to `false`.

- [ ] **Step 3: Fetch authenticated ingestion source capabilities**

In `fetchCapabilitiesFromServer`, after computing route capabilities, conditionally call:

```ts
bgRequest<{ can_create_local_directory?: unknown }, any>({
  path: "/api/v1/ingestion-sources/capabilities" as any,
  method: "GET" as any
})
```

Do not pass `noAuth: true`. Parse only boolean-like values. Failure should be non-fatal and should not disable the generic Sources page.

- [ ] **Step 4: Fix the capabilities cache scope**

Update `getCapabilitiesCacheKey()` to reuse:

```ts
buildChatSurfaceScopeKeyFromConfig(cfg)
```

from `apps/packages/ui/src/services/chat-surface-scope.ts`, and bump:

```ts
const CAPABILITIES_STORAGE_KEY = "__tldwServerCapabilitiesCacheV4"
```

This prevents one multi-user account's entitlement from being reused by another account in the same browser profile.

- [ ] **Step 5: Run capability tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
```

Expected: pass.

- [ ] **Step 6: Commit frontend capability slice**

```bash
git add apps/packages/ui/src/services/tldw/server-capabilities.ts apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
git commit -m "feat: expose local directory source capability"
```

## Task 3: Sources Preset and Schedule Switch

**Files:**
- Modify: `apps/packages/ui/src/routes/route-paths.ts`
- Modify: `apps/packages/ui/src/routes/option-sources-new.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourceForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx`

- [ ] **Step 1: Write route builder and SourceForm tests**

Add tests:
- `buildSourcesNewPath({ preset: "notes-folder-sync" })` returns `/sources/new?preset=notes-folder-sync`.
- `SourceForm` with that preset defaults to:
  - `source_type="local_directory"`
  - `sink_type="notes"`
  - `policy="canonical"`
  - `enabled=true`
  - `schedule_enabled=false`
- The rendered form includes a `Scheduled rescans` switch.
- Toggling `Scheduled rescans` sends `schedule_enabled: true`.
- If `canCreateLocalDirectoryIngestionSource` is false, local-directory create is disabled and copy says the administrator must enable server folder sync.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx
```

Expected: fails.

- [ ] **Step 2: Add preset path builder**

In `route-paths.ts`:

```ts
export type SourcesNewPreset = "notes-folder-sync"

export const buildSourcesNewPath = (options: { preset?: SourcesNewPreset } = {}): string => {
  const params = new URLSearchParams()
  if (options.preset) params.set("preset", options.preset)
  const encoded = params.toString()
  return encoded ? `${SOURCES_NEW_PATH}?${encoded}` : SOURCES_NEW_PATH
}
```

- [ ] **Step 3: Thread preset into `SourceForm`**

In `option-sources-new.tsx`, use `useSearchParams()` and pass:

```tsx
<SourceForm mode="create" preset={preset === "notes-folder-sync" ? preset : undefined} />
```

Update `SourceFormProps` with `preset?: "notes-folder-sync"`.

- [ ] **Step 4: Render schedule switch**

Add a visible switch near `Enabled`:

```tsx
<Form.Item
  name="schedule_enabled"
  label={t("sources:form.scheduleEnabled", "Scheduled rescans")}
  valuePropName="checked"
  extra={t("sources:form.scheduleEnabledHelp", "When enabled, the server may rescan this source according to its configured ingestion-source schedule.")}
>
  <Switch />
</Form.Item>
```

Do not add cadence controls.

- [ ] **Step 5: Capability-gate local-directory create**

Use `useServerCapabilities()` in `SourceForm` or pass capability from the route. For `mode="create"` and effective `source_type === "local_directory"`, disable submit when:

```ts
capsLoading || capabilities?.canCreateLocalDirectoryIngestionSource !== true
```

Keep archive and git creation available when the generic Sources feature exists.

- [ ] **Step 6: Run SourceForm tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit Sources form slice**

```bash
git add apps/packages/ui/src/routes/route-paths.ts apps/packages/ui/src/routes/option-sources-new.tsx apps/packages/ui/src/components/Option/Sources/SourceForm.tsx apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx
git commit -m "feat: add notes folder sync source preset"
```

## Task 4: Notes "Sync folder" Entry Point

**Files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesSidebar.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesListPanel.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx`

- [ ] **Step 1: Write NotesListPanel tests**

Test active view rendering and disabled states:
- Online, active Notes view, Notes supported, Sources supported, local-directory entitlement true: button is enabled.
- Trash view: button is hidden or disabled with "Switch to Notes view to sync folders".
- Offline: disabled with "Connect to sync folders".
- Sources unsupported: disabled with "Sources are not available on this server".
- Entitlement false: disabled with "Ask an administrator to enable server folder sync for this account".

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx
```

Expected: fails.

- [ ] **Step 2: Add NotesListPanel props and button**

Add props:

```ts
onSyncFolder?: () => void
syncFolderInProgress?: boolean
```

Compute:

```ts
const canSyncFolder =
  isOnline &&
  !isTrashView &&
  !capsLoading &&
  Boolean(capabilities?.hasNotes) &&
  Boolean(capabilities?.hasIngestionSources) &&
  Boolean(capabilities?.canCreateLocalDirectoryIngestionSource)
```

Render a small text button beside Import:

```tsx
<Button size="small" type="text" className="text-xs" disabled={!canSyncFolder} onClick={() => onSyncFolder?.()}>
  {t("option:notesSearch.syncFolder", { defaultValue: "Sync folder" })}
</Button>
```

Do not use hero copy or explanatory in-app text. Keep detail in tooltip only.

- [ ] **Step 3: Thread the action through NotesSidebar**

Add `onSyncFolder: () => void` to `NotesSidebarProps`, pass it to `NotesListPanel`.

- [ ] **Step 4: Add NotesManager navigation test**

Mock `useNavigate`, set capabilities with:

```ts
{
  hasNotes: true,
  hasIngestionSources: true,
  canCreateLocalDirectoryIngestionSource: true
}
```

Click "Sync folder" and assert:

```ts
expect(navigate).toHaveBeenCalledWith("/sources/new?preset=notes-folder-sync")
```

- [ ] **Step 5: Implement NotesManager navigation**

Import `buildSourcesNewPath` and pass:

```tsx
onSyncFolder={() => navigate(buildSourcesNewPath({ preset: "notes-folder-sync" }))}
```

- [ ] **Step 6: Run Notes tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit Notes entry slice**

```bash
git add apps/packages/ui/src/components/Notes/NotesManagerPage.tsx apps/packages/ui/src/components/Notes/NotesSidebar.tsx apps/packages/ui/src/components/Notes/NotesListPanel.tsx apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx
git commit -m "feat: expose folder sync from notes"
```

## Task 5: Real Sources Navigation Shortcut

**Files:**
- Modify: `apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts`
- Modify: `apps/packages/ui/src/hooks/keyboard/useKeyboardShortcuts.ts`
- Modify: `apps/packages/ui/src/components/Layouts/Layout.tsx`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Test: `apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts`
- Test: `apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx`

- [ ] **Step 1: Write shortcut config tests**

Add tests for:
- `defaultShortcuts.modeSources` is `Alt+2`.
- `mergeShortcutConfig({ modeMedia: custom })` returns defaults plus the custom override.
- Legacy persisted configs without `modeSources` still expose `modeSources`.

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts
```

Expected: fails.

- [ ] **Step 2: Add merge helper and `modeSources`**

In `useShortcutConfig.ts`, add the interface key and default:

```ts
modeSources: {
  key: "2",
  altKey: true,
  preventDefault: true,
  stopPropagation: true
}
```

Export a pure helper:

```ts
export const mergeShortcutConfig = (
  value: Partial<ShortcutConfig> | null | undefined
): ShortcutConfig => ({
  ...defaultShortcuts,
  ...(value || {})
})
```

Return `shortcuts: mergeShortcutConfig(shortcuts)` from the hook, and make `updateShortcut`/`resetShortcut` preserve unknown future keys by merging against previous values.

- [ ] **Step 3: Write navigation shortcut hook tests**

Use React Testing Library `renderHook` or a tiny component. Assert pressing:
- `Alt+2` calls navigation to `/sources`.
- `Alt+5` calls navigation to `/notes`.
- A shortcut does not fire while disabled.

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx
```

Expected: fails.

- [ ] **Step 4: Implement `useModeNavigationShortcuts`**

In `useKeyboardShortcuts.ts`, add:

```ts
export type ModeNavigationTarget = {
  key: keyof Pick<
    ShortcutConfig,
    "modePlayground" | "modeSources" | "modeMedia" | "modeKnowledge" | "modeNotes" | "modePrompts" | "modeFlashcards" | "modeWorldBooks" | "modeDictionaries" | "modeCharacters"
  >
  path: string
  description: string
}
```

The hook should accept:

```ts
useModeNavigationShortcuts(navigate: (path: string) => void, enabled = true)
```

Build shortcuts from `configuredShortcuts`, call `navigate(path)`, and set route transition loading by calling `useRouteTransitionStore.getState().start(path)` before navigation.

Use this path map:
- `modePlayground`: `/chat`
- `modeSources`: `/sources`
- `modeMedia`: `/media`
- `modeKnowledge`: `/knowledge`
- `modeNotes`: `/notes`
- `modePrompts`: `/prompts`
- `modeFlashcards`: `/flashcards`
- `modeWorldBooks`: `/world-books`
- `modeDictionaries`: `/dictionaries`
- `modeCharacters`: `/characters`

- [ ] **Step 5: Initialize in both layouts**

In `Layout.tsx`, import `useNavigate` if needed and initialize:

```ts
useModeNavigationShortcuts(navigate, !hideHeader)
```

In `WebLayout.tsx`, reuse existing `navigate`:

```ts
useModeNavigationShortcuts(navigate, !hideHeader)
```

- [ ] **Step 6: Run shortcut tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit shortcut hook slice**

```bash
git add apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts apps/packages/ui/src/hooks/keyboard/useKeyboardShortcuts.ts apps/packages/ui/src/components/Layouts/Layout.tsx apps/tldw-frontend/components/layout/WebLayout.tsx apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx
git commit -m "feat: wire mode navigation shortcuts"
```

## Task 6: Shortcut Modal and Header Launcher Discoverability

**Files:**
- Modify: `apps/packages/ui/src/components/Common/KeyboardShortcutsModal.tsx`
- Modify: `apps/packages/ui/src/components/Common/PageHelpModal.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx`
- Create: `apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx`
- Modify: `apps/packages/ui/src/services/settings/ui-settings.ts`
- Modify: `apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts`
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify if needed: `apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx`
- Modify if needed: `apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts`

- [ ] **Step 1: Write modal tests**

Update the existing KeyboardShortcutsModal mock default shortcuts to include `modeSources`. Assert "Go to Sources" appears with `Alt + 2`.

Create a PageHelpModal shortcut test that dispatches the open event and asserts "Go to Sources" appears.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx
```

Expected: fails.

- [ ] **Step 2: Add Sources rows**

Add Navigation rows to both modal components immediately after Playground:

```ts
{
  label: t("common:shortcuts.goToSources", "Go to Sources"),
  keys: formatShortcut(defaultShortcuts.modeSources)
}
```

- [ ] **Step 3: Write header launcher tests**

In `ui-settings.header-shortcuts.test.ts`, add:
- default selection contains `sources`.
- a legacy full-default selection missing only `sources` normalizes to include `sources`.
- a custom selection such as `["chat", "notes", "settings"]` does not gain `sources`.

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts
```

Expected: fails.

- [ ] **Step 4: Add `sources` header shortcut id**

Add `"sources"` near `"media"`/`"notes"` in `HEADER_SHORTCUT_IDS`.

Implement legacy full-default detection in `coerceHeaderShortcutSelection`:
- Build `HEADER_SHORTCUT_IDS_WITHOUT_SOURCES`.
- If the incoming normalized unique ids exactly match that full old default set, add `sources`.
- Do not add `sources` to smaller custom selections.

Keep existing `REQUIRED_HEADER_SHORTCUT_IDS` behavior.

- [ ] **Step 5: Add Sources launcher item**

In `header-shortcut-items.ts`, import `SOURCES_PATH` and add this to the Library group near Media/Notes:

```ts
{
  id: "sources",
  to: SOURCES_PATH,
  icon: Layers,
  labelKey: "option:header.sources",
  labelDefault: "Sources",
  descriptionKey: "option:header.sourcesDesc",
  descriptionDefault: "Manage server folders and archive snapshots that sync into notes or media"
}
```

Do not assign a `shortcutIndex`; launcher number shortcuts are already occupied.

- [ ] **Step 6: Run modal and header tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts
```

Expected: pass, or update only tests directly affected by adding `sources`.

- [ ] **Step 7: Commit discoverability slice**

```bash
git add apps/packages/ui/src/components/Common/KeyboardShortcutsModal.tsx apps/packages/ui/src/components/Common/PageHelpModal.tsx apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx apps/packages/ui/src/services/settings/ui-settings.ts apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts
git commit -m "feat: surface sources in shortcuts"
```

## Task 7: Verification, Browser QA, and Security

**Files:**
- No new files expected, but update tests if verification exposes legitimate regressions.

- [ ] **Step 1: Run focused backend tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py tldw_Server_API/tests/Ingestion_Sources/integration/test_local_directory_sync_integration.py -q
```

Expected: pass.

- [ ] **Step 2: Run focused frontend tests**

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/server-capabilities.test.ts apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx apps/packages/ui/src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx apps/packages/ui/src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts
```

Expected: pass.

- [ ] **Step 3: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Ingestion_Sources/access_policy.py tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py -f json -o /tmp/bandit_folder_notes_sources_ui.json
```

Expected: no new high or medium findings in touched code.

- [ ] **Step 4: Browser verify shared UI**

Start the WebUI if it is not already running:

```bash
bun run dev
```

Use the in-app browser or Playwright to verify:
- `/sources` appears in the header launcher under Library.
- `/sources/new?preset=notes-folder-sync` opens SourceForm with Local directory, Notes, Canonical, Enabled on, Scheduled rescans off.
- `/notes` shows "Sync folder" in the list action row when capabilities allow it.
- Clicking "Sync folder" navigates to `/sources/new?preset=notes-folder-sync`.
- Keyboard shortcut modal and PageHelp modal both show "Go to Sources" with `Alt + 2`.
- Pressing `Alt+2` navigates to `/sources`.
- Mobile width does not overflow the Notes action row; if it does, move Sync folder into a compact overflow menu rather than wrapping badly.

- [ ] **Step 5: Check worktree scope**

Run:

```bash
git status --short
git diff --stat
```

Expected: only files from this plan and related Backlog task updates are changed by this work. Unrelated pre-existing dirty files should not be staged or reverted.

- [ ] **Step 6: Final commit**

If verification required fixes, commit them:

```bash
git add <only files touched for this implementation>
git commit -m "test: verify sources notes sync exposure"
```

If no fixes were needed, do not create an empty commit.

## Completion Checklist

- [ ] Backend denies local-directory source creation in multi-user mode unless the admin flag applies.
- [ ] Single-user mode can create local-directory sources without extra admin configuration.
- [ ] Generic `/sources` remains discoverable when `hasIngestionSources` is true.
- [ ] Notes "Sync folder" only enables when the current user can create local-directory sources.
- [ ] The preset route defaults to local-directory -> notes, canonical, enabled, unscheduled.
- [ ] Scheduled rescans switch is visible and maps to `schedule_enabled`.
- [ ] `Alt+2` actually navigates to `/sources` in both WebUI and extension layouts.
- [ ] Shortcut modal and help modal include "Go to Sources".
- [ ] Header shortcut migration preserves custom user selections.
- [ ] Focused pytest, Vitest, Bandit, and browser QA have been run and recorded in the implementation Backlog task.
