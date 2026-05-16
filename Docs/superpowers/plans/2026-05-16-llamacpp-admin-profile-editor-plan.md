# llama.cpp Admin Saved Profile Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a focused `/admin/llamacpp` profile editor so admins can create, edit, duplicate, and delete durable saved llama.cpp launch profiles from the WebUI.

**Architecture:** Reuse the existing backend-owned profile APIs and `LlamacppProfile` TypeScript contract. Add one focused `LlamacppProfilesPanel` component that owns form state and payload shaping, then wire it into `LlamacppAdminPage` so saves refresh profiles and runtimes without auto-starting or auto-wiring Chat. Keep remote downloads, catalogs, and full advanced routing out of this slice.

**Tech Stack:** React, Ant Design, lucide-react, shared design-system Alert primitive, existing `tldwClient` llama.cpp profile methods, Vitest, Testing Library.

---

## Scope Check

This plan implements the next usability slice after the merged multi-instance
backend, Asset Inventory V2, mmproj launch wiring, metadata, and runtime
visibility work.

In scope:

- profile list/editor panel on `/admin/llamacpp`;
- create profile from discovered GGUF assets;
- optional mmproj selection for vision profiles;
- edit, duplicate, and delete saved profiles;
- payload shaping for known profile fields;
- warnings/errors surfaced inside the panel;
- focused Vitest coverage.

Out of scope:

- remote model downloads or catalog search;
- upload/copy/move model files;
- auto-start after profile save;
- auto-wire Chat after profile save;
- backend API changes unless a current client/server mismatch is discovered;
- full advanced args browser redesign.

## File Structure

Create:

- `apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx`
  - List saved profiles, render compact metadata, open create/edit/duplicate
    form, validate JSON fields, and call parent callbacks.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx`
  - Unit coverage for create/edit/duplicate/delete payloads and JSON validation.

Modify:

- `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
  - Reorder the llama.cpp console to render Assets, then Profiles, then
    Runtime so users move from files to saved services to process state.
  - Add profile action handlers that call existing `tldwClient` methods.
  - Refresh `listLlamacppProfiles()` and `listLlamacppInstances()` after
    profile mutations.
- `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`
  - Mock the new panel and assert page-level profile action routing.
- `backlog/tasks/task-397.8 - Implement-llama.cpp-Admin-saved-profile-editor.md`
  - Record acceptance criteria, notes, verification, and closeout.

Do not modify:

- `tldw_Server_API` backend profile APIs;
- llama.cpp supervisor or process runner behavior;
- `/api/v1/llm/models/metadata`;
- Chat or Knowledge routing.

## Task 1: Profile Panel Component

**Files:**

- Create: `apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx`

- [x] **Step 1: Write failing create/edit/duplicate/delete tests**

Add component tests with local fixtures:

```tsx
const ggufAsset = {
  asset_id: "gguf:toy",
  kind: "gguf" as const,
  display_name: "Toy 7B",
  path: "/models/toy.gguf",
  resolved_path: "/models/toy.gguf",
  identity_basis: "resolved_path" as const,
  source: "models_dir" as const,
  metadata: {},
  capabilities: ["unknown"],
  mmproj_asset_ids: ["mmproj:toy"],
  base_model_asset_ids: [],
  warnings: []
}
const mmprojAsset = {
  ...ggufAsset,
  asset_id: "mmproj:toy",
  kind: "mmproj" as const,
  display_name: "Toy projector",
  path: "/models/mmproj-toy.gguf",
  resolved_path: "/models/mmproj-toy.gguf",
  capabilities: ["vision_projector"],
  mmproj_asset_ids: [],
  base_model_asset_ids: ["gguf:toy"]
}
```

Cover:

- create opens a form, selects model/mode/port, and calls `onCreate()` with
  `name`, `mode`, `model_id`, `host`, `port`, `port_policy`, `enabled`,
  `autostart`, `server_args`, and `tags`;
- edit opens an existing profile and calls `onUpdate(profile_id, payload)`;
- duplicate opens an existing profile as a new profile with a copied name and
  calls `onCreate()` instead of `onUpdate()`;
- delete requires confirmation and calls `onDelete(profile_id)`;
- invalid server args JSON shows an inline error and does not call a mutation.

- [x] **Step 2: Run panel tests to verify failure**

Run:

```bash
./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx
```

Expected: FAIL because `LlamacppProfilesPanel` does not exist.

- [x] **Step 3: Implement `LlamacppProfilesPanel`**

Implement props:

```ts
interface LlamacppProfilesPanelProps {
  profiles: LlamacppProfile[]
  assets: LlamacppAssetsResponse | null
  loading?: boolean
  savingProfileId?: string | null
  error?: string | null
  onRefresh: () => void
  onCreate: (payload: LlamacppProfileCreateRequest) => Promise<boolean> | boolean
  onUpdate: (
    profileId: string,
    payload: LlamacppProfileUpdateRequest
  ) => Promise<boolean> | boolean
  onDelete: (profileId: string) => Promise<boolean> | boolean
}
```

Use a `Card` titled `Profiles`, a `List` for existing profiles, and a `Modal`
for the form.

Form fields:

- name: `Input`
- enabled: `Switch`
- mode: `Select` for `chat`, `vision`, `embedding`, `rerank`, `server_generic`
- model: `Select` populated from `assets.assets.filter(kind === "gguf")`
- mmproj: optional `Select` populated from `assets.assets.filter(kind === "mmproj")`
- host: `Input`
- port: `InputNumber`
- port policy: `Select` for `explicit` and `autoselect`
- autostart: `Switch`
- provider alias: `Input`
- tags: comma-separated `Input`
- server args: JSON `TextArea`

Payload rules:

- trim strings before submit;
- `profile_id` is omitted for create unless the form later adds an explicit ID;
- empty `mmproj_model_id` and `provider_alias` become `null`;
- tags split on commas, trimmed, and empty values removed;
- `server_args` must parse to an object, otherwise show `Invalid server args JSON.`;
- duplicate mode appends ` copy` to the profile name and clears no runtime state.

Do not auto-start, auto-stop, or auto-wire Chat from this panel.

- [x] **Step 4: Run panel tests**

Run:

```bash
./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx
```

Expected: PASS.

- [x] **Step 5: Commit panel component**

```bash
git add \
  apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx
git commit -m "feat: add llama.cpp profile editor panel"
```

## Task 2: Admin Page Wiring

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`

- [x] **Step 1: Write failing page wiring tests**

In `LlamacppAdminPage.test.tsx`:

- add `createLlamacppProfile`, `updateLlamacppProfile`, and
  `deleteLlamacppProfile` to `apiMock`;
- mock `LlamacppProfilesPanel` with buttons that call `onCreate`, `onUpdate`,
  and `onDelete`;
- assert create calls `tldwClient.createLlamacppProfile(payload)`;
- assert edit calls `tldwClient.updateLlamacppProfile(profile_id, payload)`;
- assert delete calls `tldwClient.deleteLlamacppProfile(profile_id)`;
- assert each successful mutation reloads profile/runtime data without calling
  start or use-in-chat methods.

- [x] **Step 2: Run page wiring tests to verify failure**

Run:

```bash
./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL because the page does not render or wire a profiles panel yet.

- [x] **Step 3: Wire panel into `LlamacppAdminPage`**

Add import:

```ts
import { LlamacppProfilesPanel } from "./LlamacppProfilesPanel"
```

Add state:

```ts
const [profileActionId, setProfileActionId] = React.useState<string | null>(null)
const [profileError, setProfileError] = React.useState<string | null>(null)
```

Add helper:

```ts
const refreshProfilesAndRuntimes = React.useCallback(
  () => loadRuntimePlane(),
  [loadRuntimePlane]
)
```

Add handlers:

```ts
const handleCreateProfile = async (payload: LlamacppProfileCreateRequest) => {
  try {
    setProfileActionId("__create__")
    setProfileError(null)
    await tldwClient.createLlamacppProfile(payload)
    await refreshProfilesAndRuntimes()
    return true
  } catch (error: unknown) {
    setProfileError(sanitizeAdminErrorMessage(error, "Failed to create llama.cpp profile."))
    markAdminGuardFromError(error)
    return false
  } finally {
    setProfileActionId(null)
  }
}
```

Repeat the same pattern for update/delete. Use `profileId` as the action ID for
update/delete.

Render:

```tsx
<LlamacppProfilesPanel
  profiles={runtimeProfiles}
  assets={assets}
  loading={loadingRuntimes}
  savingProfileId={profileActionId}
  error={profileError}
  onRefresh={loadRuntimePlane}
  onCreate={handleCreateProfile}
  onUpdate={handleUpdateProfile}
  onDelete={handleDeleteProfile}
/>
```

Move `LlamacppRuntimePanel` below `LlamacppProfilesPanel` so the page reads
Readiness → Assets → Profiles → Runtime → legacy Inventory/Launch. Keep the
existing Runtime panel behavior unchanged.

- [x] **Step 4: Run page wiring tests**

Run:

```bash
./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [x] **Step 5: Run combined frontend tests**

Run:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [x] **Step 6: Commit page wiring**

```bash
git add \
  apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
git commit -m "feat: wire llama.cpp profile editor into admin"
```

## Task 3: Verification And Closeout

**Files:**

- Modify: `backlog/tasks/task-397.8 - Implement-llama.cpp-Admin-saved-profile-editor.md`

- [x] **Step 1: Run focused Admin llama.cpp tests**

Run:

```bash
./node_modules/.bin/vitest run \
  src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [x] **Step 2: Run touched-scope TypeScript check if feasible**

Run:

```bash
./node_modules/.bin/tsc -p tsconfig.json --noEmit --pretty false
```

Expected: The package may still fail on known repo-wide baseline type debt. If
it fails, confirm whether any errors reference the touched llama.cpp files and
record that distinction in the Backlog task.

- [x] **Step 3: Run diff checks**

From repo root:

```bash
git diff --check
git diff --check origin/dev...HEAD
```

Expected: PASS.

- [x] **Step 4: Record Bandit disposition**

No Python files should be touched in this UI-only slice. Record Bandit as not
applicable in the Backlog task. If backend files become necessary, run Bandit
on the touched Python paths before closeout.

- [x] **Step 5: Update Backlog task**

Use Backlog MCP to mark `TASK-397.8` Done, fill acceptance criteria, append
verification notes, and add a final summary.

- [x] **Step 6: Commit closeout notes**

```bash
git add "backlog/tasks/task-397.8 - Implement-llama.cpp-Admin-saved-profile-editor.md"
git commit -m "docs: close out llama.cpp profile editor task"
```

- [ ] **Step 7: Prepare PR**

Check status and diff:

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
```

Then push and open a draft PR against `dev` with a human-owned `Change summary`
placeholder.
