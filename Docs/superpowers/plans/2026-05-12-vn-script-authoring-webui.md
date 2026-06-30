# VN Script Authoring WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bundled WebUI surface for creating, editing, validating, publishing, and selecting backend-owned VN scripts.

**Architecture:** Keep script semantics, policy evaluation, manifest readiness, generation-profile checks, and publish validation on the API server. The frontend adds typed clients and a workbench that displays backend responses, edits JSON drafts as opaque script programs, and links published versions into the existing VN Play setup flow through backend setup-options data.

**Tech Stack:** Next.js pages with dynamic imports, React state/hooks, existing `apiClient`, existing `JsonEditor`/`JsonViewer`, Vitest + Testing Library.

---

## File Structure

- Create: `apps/tldw-frontend/types/vn-scripts.ts`
  - TypeScript contract for `/api/v1/vn/vn-scripts` responses and request payloads.
- Create: `apps/tldw-frontend/lib/api/vnScripts.ts`
  - Thin API wrapper with canonical `/vn/vn-scripts` paths and offset-query helpers.
- Create: `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx`
  - Script list, metadata create/update controls, draft JSON editor, validation/diagnostics, publish action, version list, manifest/policy summary.
- Create: `apps/tldw-frontend/pages/vn-scripts.tsx`
  - Dynamic route entrypoint for the workbench.
- Modify: `apps/tldw-frontend/types/vn-play.ts`
  - Add `scripted_story` mode, script setup option/default fields, script-version setup types, and `script_id`/`script_version_id` session create fields.
- Modify: `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`
  - Add Scripted Story mode selection, script-version selector, warning acknowledgement handling, and link to `/vn-scripts` empty-state guidance.
- Modify: `apps/tldw-frontend/components/vn-play/SessionList.tsx`
  - Treat `scripted_story` as a first-class filter/rendered mode without collapsing it into Story/CYOA.
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
  - Allow opening the new-session dialog in scripted mode and link to the authoring workbench from setup empty states or header actions.
- Test: `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`
- Test: extend `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`
- Test: extend `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts` if setup query/session-create coverage needs new scripted fields.
- Modify: `backlog/tasks/task-289 - Add-VN-script-authoring-and-publish-WebUI.md`
  - Track progress, verification, and final summary.

## Task 1: VN Scripts API Client Contract

**Files:**
- Create: `apps/tldw-frontend/types/vn-scripts.ts`
- Create: `apps/tldw-frontend/lib/api/vnScripts.ts`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts`

- [x] **Step 1: Write failing API-wrapper tests**

Cover these calls with mocked `apiClient`:
- `createVNScript`
- `listVNScripts`
- `getVNScript`
- `patchVNScript`
- `deleteVNScript`
- `getVNScriptDraft`
- `putVNScriptDraft`
- `validateVNScriptDraft`
- `getVNScriptDiagnostics`
- `publishVNScript`
- `listVNScriptVersions`
- `getVNScriptVersion`
- `getVNScriptManifestSnapshot`
- `evaluateVNScriptVersionPolicy`

Expected canonical paths:
- `/vn/vn-scripts/scripts`
- `/vn/vn-scripts/scripts/{script_id}/draft`
- `/vn/vn-scripts/scripts/{script_id}/draft/validate`
- `/vn/vn-scripts/scripts/{script_id}/draft/diagnostics`
- `/vn/vn-scripts/scripts/{script_id}/publish`
- `/vn/vn-scripts/scripts/{script_id}/versions`
- `/vn/vn-scripts/scripts/{script_id}/versions/{version_id}`
- `/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/manifest-snapshot`
- `/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/policy/evaluate`

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-scripts/vnScriptsApi.test.ts
```

Expected: fail because the new module does not exist.

- [x] **Step 2: Add `vn-scripts` types**

Mirror backend schema names from `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`:
- `VNScriptCreate`
- `VNScriptPatch`
- `VNScriptResponse`
- `VNScriptListResponse`
- `VNScriptDraftResponse`
- `VNScriptDraftPutRequest`
- `VNScriptValidateRequest`
- `VNScriptValidationResponse`
- `VNScriptDiagnosticsResponse`
- `VNScriptPublishRequest`
- `VNScriptPublishResponse`
- `VNScriptVersionResponse`
- `VNScriptVersionListResponse`
- `VNScriptManifestSnapshotResponse`
- `VNScriptVersionPolicyEvaluateRequest`
- `VNScriptVersionPolicyEvaluateResponse`

Use `Record<string, unknown>` for opaque draft/program/diagnostics/policy payloads. Do not interpret opcode semantics client-side.

- [x] **Step 3: Add `vnScripts.ts` API wrapper**

Follow `apps/tldw-frontend/lib/api/vnPlay.ts` patterns:

```ts
const VN_SCRIPTS_BASE = '/vn/vn-scripts';

export function listVNScripts(query: VNScriptListQuery = {}): Promise<VNScriptListResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/scripts`, { params: toQueryParams(query) });
}
```

Publish must require caller-provided `idempotency_key`; the helper should not generate keys.

- [x] **Step 4: Run API-wrapper tests**

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-scripts/vnScriptsApi.test.ts
```

Expected: pass.

## Task 2: Script Authoring Workbench

**Files:**
- Create: `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx`
- Create: `apps/tldw-frontend/pages/vn-scripts.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`

- [x] **Step 1: Write failing workbench tests**

Use mocked `@web/lib/api/vnScripts` helpers. Cover:
- initial list loads scripts and selects the first item;
- create form calls `createVNScript` with title, asset pack ID, policy profile, generation profile, and content rating;
- draft JSON edits call `putVNScriptDraft` with `if_revision` and parsed JSON;
- invalid local JSON shows an editor error and does not call the backend;
- validate calls `validateVNScriptDraft` and renders backend errors/warnings;
- diagnostics calls `getVNScriptDiagnostics` and renders safe JSON;
- publish calls `publishVNScript` with an idempotency key and current draft revision;
- version list renders version number, label, snapshot IDs, validation status, and created time;
- manifest/policy summary buttons call `getVNScriptManifestSnapshot` and `evaluateVNScriptVersionPolicy`.

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-scripts/VNScriptsWorkbench.test.tsx
```

Expected: fail because the workbench does not exist.

- [x] **Step 2: Implement layout and loading state**

Build a dense app surface, not a marketing page:
- left pane: script list and create form;
- center pane: metadata summary and draft JSON editor;
- right/bottom pane: validation, diagnostics, published versions, manifest/policy summaries.

Use existing `Button`, `Input`, `Badge`, `JsonEditor`, and `JsonViewer`. Keep cards only for repeated items and panels; avoid nested card styling.

- [x] **Step 3: Implement list/create/select**

Load `listVNScripts({ limit: 25, offset: 0 })` on mount. On create success, prepend/select the created script and load its draft/versions. Store errors as user-safe strings.

- [x] **Step 4: Implement draft edit/save**

When a script is selected:
- call `getVNScriptDraft`;
- show formatted JSON in `JsonEditor`;
- track dirty state;
- parse JSON on save;
- call `putVNScriptDraft(script.id, { if_revision: draft.revision, draft: parsed })`;
- on conflict/error, show backend error and keep unsaved editor text.

Do not attempt to validate opcodes locally beyond JSON parse.

- [x] **Step 5: Implement validation, diagnostics, publish, versions**

Validation:
- call `validateVNScriptDraft(script.id, parsed-or-current)`;
- render `valid`, errors, and warnings from backend response.

Diagnostics:
- call `getVNScriptDiagnostics(script.id)`;
- show `diagnostics` through `JsonViewer`.

Publish:
- call `publishVNScript(script.id, { draft_revision, label, idempotency_key, acknowledgements })`;
- generate key with a local helper such as `vn-script-publish-${script.id}-${Date.now()}` plus UUID when available;
- reload versions after success.
- do not derive publish acknowledgement codes by stripping prefixes from validation warnings or by parsing messages. If the backend returns `script_publish_acknowledgement_required` and does not expose raw acknowledgement codes, render that publish error and ask the user to resolve/acknowledge through a backend-supported path in a follow-up. Only send acknowledgement strings that are explicitly exposed by the backend as publish acknowledgement codes.

Versions:
- call `listVNScriptVersions(script.id)`;
- render safe snapshot IDs and validation summary;
- provide buttons for manifest snapshot and policy evaluation.

- [x] **Step 6: Add dynamic route**

Create `apps/tldw-frontend/pages/vn-scripts.tsx`:

```ts
import dynamic from 'next/dynamic';

export default dynamic(() => import('@web/components/vn-scripts/VNScriptsWorkbench'), { ssr: false });
```

- [x] **Step 7: Run workbench tests**

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-scripts/VNScriptsWorkbench.test.tsx
```

Expected: pass.

## Task 3: Scripted Story Setup Bridge

**Files:**
- Modify: `apps/tldw-frontend/types/vn-play.ts`
- Modify: `apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/SessionList.tsx`
- Modify: `apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx`
- Test: extend `apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx`
- Test: extend `apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts`

- [x] **Step 1: Write failing setup bridge tests**

Add tests that:
- setup options with `mode: 'scripted_story'` render published script-version choices;
- no published scripts shows guidance linking to `/vn-scripts`;
- creating a scripted session submits `mode: 'scripted_story'`, `script_id`, `script_version_id`, `vn_asset_pack_id` from the selected script option, `content_rating` from the selected script option, and `primary_character_id` derived from the backend-provided matching asset-pack option for `selectedScript.asset_pack_id`;
- scripted-session acknowledgements submit top-level `acknowledgements: string[]` using only `selectedScript.warning_summary.warnings[].code` values that require acknowledgement, and do not use `settings.setup_acknowledgements`;
- readiness or acknowledgement-required script warnings block submit until acknowledged;
- `SessionList` and workspace mode controls show `Scripted Story` distinctly.

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts
```

Expected: fail because frontend types and setup dialog do not support `scripted_story`.

- [x] **Step 2: Extend VN Play frontend types**

Update:
- `VNPlayMode = 'freeform' | 'story' | 'scripted_story'`
- `VNPlaySessionCreate` with `script_id?: number` and `script_version_id?: number`
- `VNPlaySessionCreate` with top-level `acknowledgements?: string[]`
- `VNPlaySession` with response fields `script_id`, `script_version_id`, `script_manifest_snapshot_id`, `script_policy_snapshot_id`, `script_generation_profile_snapshot_id`, and `script_position`
- `VNPlaySetupScriptVersionOption` matching backend `VNPlaySetupScriptVersionOption`
- `VNPlaySetupDefaults` with script/profile defaults
- `VNPlaySetupOptionsResponse.script_versions`
- Keep `VNPlaySetupOptionsResponse.pagination` aligned with the backend `VNPlaySetupPaginationSet`: it currently includes `characters` and `asset_packs` only. Do not add `pagination.script_versions` unless the backend contract changes in the same PR, which is out of scope for this frontend consumer slice.

Keep unknown profile metadata as plain strings/records; do not embed frontend policy logic.

- [x] **Step 3: Add script selector to `NewSessionDialog`**

When mode is `scripted_story`:
- call setup options with `mode: 'scripted_story'`;
- render script-version selector from `setupOptions.script_versions`;
- choose backend default script when available;
- show script warning messages and readiness status;
- if no scripts exist, show empty-state guidance linking to `/vn-scripts`;
- use the selected script's `asset_pack_id` and match it to `setupOptions.asset_packs.find((pack) => pack.id === selectedScript.asset_pack_id)` to derive the required `primary_character_id`;
- submit `content_rating` from the selected script version option so the create payload matches the published script's policy snapshot context;
- include `script_id` and `script_version_id` in create payload.
- include top-level backend `acknowledgements` only when the selected script option requires them. Use warning code strings from `selectedScript.warning_summary.warnings[].code`; do not send messages, do not strip prefixes, and do not reuse the asset-pack `settings.setup_acknowledgements` object shape for scripted-session policy acknowledgements.

Manual-ID fallback may remain for freeform/story only unless setup options fail; if used for scripted mode, require script/version IDs too.

- [x] **Step 4: Update workspace/session mode UI**

Add a `Scripted Story` launch button or mode option near existing Freeform/Story controls. Update mode labels and filters to avoid treating scripted sessions as generic Story sessions.

- [x] **Step 5: Run setup bridge tests**

Run:

```bash
cd apps/tldw-frontend
bun run test:run __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts
```

Expected: pass.

## Task 4: Closeout, Verification, And Task Updates

**Files:**
- Modify: `backlog/tasks/task-289 - Add-VN-script-authoring-and-publish-WebUI.md`
- Optional modify: `Docs/API/VN.md` only if UI links or setup guidance need a doc note. Do not change backend API semantics.

- [x] **Step 1: Run focused frontend tests**

Run:

```bash
cd apps/tldw-frontend
bun run test:run \
  __tests__/vn-scripts/vnScriptsApi.test.ts \
  __tests__/vn-scripts/VNScriptsWorkbench.test.tsx \
  __tests__/vn-play/VNPlayWorkspace.test.tsx \
  __tests__/vn-play/vnPlayApi.test.ts
```

- [x] **Step 2: Run targeted lint**

Run:

```bash
cd apps/tldw-frontend
bun run lint -- \
  components/vn-scripts/VNScriptsWorkbench.tsx \
  components/vn-play/NewSessionDialog.tsx \
  components/vn-play/SessionList.tsx \
  components/vn-play/VNPlayWorkspace.tsx \
  __tests__/vn-scripts/vnScriptsApi.test.ts \
  __tests__/vn-scripts/VNScriptsWorkbench.test.tsx \
  __tests__/vn-play/VNPlayWorkspace.test.tsx \
  __tests__/vn-play/vnPlayApi.test.ts
```

Existing repo-wide warnings are acceptable only if no new touched-file errors are present.

- [x] **Step 3: Run TypeScript if practical**

Run:

```bash
cd apps/tldw-frontend
bunx tsc --noEmit --pretty false
```

If it fails on known existing `packages/ui` baseline errors, record exact files and confirm no touched-file diagnostics.

- [x] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

- [x] **Step 5: Record Bandit status**

Bandit is not applicable if this remains a frontend-only TypeScript/React/Markdown slice. Record that in TASK-289 notes. If backend Python changes are added, run Bandit on touched Python paths.

- [x] **Step 6: Update TASK-289**

Check acceptance criteria, add verification notes, mark Definition of Done items, and add final summary.

- [x] **Step 7: Commit**

```bash
git add \
  apps/tldw-frontend/types/vn-scripts.ts \
  apps/tldw-frontend/lib/api/vnScripts.ts \
  apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx \
  apps/tldw-frontend/pages/vn-scripts.tsx \
  apps/tldw-frontend/types/vn-play.ts \
  apps/tldw-frontend/components/vn-play/NewSessionDialog.tsx \
  apps/tldw-frontend/components/vn-play/SessionList.tsx \
  apps/tldw-frontend/components/vn-play/VNPlayWorkspace.tsx \
  apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts \
  apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx \
  apps/tldw-frontend/__tests__/vn-play/VNPlayWorkspace.test.tsx \
  apps/tldw-frontend/__tests__/vn-play/vnPlayApi.test.ts \
  "backlog/tasks/task-289 - Add-VN-script-authoring-and-publish-WebUI.md" \
  Docs/superpowers/plans/2026-05-12-vn-script-authoring-webui.md
git commit -m "Add VN script authoring WebUI"
```

## Open Implementation Notes

- If backend setup options expose script warning acknowledgements with a different payload shape than asset-pack warnings, follow the backend response exactly and avoid inventing a client-only acknowledgement model.
- Keep the workbench JSON-first for this sprint. A graph editor or text DSL is explicitly out of scope.
- If `JsonEditor` Monaco behavior is difficult in unit tests, mock it as a textarea in the workbench tests rather than weakening the production component.
